"""Sanity checks on synthetic inputs. Run with: python -m pytest tests (or python tests/test_core.py)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ordihand.depth_hinge import hand_order_hinge  # noqa: E402
from ordihand.ordinal_cues import depth_cue, fuse, hand_order, size_cue  # noqa: E402
from ordihand.residual_head import HandResidualHead, apply_hand_residual, hand_ownership  # noqa: E402


def _hand(cx, cy, r, n=21):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _keypoints(left, right):
    return np.concatenate([np.zeros((91, 2)), left, right])


def test_cues():
    hw = (256, 256)
    kp = _keypoints(_hand(80, 128, 30), _hand(180, 128, 20))   # left hull larger
    depth = np.zeros(hw, np.float32)
    depth[:, :128] = 1.0                                       # left half closer (larger = closer)
    s, c, _ = depth_cue(depth, kp)
    assert s == 1.0 and c == 1.0
    s, c, _ = size_cue(kp, hw)
    assert s == 1.0 and 0 < c <= 1
    assert hand_order(depth, kp)[0] == 1.0
    assert fuse((1.0, 0.5, 0), (-1.0, 0.5, 0))[0] == 0.0     # equal and opposite votes abstain
    assert depth_cue(np.ones(hw, np.float32), kp)[0] == 0.0  # flat depth abstains


def test_hinge():
    v = torch.zeros(2, 4, 3)
    v[0, :2, 2], v[0, 2:, 2] = 1.0, 2.0       # frame 0: left closer
    v[1, :2, 2], v[1, 2:, 2] = 2.0, 1.0       # frame 1: right closer
    il, ir = torch.tensor([0, 1]), torch.tensor([2, 3])
    loss, st = hand_order_hinge(v, il, ir, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert st["n_wrong"] == 1 and abs(float(loss) - (0.0 + 1.03) / 2) < 1e-6
    loss, _ = hand_order_hinge(v, il, ir, torch.tensor([1.0, -1.0]), torch.tensor([1.0, 1.0]))
    assert float(loss) == 0.0


def test_residual_head():
    head = HandResidualHead()
    assert sum(p.numel() for p in head.parameters()) == 221571
    out = head(torch.randn(1, 128, 16, 16))
    assert out.shape == (1, 3, 16, 16) and torch.count_nonzero(out) == 0   # zero-initialised
    own = hand_ownership(torch.tensor([[0, 1, 2], [2, 3, 4]]), torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]),
                         torch.tensor([1, 1, 0, 0]))
    assert torch.allclose(own, torch.tensor([0.5, 0.0]))
    p = apply_hand_residual(torch.zeros(1, 2, 3), torch.ones(1, 2, 3), own)
    assert torch.allclose(p[0, :, 0], torch.tensor([0.5, 0.0]))


if __name__ == "__main__":
    test_cues(); test_hinge(); test_residual_head()
    print("all tests passed")
