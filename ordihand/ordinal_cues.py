"""Per-frame front/back order of the two hands from two monocular cues.

Two readouts are computed on each frame and combined by a confidence-weighted vote with abstention:

* relative depth: the median relative depth of each hand inside its 2D keypoint hull, with the region where
  the two hulls overlap removed, normalised by the 5-95 percentile spread of both hands;
* apparent hand size: the log ratio of the two hull areas (the larger hand is taken to be in front).

Every function returns ``(sign, conf, value)``:
``sign = +1`` means the signer's LEFT hand is in front, ``-1`` the RIGHT hand, ``0`` abstain;
``conf`` is in [0, 1] (0 when abstaining); ``value`` is the underlying readout.

Inputs
------
depth : (H, W) float array of relative depth in which a LARGER value is CLOSER to the camera
        (the inverse-depth output of Video Depth Anything has this convention).
kp    : (K, 2) array of whole-body 2D keypoints in the pixel frame of ``depth``; the last 42 rows are the
        21 left-hand and then the 21 right-hand keypoints (COCO-WholeBody order, as produced by DWPose).
"""
import cv2
import numpy as np

# constants used in the paper
TAU = 0.02             # relative-depth readout: |delta| below this abstains
CONF_FULL = 0.10       # relative-depth readout: |delta| at which confidence reaches 1
DILATE_PX = 8          # hull dilation in pixels
OVERLAP_EXCL_PX = 4    # extra dilation of the hull overlap that is removed from both hands
MIN_PIX = 50           # minimum support pixels per hand
SIZE_ABSTAIN = 0.01    # apparent-size readout: |log area ratio| at or below this abstains
SIZE_CONF_FULL = 0.2   # apparent-size readout: |log area ratio| at which confidence reaches 1


def hull_mask(pts, hw, dil):
    """Filled convex hull of ``pts`` in an image of shape ``hw``, dilated by ``dil`` pixels."""
    m = np.zeros(hw, np.uint8)
    p = np.round(pts).astype(np.int32)
    if len(p) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(p), 1)
    if dil > 0:
        m = cv2.dilate(m, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    return m.astype(bool)


def depth_cue(depth, kp):
    """Relative-depth readout -> (sign, conf, delta); sign = +1: the LEFT hand is closer."""
    hw = depth.shape
    mL, mR = hull_mask(kp[-42:-21], hw, DILATE_PX), hull_mask(kp[-21:], hw, DILATE_PX)
    ov = mL & mR
    if ov.any():
        ov = cv2.dilate(ov.astype(np.uint8), np.ones((2 * OVERLAP_EXCL_PX + 1,) * 2, np.uint8)).astype(bool)
    sL, sR = mL & ~ov, mR & ~ov
    if min(int(sL.sum()), int(sR.sum())) < MIN_PIX:
        return 0.0, 0.0, float("nan")
    dL, dR = depth[sL], depth[sR]
    both = np.concatenate([dL, dR])
    scale = float(np.percentile(both, 95) - np.percentile(both, 5))
    if scale <= 1e-6:
        return 0.0, 0.0, float("nan")
    delta = float((np.median(dL) - np.median(dR)) / scale)
    if abs(delta) < TAU:
        return 0.0, 0.0, delta
    return (1.0 if delta > 0 else -1.0), float(min(1.0, abs(delta) / CONF_FULL)), delta


def size_cue(kp, hw):
    """Apparent-size readout -> (sign, conf, log_ratio); sign = +1: the LEFT hull is larger (predicted in front)."""
    aL = float(hull_mask(kp[-42:-21], hw, DILATE_PX).sum())
    aR = float(hull_mask(kp[-21:], hw, DILATE_PX).sum())
    lr = float(np.log(max(aL, 1.0) / max(aR, 1.0)))
    if abs(lr) <= SIZE_ABSTAIN:
        return 0.0, 0.0, lr
    return (1.0 if lr > 0 else -1.0), float(min(1.0, abs(lr) / SIZE_CONF_FULL)), lr


def fuse(depth_vote, size_vote):
    """Confidence-weighted vote v = c_d * s_d + c_s * s_s -> (sign(v), min(1, |v|), v); v == 0 abstains."""
    v = depth_vote[0] * depth_vote[1] + size_vote[0] * size_vote[1]
    if v == 0.0:
        return 0.0, 0.0, v
    return (1.0 if v > 0 else -1.0), float(min(1.0, abs(v))), v


def hand_order(depth, kp):
    """Fused front/back order of the two hands on one frame -> (sign, conf, vote)."""
    return fuse(depth_cue(depth, kp), size_cue(kp, depth.shape))
