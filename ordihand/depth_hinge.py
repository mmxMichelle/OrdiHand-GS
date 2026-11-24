"""Hand-level depth hinge that imposes a per-frame front/back order on a fitted body mesh.

For every frame b with an order sign s_b (+1: the LEFT hand is in front) and confidence c_b, the median camera
depth of each hand's vertices is compared and

    L_order = sum_b c_b * relu(m + s_b * (zL_b - zR_b)) / sum_b c_b

Larger z is farther from the camera, so the term is zero once the hand that the cue puts in front is closer
by at least the margin m. In the paper the term is added with weight LAMBDA_ORDER to the objective of the
expressive-body fit while the fit is resumed from its original solution.
"""
import torch

LAMBDA_ORDER = 100.0   # weight of the term in the fitting objective
MARGIN = 0.03          # margin in the body model's units


def hand_order_hinge(proj_vertices, idx_left, idx_right, sign, conf, margin=MARGIN):
    """
    proj_vertices : (B, V, 3) vertices in camera space, z = depth (larger = farther)
    idx_left, idx_right : 1D long tensors with the vertex indices of the left and right hand
    sign, conf : (B,) float tensors from ``ordihand.ordinal_cues`` (frames with conf 0 are ignored)
    returns the scalar loss and a dict of statistics
    """
    z = proj_vertices[..., 2]
    dev = z.device
    idx_left, idx_right = idx_left.to(dev), idx_right.to(dev)
    sign, conf = sign.to(dev, z.dtype), conf.to(dev, z.dtype)
    zl = z[:, idx_left].median(dim=1).values
    zr = z[:, idx_right].median(dim=1).values
    signed = sign * (zl - zr)                 # > 0 <=> the mesh order contradicts the cue
    per = torch.relu(margin + signed)
    loss = (conf * per).sum() / (conf.sum() + 1e-8)
    with torch.no_grad():
        active = conf > 0
        stats = dict(n_active=int(active.sum()), n_wrong=int(((signed > 0) & active).sum()),
                     n_margin_violated=int(((per > 0) & active).sum()), loss=float(loss))
    return loss, stats
