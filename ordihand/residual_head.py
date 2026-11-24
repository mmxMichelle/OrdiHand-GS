"""Hand residual head for UV-space Gaussians.

GUAVA predicts, for every Gaussian on the SMPL-X UV map, a local position offset relative to the triangle it is
bound to. OrdiHand-GS adds a small convolutional branch on the same UV feature map that predicts a residual
offset, and applies it only in proportion to how much each Gaussian belongs to the hands:

    p_local = p_base + m_hand * delta_p_hand

m_hand is the barycentric interpolation of a binary per-vertex hand mask over the Gaussian's triangle, so it is
continuous in [0, 1] near the wrist. The last layer is zero-initialised, so the branch starts as an identity
change. In the paper the GUAVA backbone is frozen and only this branch (221,571 parameters) is trained.
"""
import torch
import torch.nn as nn


class HandResidualHead(nn.Module):
    """Predicts a (B, 3, H, W) local-position residual from a (B, C, H, W) UV Gaussian feature map."""

    def __init__(self, feat_dim=128, hid_dim=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, hid_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim, out_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, uv_features):
        return self.net(uv_features)


def hand_ownership(binding_vertices, face_bary, hand_vertex_mask):
    """
    binding_vertices : (N, 3) long, the three mesh vertices of the triangle each UV Gaussian is bound to
    face_bary        : (N, 3) barycentric coordinates of each Gaussian in its triangle
    hand_vertex_mask : (V,) 1 for hand vertices, 0 otherwise (vertices beyond the mask, e.g. added
                       auxiliary vertices, are treated as non-hand)
    returns (N,) hand ownership in [0, 1]
    """
    num_vertices = int(binding_vertices.max().item()) + 1
    mask = hand_vertex_mask.float()
    if num_vertices > mask.shape[0]:
        mask = torch.cat([mask, mask.new_zeros(num_vertices - mask.shape[0])])
    return (mask[binding_vertices.long()] * face_bary).sum(dim=-1)


def apply_hand_residual(local_pos, residual, ownership):
    """local_pos, residual: (B, N, 3); ownership: (N,) -> (B, N, 3) corrected local positions."""
    return local_pos + ownership[None, :, None].to(local_pos) * residual
