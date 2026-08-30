"""Projection heads: the small networks the contrastive losses are computed
on, kept separate from the encoder so the encoder's own (pooled/dense)
output - what gets frozen and reused downstream - isn't the thing directly
optimized against the pretext task's augmentation-invariance pressure.

Design (see the architecture discussion in `progress_so_far.md` Section 2):
a 2-layer nonlinear MLP, matching standard MoCo v2 / SimCLR practice, with
the output L2-normalized so cosine similarity (what the contrastive losses
compare with) is well-defined. `GlobalHead` runs on the encoder's pooled
vector; `DenseHead` runs the same MLP independently at every grid cell of
the encoder's dense feature map, via a 1x1 convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM = 256       # must match Encoder.out_channels / feature_dim
HIDDEN_DIM = 256
PROJECTION_DIM = 128  # MoCo/SimCLR convention - tunable, not load-bearing


class GlobalHead(nn.Module):
    """Pooled (batch, IN_DIM) encoder vector -> L2-normalized (batch, PROJECTION_DIM)."""

    def __init__(self, in_dim: int = IN_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = PROJECTION_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        if pooled_features.ndim != 2:
            raise ValueError(f"Expected shape (batch, {IN_DIM}), got {tuple(pooled_features.shape)}")
        projected = self.mlp(pooled_features)
        return F.normalize(projected, dim=1)


class DenseHead(nn.Module):
    """Dense (batch, IN_DIM, H, W) encoder grid -> L2-normalized (batch, PROJECTION_DIM, H, W).

    Implemented as two 1x1 convolutions (equivalent to applying the same
    small MLP independently to every grid cell's channel vector - a 1x1
    conv's weights are shared across spatial positions, so this is not a
    different network per cell, just the same one applied everywhere).
    """

    def __init__(self, in_channels: int = IN_DIM, hidden_channels: int = HIDDEN_DIM, out_channels: int = PROJECTION_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, dense_features: torch.Tensor) -> torch.Tensor:
        if dense_features.ndim != 4:
            raise ValueError(f"Expected shape (batch, {IN_DIM}, H, W), got {tuple(dense_features.shape)}")
        projected = self.mlp(dense_features)
        return F.normalize(projected, dim=1)  # normalize each spatial cell's channel vector


if __name__ == "__main__":
    torch.manual_seed(0)

    global_head = GlobalHead()
    dense_head = DenseHead()

    pooled = torch.randn(4, IN_DIM)
    grid = torch.randn(4, IN_DIM, 32, 32)

    global_out = global_head(pooled)
    dense_out = dense_head(grid)

    print(f"Global head: {tuple(pooled.shape)} -> {tuple(global_out.shape)}")
    print(f"  L2 norm per row (expected 1.0): {global_out.norm(dim=1)[:4].tolist()}")

    print(f"Dense head:  {tuple(grid.shape)} -> {tuple(dense_out.shape)}")
    cell_norms = dense_out.norm(dim=1)  # (batch, H, W)
    print(f"  L2 norm at cell (0,0,0) and (0,16,16) (expected ~1.0): "
          f"{cell_norms[0, 0, 0].item():.4f}, {cell_norms[0, 16, 16].item():.4f}")

    # End-to-end with the real encoder + momentum twins.
    from encoder import Encoder
    from momentum import EMAModule, MomentumEncoder

    online_encoder = Encoder()
    momentum_encoder = MomentumEncoder(online_encoder)
    momentum_global_head = EMAModule(global_head)
    momentum_dense_head = EMAModule(dense_head)

    dummy = torch.zeros(2, 1, 256, 256)
    online_pooled = online_encoder(dummy, pool=True)
    online_dense = online_encoder(dummy, pool=False)
    momentum_pooled = momentum_encoder(dummy, pool=True)
    momentum_dense = momentum_encoder(dummy, pool=False)

    z_a_global = global_head(online_pooled)
    z_b_global = momentum_global_head(momentum_pooled)
    z_a_dense = dense_head(online_dense)
    z_b_dense = momentum_dense_head(momentum_dense)

    print("\nFull online + momentum path:")
    print(f"  z_a_global: {tuple(z_a_global.shape)}, requires_grad={z_a_global.requires_grad}")
    print(f"  z_b_global: {tuple(z_b_global.shape)}, requires_grad={z_b_global.requires_grad}")
    print(f"  z_a_dense:  {tuple(z_a_dense.shape)}, requires_grad={z_a_dense.requires_grad}")
    print(f"  z_b_dense:  {tuple(z_b_dense.shape)}, requires_grad={z_b_dense.requires_grad}")
