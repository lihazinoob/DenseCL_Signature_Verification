"""Momentum (EMA) twin of `Encoder`, MoCo-style.

The online encoder is trained by backprop as usual. The momentum encoder is
a separate copy that never receives gradients - after each training step,
its weights are nudged slightly toward the online encoder's current weights
via an exponential moving average (EMA):

    momentum_weight <- momentum * momentum_weight + (1 - momentum) * online_weight

With `momentum` close to 1 (default 0.999), the momentum encoder changes
very slowly step to step, which is what makes it useful as a stable target
for View B's features and for the vectors pushed into the memory queue -
see `progress_so_far.md` Section 2 / the architecture discussion for why a
single, fast-changing encoder for both views would give the contrastive
loss a moving target.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from encoder import Encoder

DEFAULT_MOMENTUM = 0.999


class MomentumEncoder(nn.Module):
    """Wraps a frozen deep copy of an `Encoder`, updated only via EMA."""

    def __init__(self, online_encoder: Encoder) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.feature_dim = self.encoder.feature_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        return self.encoder(x, pool=pool)

    @torch.no_grad()
    def update(self, online_encoder: Encoder, momentum: float = DEFAULT_MOMENTUM) -> None:
        """EMA-update this encoder's weights toward `online_encoder`'s current weights.

        Call this once per training step, after the optimizer step on
        `online_encoder`. Trainable parameters are EMA-updated; BatchNorm
        running statistics (buffers) are copied directly rather than
        EMA-blended - they already track a running average internally, so
        blending them again would just double-smooth the same statistic.
        """
        for param_m, param_o in zip(self.encoder.parameters(), online_encoder.parameters()):
            param_m.data.mul_(momentum).add_(param_o.data, alpha=1.0 - momentum)
        for buffer_m, buffer_o in zip(self.encoder.buffers(), online_encoder.buffers()):
            buffer_m.data.copy_(buffer_o.data)


class EMAModule(nn.Module):
    """Generic momentum (EMA) twin of any `nn.Module` - same mechanism as
    `MomentumEncoder` above, generalized so `heads.py`'s `GlobalHead` and
    `DenseHead` (which take no `pool` argument, unlike `Encoder`) can reuse
    it instead of duplicating the EMA update math a second and third time.
    `MomentumEncoder` is kept as its own class rather than refactored onto
    this, since it's already verified working and its `pool`-forwarding
    `forward` signature is encoder-specific.
    """

    def __init__(self, online_module: nn.Module) -> None:
        super().__init__()
        self.module = copy.deepcopy(online_module)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def update(self, online_module: nn.Module, momentum: float = DEFAULT_MOMENTUM) -> None:
        for param_m, param_o in zip(self.module.parameters(), online_module.parameters()):
            param_m.data.mul_(momentum).add_(param_o.data, alpha=1.0 - momentum)
        for buffer_m, buffer_o in zip(self.module.buffers(), online_module.buffers()):
            buffer_m.data.copy_(buffer_o.data)


if __name__ == "__main__":
    online = Encoder()
    momentum_encoder = MomentumEncoder(online)

    trainable = sum(p.requires_grad for p in momentum_encoder.parameters())
    print(f"Momentum encoder trainable parameters: {trainable} (expected 0)")

    # Snapshot one weight, perturb the online encoder (simulating an
    # optimizer step), then confirm the EMA update moves the momentum
    # encoder's weight partway toward it - and not all the way, since
    # momentum < 1.
    before = momentum_encoder.encoder.stage1.blocks[0].conv1.weight.clone()
    with torch.no_grad():
        online.stage1.blocks[0].conv1.weight.add_(1.0)  # large perturbation for a visible effect
    momentum_encoder.update(online, momentum=0.9)
    after = momentum_encoder.encoder.stage1.blocks[0].conv1.weight

    max_shift = (after - before).abs().max().item()
    print(f"Max weight shift after one EMA update (momentum=0.9): {max_shift:.4f} (expected ~0.1, i.e. (1-momentum)*1.0)")

    dummy = torch.zeros(2, 1, 256, 256)
    dense_out = momentum_encoder(dummy, pool=False)
    pooled_out = momentum_encoder(dummy, pool=True)
    print(f"Momentum encoder dense output shape:  {tuple(dense_out.shape)}")
    print(f"Momentum encoder pooled output shape: {tuple(pooled_out.shape)}")
    print(f"Output requires_grad: {dense_out.requires_grad} (expected False)")
