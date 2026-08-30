"""ResNet-style backbone - faithful port of the original thesis encoder.

Ported from `Thesis_Final/ssl_pretraining/models/Encoder.py` and
`Thesis_Final/ssl_pretraining/models/encoder/*.py` (the code that actually
produced the report's numbers - `SelfSupervisedNetwork.__init__` there
instantiates `Encoder(norm_type=norm_type)` with every other argument left
at its default, so the defaults below are not a guess, they're what was
used). This replaces an earlier version of this file that reconstructed the
architecture from the report's prose description alone (stage widths and
downsample placement only - it did not specify a stem or block layout), an
assumption flagged at the time and now resolved by reading the real source.

Two differences from that earlier guess, now corrected:
  - There is a dedicated 2-layer stem (plain `ConvNormAct`, no residual
    connection) that expands 1 -> 32 -> 32 channels before stage 1, not
    folded into stage 1 itself.
  - Stages 2-4 are each exactly one `ProjectionResidualBlock` (the
    stride-2 downsampling block) followed by `num_identity_blocks`
    `IdentityResidualBlock`s (1 by default) - not a generic "first block
    strided, rest identical" pattern with an arbitrary block count.

Architecture, in order: stem (1->32->32, no downsample) -> stage1 (2
identity blocks @ 32, no downsample) -> stage2 (32->64, downsample) ->
stage3 (64->128, downsample) -> stage4 (128->256, downsample). Three
downsamples total, matching report Fig 3.3 (256x256 input -> 32x32x256
dense feature grid), and the same 32x32 resolution `mask.py`'s foreground
grid is computed at.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TARGET_SIZE = 256  # must match preprocess.TARGET_SIZE for the 32x32 grid to hold


def build_norm_layer(num_features: int, norm_type: str = "batch") -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvNormAct(nn.Module):
    """Conv -> norm -> activation. Used only by the stem (no residual connection)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        norm_type: str = "batch",
        activation_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            build_norm_layer(out_channels, norm_type=norm_type),
            activation_layer(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SignatureStem(nn.Module):
    """Two plain ConvNormAct layers, no downsample: 1 -> 32 -> 32 channels by default."""

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: tuple[int, ...] = (32, 32),
        norm_type: str = "batch",
        activation_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        blocks = []
        current_in = in_channels
        for current_out in stem_channels:
            blocks.append(
                ConvNormAct(current_in, current_out, kernel_size=3, stride=1,
                             norm_type=norm_type, activation_layer=activation_layer)
            )
            current_in = current_out

        self.layers = nn.Sequential(*blocks)
        self.out_channels = stem_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class IdentityResidualBlock(nn.Module):
    """Two 3x3 convs at a fixed channel count, plain identity skip (no projection)."""

    def __init__(self, in_channels: int, norm_type: str = "batch", activation_layer: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = build_norm_layer(in_channels, norm_type=norm_type)
        self.act1 = activation_layer(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = build_norm_layer(in_channels, norm_type=norm_type)

        self.out_act = activation_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.out_act(out + residual)


class ProjectionResidualBlock(nn.Module):
    """Stride-2, channel-changing residual block: 1x1-conv shortcut, two 3x3 convs on the main path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_type: str = "batch",
        activation_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = build_norm_layer(out_channels, norm_type=norm_type)
        self.act1 = activation_layer(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = build_norm_layer(out_channels, norm_type=norm_type)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            build_norm_layer(out_channels, norm_type=norm_type),
        )
        self.out_act = activation_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.out_act(out + residual)


class ResidualStage(nn.Module):
    """Stage 1: `num_blocks` IdentityResidualBlocks at a fixed channel count, no downsample."""

    def __init__(self, channels: int, num_blocks: int = 2, norm_type: str = "batch", activation_layer: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            IdentityResidualBlock(channels, norm_type=norm_type, activation_layer=activation_layer)
            for _ in range(num_blocks)
        ])
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TransitionResidualStage(nn.Module):
    """Stages 2-4: one ProjectionResidualBlock (downsample + channel change) then `num_identity_blocks` IdentityResidualBlocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        num_identity_blocks: int = 1,
        norm_type: str = "batch",
        activation_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        blocks = [ProjectionResidualBlock(in_channels, out_channels, stride=stride,
                                           norm_type=norm_type, activation_layer=activation_layer)]
        for _ in range(num_identity_blocks):
            blocks.append(IdentityResidualBlock(out_channels, norm_type=norm_type, activation_layer=activation_layer))

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Encoder(nn.Module):
    """The full backbone: stem -> stage1 -> stage2 -> stage3 -> stage4.

    `forward(x, pool)`: `pool=False` returns the dense `(batch, 256, 32, 32)`
    feature grid (what the DenseCL dense head and the foreground masks need);
    `pool=True` returns a globally-averaged `(batch, 256)` vector (what a
    global head - or the existing thesis's frozen-encoder downstream stage -
    needs). Matches the original `Encoder.forward` interface exactly, so
    this module is a drop-in match for how the thesis's own code calls it.
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: tuple[int, ...] = (32, 32),
        stage1_blocks: int = 2,
        stage2_out_channels: int = 64,
        stage2_identity_blocks: int = 1,
        stage3_out_channels: int = 128,
        stage3_identity_blocks: int = 1,
        stage4_out_channels: int = 256,
        stage4_identity_blocks: int = 1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()

        self.stem = SignatureStem(in_channels=in_channels, stem_channels=stem_channels, norm_type=norm_type)

        self.stage1 = ResidualStage(channels=self.stem.out_channels, num_blocks=stage1_blocks, norm_type=norm_type)

        self.stage2 = TransitionResidualStage(
            in_channels=self.stage1.out_channels, out_channels=stage2_out_channels,
            stride=2, num_identity_blocks=stage2_identity_blocks, norm_type=norm_type,
        )
        self.stage3 = TransitionResidualStage(
            in_channels=self.stage2.out_channels, out_channels=stage3_out_channels,
            stride=2, num_identity_blocks=stage3_identity_blocks, norm_type=norm_type,
        )
        self.stage4 = TransitionResidualStage(
            in_channels=self.stage3.out_channels, out_channels=stage4_out_channels,
            stride=2, num_identity_blocks=stage4_identity_blocks, norm_type=norm_type,
        )

        self.out_channels = self.stage4.out_channels
        self.feature_dim = self.out_channels  # alias used elsewhere in this project (dataset.py, momentum.py)

    def forward(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, channels, H, W), got {tuple(x.shape)}")
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        if pool:
            out = torch.mean(out, dim=(2, 3))
        return out


if __name__ == "__main__":
    encoder = Encoder()
    dummy = torch.zeros(2, 1, TARGET_SIZE, TARGET_SIZE)

    dense = encoder(dummy, pool=False)
    pooled = encoder(dummy, pool=True)

    num_params = sum(p.numel() for p in encoder.parameters())
    expected_grid = TARGET_SIZE // 8  # three stride-2 downsamples

    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Dense output shape:  {tuple(dense.shape)} (expected grid size {expected_grid}x{expected_grid})")
    print(f"Pooled output shape: {tuple(pooled.shape)}")
    print(f"Parameter count: {num_params:,}")
