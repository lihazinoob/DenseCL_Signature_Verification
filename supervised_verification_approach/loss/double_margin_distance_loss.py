"""Distance-based double-margin contrastive loss for Step 4's combined
distance - the precomputed-scalar-distance counterpart to
`double_margin_loss.DoubleMarginLoss` (which takes two embeddings and
computes `pairwise_distance` internally).

Needed because the combined distance (global + structural) is not a
simple embedding distance: the structural term only ever exists as a
PAIRWISE scalar produced by Sinkhorn matching two piles of pieces - there
is no single fixed-length "combined embedding" per image a generic
embedding-based loss could consume. Same relationship
`distance_triplet_loss.DistanceDualTripletLoss` already has to
`dual_triplet_loss.DualTripletLoss` in this codebase, just for the
pair/double-margin loss instead of the triplet one.

Same formula as `DoubleMarginLoss` (DetailSemNet Eq. 13):

    y * max(0, dist - m)^2 + (1 - y) * max(0, n - dist)^2

`m`/`n` here are NOT Step 2/3's 0.46/0.96 - those were swept on pure
global-embedding distance and do not transfer to the combined distance's
different composition/scale. See `training/sweep_margins_combined.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` where `mask` is True; 0.0 (not NaN) if no elements
    match - same rationale as `double_margin_loss._masked_mean`."""
    if mask.sum() == 0:
        return torch.zeros((), device=values.device)
    return values[mask].mean()


class DoubleMarginDistanceLoss(nn.Module):
    def __init__(self, margin_m: float, margin_n: float) -> None:
        super().__init__()
        if not margin_m < margin_n:
            raise ValueError(f"margin_m ({margin_m}) must be < margin_n ({margin_n})")
        self.margin_m = margin_m
        self.margin_n = margin_n

    def forward(self, distance: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        is_positive = label > 0.5

        positive_term = label * torch.clamp(distance - self.margin_m, min=0.0).pow(2)
        negative_term = (1.0 - label) * torch.clamp(self.margin_n - distance, min=0.0).pow(2)
        loss = (positive_term + negative_term).mean()

        with torch.no_grad():
            positive_active_rate = _masked_mean((distance > self.margin_m).float(), is_positive)
            negative_active_rate = _masked_mean((distance < self.margin_n).float(), ~is_positive)
            positive_distance_mean = _masked_mean(distance, is_positive)
            negative_distance_mean = _masked_mean(distance, ~is_positive)
            positive_loss_mean = _masked_mean(positive_term, is_positive)
            negative_loss_mean = _masked_mean(negative_term, ~is_positive)

        return {
            "loss": loss,
            "positive_loss": positive_loss_mean,
            "negative_loss": negative_loss_mean,
            "positive_distance_mean": positive_distance_mean,
            "negative_distance_mean": negative_distance_mean,
            "positive_active_rate": positive_active_rate,
            "negative_active_rate": negative_active_rate,
        }
