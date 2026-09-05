"""Distance-based dual triplet loss - takes precomputed distances directly
instead of embeddings.

`nn.TripletMarginLoss` (which `DualTripletLoss` wraps) computes its own
distance internally from three embeddings using a fixed formula (Euclidean,
by default) - it has no way to accept a distance that was already computed
some other way. Step 3's `blended_distance` doesn't produce an embedding at
all; it needs the full dense feature grid and foreground mask for each
image (not just a summary vector) to compute Method B's half of the blend,
so there is no vector that could be handed to `nn.TripletMarginLoss` and
have it "discover" the blended distance on its own. This module keeps
`DualTripletLoss`'s exact margin math (`max(0, d(a,p) - d(a,n) + margin)`,
intra + inter_loss_weight * inter, summed) but applies it directly to
distances the caller already computed - by `blended_distance` or anything
else - instead of computing them internally from embeddings.

`DualTripletLoss` (embedding-based) is left unchanged and still used for
the Method-A-only fast path (`run_one_epoch.py` / `trainer.py`) - it
doesn't need any of this machinery when alpha=0, and reusing it there
keeps that already-verified path untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def triplet_margin_from_distances(
    distance_positive: torch.Tensor,
    distance_negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """`max(0, d(a,p) - d(a,n) + margin)`, per sample, averaged over the
    batch - exactly what `nn.TripletMarginLoss` computes internally, minus
    the distance computation itself."""
    return torch.clamp(distance_positive - distance_negative + margin, min=0.0).mean()


class DistanceDualTripletLoss(nn.Module):
    def __init__(
        self,
        intra_margin: float = 0.2,
        inter_margin: float = 0.2,
        inter_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.intra_margin = intra_margin
        self.inter_margin = inter_margin
        self.inter_loss_weight = inter_loss_weight

    def forward(
        self,
        distance_positive: torch.Tensor,
        distance_negative_intra: torch.Tensor,
        distance_negative_inter: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intra_loss = triplet_margin_from_distances(distance_positive, distance_negative_intra, self.intra_margin)
        inter_loss = triplet_margin_from_distances(distance_positive, distance_negative_inter, self.inter_margin)
        total_loss = intra_loss + (self.inter_loss_weight * inter_loss)

        return {
            "loss": total_loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
        }
