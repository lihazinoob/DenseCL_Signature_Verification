"""Double-margin contrastive loss for downstream Step 2+ supervised training.

Implements DetailSemNet's Eq. 13 double-margin contrastive loss on labeled
PAIRS (not triplets) - see
`docs/claude_response/downstream_supervised_learning_approach.md` SS3/SS6:

    y * max(0, dist - m)^2 + (1 - y) * max(0, n - dist)^2

y=1 for a genuine-genuine pair (same writer, should be close), y=0 for a
genuine-forged or genuine-different-writer pair (should be far). m < n.

Unlike `DualTripletLoss` (Step 1), which only ever compares a positive
distance against a negative distance INSIDE one triplet and is satisfied
the instant one is smaller than the other by a margin, this loss pins
each pair's distance to an absolute target zone (below m if genuine,
above n if not) - the mechanism intended to fix the cross-writer
calibration problem measured in SS4 of the roadmap doc, where every
writer's distances had drifted onto its own scale because nothing before
this ever asked for a shared one.

`m`/`n` are not published anywhere (DetailSemNet doesn't report theirs) -
see `training/sweep_margins.py` for how they were chosen on this
project's own validation-writer embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` where `mask` is True; 0.0 (not NaN) if no elements
    match, so an all-positive or all-negative batch never poisons the
    running average - a real possibility at small batch sizes with pairs
    shuffled independently instead of one-of-each-type per batch."""
    if mask.sum() == 0:
        return torch.zeros((), device=values.device)
    return values[mask].mean()


class DoubleMarginLoss(nn.Module):
    def __init__(self, margin_m: float, margin_n: float, distance_p: float = 2.0) -> None:
        super().__init__()
        if not margin_m < margin_n:
            raise ValueError(f"margin_m ({margin_m}) must be < margin_n ({margin_n})")
        self.margin_m = margin_m
        self.margin_n = margin_n
        self.distance_p = distance_p

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        distance = F.pairwise_distance(embedding_a, embedding_b, p=self.distance_p)
        is_positive = label > 0.5

        positive_term = label * torch.clamp(distance - self.margin_m, min=0.0).pow(2)
        negative_term = (1.0 - label) * torch.clamp(self.margin_n - distance, min=0.0).pow(2)
        loss = (positive_term + negative_term).mean()

        # Diagnostics only (never backprop'd through): how many pairs in
        # this batch still have a nonzero gradient contribution. Direct
        # analogue of the "dead triplet" tracking from Step 1 - see
        # SS4 item 5 of the roadmap doc.
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
