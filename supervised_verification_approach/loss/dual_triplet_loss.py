"""Dual triplet loss for downstream supervised verification training.

Direct port of the SURDS-era `DualTripletLoss`
(`Thesis_Final/downstream_verification/loss/dual_triplet_loss.py`) - the
math is unchanged, matching the report's Eq. 3.6
(`docs/report/2007038_report_v4.pdf`, Chapter 3):

    TripletLoss_Intra = max(0, d(a,p) - d(a,n_intra) + margin_intra)
    TripletLoss_Inter = max(0, d(a,p) - d(a,n_inter) + margin_inter)
    L_total = TripletLoss_Intra + inter_loss_weight * TripletLoss_Inter

where a/p are genuine anchor/positive (same writer), n_intra is a skilled
forgery of the same writer, and n_inter is a genuine signature of a
different writer - the two negatives `DualTripletDataset` /
`FixedDualTripletDataset` already produce per 4-tuple.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualTripletLoss(nn.Module):
    def __init__(
        self,
        intra_margin: float = 0.2,
        inter_margin: float = 0.2,
        inter_loss_weight: float = 1.0,
        distance_p: float = 2.0,
    ) -> None:
        super().__init__()
        self.intra_triplet = nn.TripletMarginLoss(margin=intra_margin, p=distance_p)
        self.inter_triplet = nn.TripletMarginLoss(margin=inter_margin, p=distance_p)
        self.inter_loss_weight = inter_loss_weight

    def forward(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_intra_embedding: torch.Tensor,
        negative_inter_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intra_loss = self.intra_triplet(anchor_embedding, positive_embedding, negative_intra_embedding)
        inter_loss = self.inter_triplet(anchor_embedding, positive_embedding, negative_inter_embedding)
        total_loss = intra_loss + (self.inter_loss_weight * inter_loss)

        return {
            "loss": total_loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
        }
