"""Runs one training or evaluation pass using Step 3's blended distance
(Method A + Method B) instead of Method A's embeddings alone.

Parallel to `run_one_epoch.py`, not a replacement for it -
`run_one_epoch.py` stays the fast, already-verified Method-A-only path
(encode 4 images once per quadruple, reuse the embeddings for all 3
distances via `nn.TripletMarginLoss`'s own internal distance). That
shortcut doesn't exist here: `blended_distance` computes a distance
directly (not an embedding) and needs its own forward pass through both
`forward_global` and `forward_dense` for every pair, so each of the 3
required distances per quadruple (anchor-positive, anchor-negative_intra,
anchor-negative_inter) needs its own `blended_distance` call.

`blended_distance` (and `dense_matching_distance` beneath it) operates on
ONE pair at a time - no batched Sinkhorn support yet (see
`dense_matching.py`'s module docstring: piles are different sizes per
image, so a straightforward batch dimension doesn't apply the same way it
does for Method A). This module's batch handling is therefore a Python
loop over the batch dimension, correctness-first rather than optimized -
worth profiling and batching properly if it becomes the bottleneck once
real training runs are timed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAINING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(SUPERVISED_DIR / "matching"))

from blended_distance import DistanceScales, blended_distance  # noqa: E402


def _compute_batch_distances(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    alpha: float,
    scales: DistanceScales,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> dict[str, torch.Tensor]:
    batch_size = batch["anchor"].shape[0]

    positive_distances: list[torch.Tensor] = []
    negative_intra_distances: list[torch.Tensor] = []
    negative_inter_distances: list[torch.Tensor] = []

    for sample_index in range(batch_size):
        anchor = batch["anchor"][sample_index].to(device)
        positive = batch["positive"][sample_index].to(device)
        negative_intra = batch["negative_intra"][sample_index].to(device)
        negative_inter = batch["negative_inter"][sample_index].to(device)

        positive_distances.append(
            blended_distance(model, anchor, positive, alpha, scales, sinkhorn_epsilon, sinkhorn_iterations)
        )
        negative_intra_distances.append(
            blended_distance(model, anchor, negative_intra, alpha, scales, sinkhorn_epsilon, sinkhorn_iterations)
        )
        negative_inter_distances.append(
            blended_distance(model, anchor, negative_inter, alpha, scales, sinkhorn_epsilon, sinkhorn_iterations)
        )

    return {
        "positive_distance": torch.stack(positive_distances),
        "negative_intra_distance": torch.stack(negative_intra_distances),
        "negative_inter_distance": torch.stack(negative_inter_distances),
    }


def run_one_epoch_blended(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    alpha: float,
    scales: DistanceScales,
    optimizer: torch.optim.Optimizer | None = None,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
    description: str = "Eval",
) -> dict[str, float]:
    """`loss_function` must be a `DistanceDualTripletLoss` (takes
    precomputed distances, not embeddings). Same running-metrics shape as
    `run_one_epoch` for consistency (loss/intra_loss/inter_loss,
    per-pair-type mean distance, intra/inter ranking accuracy)."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    running = {
        "loss": 0.0, "intra_loss": 0.0, "inter_loss": 0.0,
        "positive_distance_mean": 0.0, "negative_intra_distance_mean": 0.0, "negative_inter_distance_mean": 0.0,
        "intra_ranking_accuracy": 0.0, "inter_ranking_accuracy": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress_bar = tqdm(data_loader, desc=description, ncols=100, mininterval=10)
        for batch in progress_bar:
            if is_training:
                optimizer.zero_grad()

            distances = _compute_batch_distances(
                model, batch, device, alpha, scales, sinkhorn_epsilon, sinkhorn_iterations,
            )
            loss_outputs = loss_function(
                distances["positive_distance"],
                distances["negative_intra_distance"],
                distances["negative_inter_distance"],
            )

            if is_training:
                loss_outputs["loss"].backward()
                optimizer.step()

            intra_ranking_accuracy = (
                distances["positive_distance"] < distances["negative_intra_distance"]
            ).float().mean()
            inter_ranking_accuracy = (
                distances["positive_distance"] < distances["negative_inter_distance"]
            ).float().mean()

            for key in ("loss", "intra_loss", "inter_loss"):
                running[key] += float(loss_outputs[key].item())
            running["positive_distance_mean"] += float(distances["positive_distance"].mean().item())
            running["negative_intra_distance_mean"] += float(distances["negative_intra_distance"].mean().item())
            running["negative_inter_distance_mean"] += float(distances["negative_inter_distance"].mean().item())
            running["intra_ranking_accuracy"] += float(intra_ranking_accuracy.item())
            running["inter_ranking_accuracy"] += float(inter_ranking_accuracy.item())
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{running['loss'] / num_batches:.4f}",
                "intra_acc": f"{running['intra_ranking_accuracy'] / num_batches:.4f}",
                "inter_acc": f"{running['inter_ranking_accuracy'] / num_batches:.4f}",
            })

    return {key: value / max(1, num_batches) for key, value in running.items()}
