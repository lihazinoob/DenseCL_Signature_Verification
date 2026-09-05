"""Runs one training or evaluation pass using Step 4's combined distance
(global + local/structural, both branches trainable via a shared encoder
forward pass) instead of a single-branch embedding distance.

Parallel to `run_one_epoch_pairs.py` (Step 2/3's double-margin loss on
global-only embeddings) - reuses the exact same `PairDataset` batch shape
(`image_a`, `image_b`, `label`), but the combined distance can't be
decomposed into two independent per-image embeddings the way global
distance can (the structural term only exists as a pairwise Sinkhorn
output), so this module computes `dist` directly via
`matching.combined_distance.combined_distance_batched` for the whole
batch at once, instead of calling `model(x)` on each image separately.
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

from combined_distance import combined_distance_batched  # noqa: E402

_METRIC_KEYS = (
    "loss", "positive_loss", "negative_loss",
    "positive_distance_mean", "negative_distance_mean",
    "positive_active_rate", "negative_active_rate",
)


def run_one_epoch_combined(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
    description: str = "Eval",
) -> dict[str, float]:
    """`loss_function` must be a `DoubleMarginDistanceLoss`. Matches
    `run_one_epoch`'s `run_epoch_fn`-compatible keyword signature plus the
    three combined-distance-specific keywords, bound via
    `functools.partial` in `trainer.py` exactly the way
    `run_one_epoch_blended` binds `alpha`/`scales`."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    running = {key: 0.0 for key in _METRIC_KEYS}
    num_batches = 0

    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        progress_bar = tqdm(data_loader, desc=description, ncols=100, mininterval=10)
        for batch in progress_bar:
            if is_training:
                optimizer.zero_grad()

            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            label = batch["label"].to(device).float()

            dist = combined_distance_batched(
                model, image_a, image_b,
                lambda_0=lambda_0, sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iterations=sinkhorn_iterations,
            )
            loss_outputs = loss_function(dist, label)

            if is_training:
                loss_outputs["loss"].backward()
                optimizer.step()

            for key in _METRIC_KEYS:
                running[key] += float(loss_outputs[key].item())
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{running['loss'] / num_batches:.4f}",
                "pos_active": f"{running['positive_active_rate'] / num_batches:.4f}",
                "neg_active": f"{running['negative_active_rate'] / num_batches:.4f}",
            })

    return {key: value / max(1, num_batches) for key, value in running.items()}
