"""Runs one training or evaluation pass over a labeled-pair dataloader
(Step 2's double-margin loss) - parallel to `run_one_epoch.py`
(4-tuple/triplet) and `run_one_epoch_blended.py` (Step 3's blended
distance), not a replacement for either; `trainer.py`'s `LOSS_TYPE` picks
which one a given run actually uses.

Simpler than the other two: one pair per training example means only two
embeddings per sample, and `forward_global` already returns a normalized
embedding cheaply, so the whole batch encodes and scores in one shot - no
per-sample Python loop like `run_one_epoch_blended` needs for its
per-pair Sinkhorn distance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_METRIC_KEYS = (
    "loss", "positive_loss", "negative_loss",
    "positive_distance_mean", "negative_distance_mean",
    "positive_active_rate", "negative_active_rate",
)


def run_one_epoch_pairs(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Eval",
) -> dict[str, float]:
    """`loss_function` must be a `DoubleMarginLoss`. Matches
    `run_one_epoch`'s `run_epoch_fn`-compatible keyword signature exactly,
    so it plugs straight into `train_and_validate_model` with no
    `functools.partial` wrapper (unlike `run_one_epoch_blended`, which
    needs one to bind alpha/scales)."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    running = {key: 0.0 for key in _METRIC_KEYS}
    num_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress_bar = tqdm(data_loader, desc=description, ncols=100, mininterval=10)
        for batch in progress_bar:
            if is_training:
                optimizer.zero_grad()

            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            label = batch["label"].to(device).float()

            embedding_a = model(image_a)
            embedding_b = model(image_b)
            loss_outputs = loss_function(embedding_a, embedding_b, label)

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
