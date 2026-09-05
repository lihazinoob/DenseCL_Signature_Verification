"""Runs one training or evaluation pass over a dual-triplet dataloader.

Adapted from the SURDS-era `run_one_epoch.py`
(`Thesis_Final/downstream_verification/utils/run_one_epoch.py`) - same
per-batch mechanics (compute the 4 embeddings via the model's default
forward, compute the dual-triplet loss, track ranking-accuracy
diagnostics, backward + optimizer step only when an optimizer is given).
Unchanged from the old pipeline: `DownstreamVerificationModel.forward()`
already defaults to `forward_global` (Method A), so `model(x)` here is
Method A exactly, same as the old model's single forward path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _compute_batch_embeddings(model: nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    anchor = batch["anchor"].to(device)
    positive = batch["positive"].to(device)
    negative_intra = batch["negative_intra"].to(device)
    negative_inter = batch["negative_inter"].to(device)

    return {
        "anchor_embedding": model(anchor),
        "positive_embedding": model(positive),
        "negative_intra_embedding": model(negative_intra),
        "negative_inter_embedding": model(negative_inter),
    }


def _compute_distance_diagnostics(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_intra_embedding: torch.Tensor,
    negative_inter_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Beyond the loss number itself: are genuine pairs actually ending up
    closer than forgeries/other-writers, on average, in this batch? A
    falling loss doesn't guarantee this - these ranking-accuracy numbers
    are a more direct per-epoch read on whether embeddings are ordering
    correctly, independent of the loss margin's exact scale."""
    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
    negative_intra_distance = F.pairwise_distance(anchor_embedding, negative_intra_embedding)
    negative_inter_distance = F.pairwise_distance(anchor_embedding, negative_inter_embedding)

    return {
        "positive_distance_mean": positive_distance.mean(),
        "negative_intra_distance_mean": negative_intra_distance.mean(),
        "negative_inter_distance_mean": negative_inter_distance.mean(),
        "intra_ranking_accuracy": (positive_distance < negative_intra_distance).float().mean(),
        "inter_ranking_accuracy": (positive_distance < negative_inter_distance).float().mean(),
    }


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Eval",
) -> dict[str, float]:
    """One pass over `data_loader`. Training mode (backward + optimizer
    step) iff `optimizer` is given; otherwise a forward-only evaluation
    pass - used for the fixed-quadruple validation loss (see
    `downstream_supervised_learning_approach.md`, "Validation loss curve:
    fixed sampling, not dynamic" - a training-health diagnostic, not the
    real checkpoint-selection metric)."""
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

            embeddings = _compute_batch_embeddings(model, batch, device)
            loss_outputs = loss_function(**embeddings)
            diagnostics = _compute_distance_diagnostics(**embeddings)

            if is_training:
                loss_outputs["loss"].backward()
                optimizer.step()

            for key in ("loss", "intra_loss", "inter_loss"):
                running[key] += float(loss_outputs[key].item())
            for key in (
                "positive_distance_mean", "negative_intra_distance_mean", "negative_inter_distance_mean",
                "intra_ranking_accuracy", "inter_ranking_accuracy",
            ):
                running[key] += float(diagnostics[key].item())
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{running['loss'] / num_batches:.4f}",
                "intra_acc": f"{running['intra_ranking_accuracy'] / num_batches:.4f}",
                "inter_acc": f"{running['inter_ranking_accuracy'] / num_batches:.4f}",
            })

    return {key: value / max(1, num_batches) for key, value in running.items()}
