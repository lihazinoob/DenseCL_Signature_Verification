"""Epoch loop for downstream supervised verification training.

Adapted from the SURDS-era `train_and_validate_model`
(`Thesis_Final/downstream_verification/utils/train_validation.py`): per
epoch, run a training pass (dynamic quadruples, `train_dataset.set_epoch()`
advanced first) then a validation-LOSS pass (fixed quadruples), log both
to a CSV flushed after every epoch (so a mid-training crash loses no
history), and checkpoint periodically plus whenever the "best" score
improves.

`run_epoch_fn` is a second extension point (alongside `verification_callback`),
letting this same epoch loop drive either the Method-A-only path
(`run_one_epoch`, the default - encode once per image, cheap) or the
blended path (`run_one_epoch_blended`, wrapped via `functools.partial` to
bind `alpha`/`scales`/the Sinkhorn settings so it matches `run_one_epoch`'s
call signature - see `trainer.py`). This loop itself stays agnostic to
which one is in play; it just calls whatever `run_epoch_fn` it was given
for both the training pass and the fixed-quadruple validation-loss pass.

One deliberate difference from the old pipeline: the old `val_protocol.py`
selected checkpoints by a prototype-averaged AUC (`run_prototype_validation`),
which `downstream_supervised_learning_approach.md`'s "Inference and
validation protocol" section explicitly rules out as the wrong aggregation
for Method B (prototypes don't exist there - averaging would destroy the
piece-level structure Method B relies on) and, for consistency, drops for
Method A too (per-reference-then-average is the shared primary protocol
for both methods, so the only thing differing between them is the distance
function itself, not the aggregation scheme). That K-reference AUC/EER
protocol (Step 5) is not yet built in this project, so `verification_callback`
is an explicit extension point: pass it once that protocol exists, and it
becomes the real checkpoint-selection metric, exactly as documented in
"What validation does differently from test." Until then, this function
loudly falls back to validation triplet loss - a metric the same markdown
section explicitly names as the wrong one for this job (a proxy that can
diverge from real verification performance) - and says so every time it
runs, so this can never be mistaken for the finished protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_DIR))

from run_one_epoch import run_one_epoch  # noqa: E402


def train_and_validate_model(
    model: nn.Module,
    train_dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    history_csv_path: Path,
    model_dir: Path,
    best_model_name: str = "best_model.pt",
    periodic_save_frequency: int = 5,
    verification_callback: Optional[Callable[[nn.Module, torch.device], dict]] = None,
    run_epoch_fn: Optional[Callable] = None,
    patience: Optional[int] = None,
    min_epochs: Optional[int] = None,
) -> pd.DataFrame:
    """`train_dataset` must expose `set_epoch(epoch)` (i.e. be a
    `DualTripletDataset`); `train_loader`/`val_loader` wrap it and a
    `FixedDualTripletDataset` respectively. `verification_callback`, if
    given, is called once per epoch as `callback(model, device)` and must
    return a dict containing `'primary_metric'` (higher = better, e.g.
    ROC-AUC) - that becomes the real checkpoint-selection score. If
    omitted, checkpoint selection falls back to validation triplet loss
    (lower = better), with a loud warning printed at the start of every
    call, since the roadmap doc explicitly documents that fallback as
    methodologically wrong for final use - a stand-in for while Step 5's
    evaluation protocol doesn't exist yet, not a silent substitute for it.

    `run_epoch_fn`, if given, replaces `run_one_epoch` as the function
    called for both the training pass and the validation-loss pass - must
    match `run_one_epoch`'s keyword-callable signature
    (`model, data_loader, loss_function, device, optimizer, description`).
    Defaults to `run_one_epoch` (Method A) when omitted.

    The per-epoch CSV row is built from WHATEVER keys `run_epoch_fn`
    returns (each prefixed `train_`/`val_`), not a fixed hardcoded list -
    `run_one_epoch` (4-tuple/triplet), `run_one_epoch_blended` (Step 3)
    and `run_one_epoch_pairs` (Step 2's double-margin loss) each return a
    different, self-describing set of metric names, and forcing them into
    one fixed schema would either break or silently mislabel columns.
    """
    run_epoch_fn = run_epoch_fn or run_one_epoch
    if verification_callback is None:
        print(
            "WARNING: no verification_callback given - checkpoint selection is "
            "falling back to VALIDATION TRIPLET LOSS. downstream_supervised_learning_approach.md "
            "explicitly documents this as the wrong metric for checkpoint selection (a proxy that "
            "can diverge from real verification performance - see 'What validation does "
            "differently from test'). This is a placeholder until the K-reference AUC/EER "
            "evaluation protocol (Step 5) is built; pass verification_callback once it exists."
        )

    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_score = float("-inf") if verification_callback is not None else float("inf")
    epochs_since_best = 0

    for epoch in range(epochs):
        current_epoch = epoch + 1
        print(f"===== Epoch {current_epoch}/{epochs} =====")

        train_dataset.set_epoch(epoch)

        train_metrics = run_epoch_fn(
            model=model, data_loader=train_loader, loss_function=loss_function,
            device=device, optimizer=optimizer, description="Train",
        )
        val_metrics = run_epoch_fn(
            model=model, data_loader=val_loader, loss_function=loss_function,
            device=device, optimizer=None, description="Val loss (fixed quadruples)",
        )

        epoch_record: dict = {"epoch": current_epoch}
        for key, value in train_metrics.items():
            epoch_record[f"train_{key}"] = value
        for key, value in val_metrics.items():
            epoch_record[f"val_{key}"] = value

        if verification_callback is not None:
            print("Running verification protocol...")
            verification_metrics = verification_callback(model, device)
            epoch_record.update(verification_metrics)
            current_score = verification_metrics["primary_metric"]
            improved = current_score > best_score
        else:
            current_score = val_metrics["loss"]
            improved = current_score < best_score

        history.append(epoch_record)
        print(epoch_record)

        if improved:
            best_score = current_score
            epochs_since_best = 0
            best_model_path = model_dir / best_model_name
            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "history": history,
            }, best_model_path)
            print(f"Best model saved -> {best_model_path}")
        else:
            epochs_since_best += 1

        if current_epoch % periodic_save_frequency == 0:
            periodic_path = model_dir / f"epoch{current_epoch}.pt"
            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "history": history,
            }, periodic_path)
            print(f"Periodic checkpoint saved -> {periodic_path}")

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv_path, index=False)

        if patience is not None and current_epoch >= (min_epochs or epochs) and epochs_since_best >= patience:
            print(
                f"Early stopping at epoch {current_epoch} "
                f"(no improvement in {patience} epochs; best={best_score:.4f})"
            )
            break

    return pd.DataFrame(history)
