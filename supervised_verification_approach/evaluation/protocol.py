"""Step 5, part 5: top-level orchestration.

Ties `encoding.py` + `scoring.py` + `metrics.py` + `threshold.py` into the
full K-reference protocol from
`docs/claude_response/downstream_supervised_learning_approach.md`, plus a
`build_verification_callback` factory matching Step 4's
`train_and_validate_model(verification_callback=...)` extension point -
this is the module that finally lets checkpoint selection use the real
metric instead of the loudly-flagged validation-loss placeholder.

Two distinct entry points, matching the roadmap doc's "What validation
does differently from test" table - deliberately NOT the same computation
run at different times:

  - `build_verification_callback`: cheap, threshold-free AUC/EER over a
    FEW validation draws, run every training epoch, for checkpoint
    selection only. Never sweeps tau - tau selection is a one-time cost,
    not a per-epoch one.
  - `run_full_protocol`: the full final-reporting pipeline (tau selection
    on validation with the full draw count, then that fixed tau applied to
    test with the full draw count, plus per-writer AUC), run once on the
    final chosen checkpoint.

Three ready-made distance functions (`make_method_a_distance_fn`,
`make_method_b_distance_fn`, `make_blended_distance_fn`) so callers don't
need to import `encoding.py`'s cached-distance functions directly - these
are exactly Step 5's four evaluation cells: Method A alone
(`make_method_a_distance_fn`), Method B alone (`make_method_b_distance_fn`),
and the blend at any alpha (`make_blended_distance_fn`), run against
both the old and new pretraining checkpoints.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

EVALUATION_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))

from encoding import (  # noqa: E402
    blended_distance_cached,
    combined_distance_batch_cached,
    combined_distance_cached,
    method_a_distance_cached,
    method_b_distance_cached,
)
from metrics import compute_eer, compute_per_writer_auc, compute_roc_metrics  # noqa: E402
from scoring import BatchDistanceFn, DistanceFn, run_verification_draw  # noqa: E402
from threshold import (  # noqa: E402
    evaluate_at_fixed_tau_per_draw,
    evaluate_test_optimal_tau_per_draw,
    select_tau_per_draw_average,
)


def make_method_a_distance_fn() -> DistanceFn:
    return method_a_distance_cached


def make_method_b_distance_fn(sinkhorn_epsilon: float = 0.05, sinkhorn_iterations: int = 50) -> DistanceFn:
    return lambda a, b: method_b_distance_cached(a, b, sinkhorn_epsilon, sinkhorn_iterations)


def make_blended_distance_fn(
    alpha: float,
    method_a_scale: float,
    method_b_scale: float,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> DistanceFn:
    return lambda a, b: blended_distance_cached(
        a, b, alpha, method_a_scale, method_b_scale, sinkhorn_epsilon, sinkhorn_iterations,
    )


def make_combined_distance_fn(
    model: nn.Module,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> DistanceFn:
    """Step 4's combined distance as a `DistanceFn` - closes over
    `model.local_projection` at call time (not passed as a separate
    argument), so it always uses whatever the projection's CURRENT
    weights are, correct for the per-epoch verification callback where
    those weights change every epoch."""
    return lambda a, b: combined_distance_cached(
        a, b, model.local_projection, lambda_0, sinkhorn_epsilon, sinkhorn_iterations,
    )


def make_combined_batch_distance_fn(
    model: nn.Module,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> BatchDistanceFn:
    """Batched sibling of `make_combined_distance_fn` - the fix for the
    verification protocol's dominant per-epoch cost (see
    downstream_supervised_learning_approach.md, "Verification protocol
    speed"). Same closure-over-`model.local_projection` behaviour (always
    uses the CURRENT weights, correct across epochs), but scores many
    pairs per call via `combined_distance_batch_cached` instead of one."""
    return lambda pairs: combined_distance_batch_cached(
        pairs, model.local_projection, lambda_0, sinkhorn_epsilon, sinkhorn_iterations,
    )


def build_verification_callback(
    dataset_dir: Path,
    dataset_name: str,
    validation_writer_ids: list[str],
    distance_fn: DistanceFn,
    num_references: int = 5,
    seeds: tuple[int, ...] = (101, 202),
    show_progress: bool = False,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> Callable[[nn.Module, torch.device], dict]:
    """Per-epoch checkpoint-selection metric: threshold-free mean ROC-AUC
    over `seeds` validation-writer reference draws (default: 2 draws, not
    the full 5 - a cheaper per-epoch monitoring budget, per the roadmap
    doc's "Per-epoch monitoring budget" note). Deliberately does NOT run
    the tau sweep - that is a one-time cost on the final checkpoint
    (`run_full_protocol`), not something recomputed every epoch.

    `batch_distance_fn`, if given (e.g. `make_combined_batch_distance_fn`),
    is passed straight through to `run_verification_draw` and takes over
    scoring entirely - `distance_fn` is then only used for its identity
    (unused otherwise) but still required, since callers not yet on a
    batched distance function (Method A alone) get the exact old behaviour
    when `batch_distance_fn=None`."""
    def callback(model: nn.Module, device: torch.device) -> dict:
        # Fresh cache per invocation: the model's weights change every
        # epoch, and the cache is keyed by image path only, so one that
        # outlived a single call would serve stale embeddings.
        encoded_cache: dict = {}
        aucs: list[float] = []
        eers: list[float] = []
        for seed in seeds:
            scores = run_verification_draw(
                model, validation_writer_ids, dataset_dir, dataset_name, distance_fn,
                num_references, seed, device, show_progress=show_progress,
                encoded_cache=encoded_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
            )
            roc = compute_roc_metrics(scores)
            aucs.append(roc.roc_auc)
            eers.append(compute_eer(roc)["eer"])

        mean_auc = float(np.mean(aucs))
        return {
            "primary_metric": mean_auc,
            "val_proto_auc_mean": mean_auc,
            "val_proto_auc_std": float(np.std(aucs)),
            "val_proto_eer_mean": float(np.mean(eers)),
        }

    return callback


@dataclass(frozen=True)
class ProtocolResult:
    """Both threshold conventions, always reported together.

    `test_summary` is the STRICT number (tau chosen on validation writers,
    applied fixed to test - this thesis's own methodology).
    `test_summary_surds_convention` is the SURDS-MATCHED number (tau swept
    on the test scores themselves, their Eq. 8), which is what their
    published 89.50%/87.34% figures mean. Reporting only the strict number
    against their optimistic one understates this work for a purely
    methodological reason - see `evaluate_test_optimal_tau_per_draw`.
    """
    val_tau_star: float
    val_balanced_acc_mean: float
    test_summary: dict[str, float]
    test_summary_surds_convention: dict[str, float]
    per_writer_auc: dict[str, float]
    num_references: int


def run_full_protocol(
    model: nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    distance_fn: DistanceFn,
    validation_writer_ids: list[str],
    test_writer_ids: list[str],
    num_references: int,
    val_seeds: list[int],
    test_seeds: list[int],
    device: torch.device,
    show_progress: bool = True,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> ProtocolResult:
    """The complete Step 5 pipeline for one (checkpoint, distance_fn)
    setting: select tau on validation writers (per-draw sweep then
    average), apply that fixed tau to test writers (per-draw then
    average), and report per-writer AUC as a secondary diagnostic (from
    the last test-writer draw - never used to pick anything, just to check
    no single writer is hiding inside a good-looking average).

    `batch_distance_fn`/`batch_size`: same batched-scoring override as
    `build_verification_callback` - pass `make_combined_batch_distance_fn(...)`
    here for the same speedup on the one-time final test-set run."""
    # One cache for the whole run - the model is fixed throughout, and all
    # ~10 draws below cover the same two image sets. Separate caches per
    # writer set only because nothing is shared between them anyway.
    val_cache: dict = {}
    test_cache: dict = {}

    tau_selection = select_tau_per_draw_average(
        model, validation_writer_ids, dataset_dir, dataset_name, distance_fn,
        num_references, val_seeds, device, show_progress=show_progress,
        encoded_cache=val_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
    )
    test_summary = evaluate_at_fixed_tau_per_draw(
        model, test_writer_ids, dataset_dir, dataset_name, distance_fn,
        num_references, test_seeds, tau_selection["tau_star"], device, show_progress=show_progress,
        encoded_cache=test_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
    )
    test_summary_surds = evaluate_test_optimal_tau_per_draw(
        model, test_writer_ids, dataset_dir, dataset_name, distance_fn,
        num_references, test_seeds, device, show_progress=show_progress,
        encoded_cache=test_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
    )
    diagnostic_scores = run_verification_draw(
        model, test_writer_ids, dataset_dir, dataset_name, distance_fn,
        num_references, test_seeds[-1], device, show_progress=False,
        encoded_cache=test_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
    )
    per_writer_auc = compute_per_writer_auc(diagnostic_scores)

    return ProtocolResult(
        val_tau_star=tau_selection["tau_star"],
        val_balanced_acc_mean=tau_selection["val_balanced_acc_mean"],
        test_summary=test_summary,
        test_summary_surds_convention=test_summary_surds,
        per_writer_auc=per_writer_auc,
        num_references=num_references,
    )
