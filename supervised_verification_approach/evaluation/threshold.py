"""Step 5, part 4: leakage-free threshold (tau) selection and application.

Direct port of the SURDS-era `threshold_metrics.py`'s reasoning and math
(`Thesis_Final/downstream_verification/evaluation/threshold_metrics.py`),
adapted to `list[QueryScore]` (from `scoring.py`).

Decision rule (fixed): a query is predicted GENUINE iff its (mean, over K
references) distance is <= tau. With genuine=1, forgery=0:

    TP = genuine accepted      FN = genuine rejected
    TN = forgery rejected      FP = forgery accepted
    TPR = TP/(TP+FN)  (genuine acceptance rate)
    TNR = TN/(TN+FP)  (forgery rejection rate)
    FAR = FPR = FP/(FP+TN)     (forgeries wrongly accepted)
    FRR = FNR = FN/(FN+TP)     (genuines wrongly rejected)
    balanced_accuracy = (TPR + TNR) / 2

Why per-draw, not pooled: the prototype protocol's only randomness is
WHICH genuine signatures become references. Each reference draw (seed)
produces different reference sets and therefore a different distance
scale for that draw. Pooling distances across draws before sweeping a
single tau would fit the cutoff to a mixture of inconsistent geometries.
So tau is swept inside each draw separately, and only the resulting
tau/metrics are averaged across draws - the test side mirrors this: apply
the one fixed tau within each draw, then average.

tau is selected ONLY on validation-writer draws (`select_tau_per_draw_average`);
test-writer draws (`evaluate_at_fixed_tau_per_draw`) only ever APPLY that
already-fixed tau - test writers never influence where the cutoff is drawn
(see downstream_supervised_learning_approach.md, "held-out threshold").
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

EVALUATION_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(SUPERVISED_DIR / "models"))

from embedding_model import DownstreamVerificationModel  # noqa: E402
from metrics import compute_eer, compute_roc_metrics  # noqa: E402
from scoring import BatchDistanceFn, DistanceFn, QueryScore, run_verification_draw  # noqa: E402

DEFAULT_TAU_STEP = 0.00005


def metrics_at_threshold(distances: np.ndarray, labels: np.ndarray, tau: float) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    predicted_genuine = distances <= tau
    is_genuine = labels == 1
    is_forgery = labels == 0

    true_positive = int(np.sum(predicted_genuine & is_genuine))
    false_negative = int(np.sum(~predicted_genuine & is_genuine))
    true_negative = int(np.sum(~predicted_genuine & is_forgery))
    false_positive = int(np.sum(predicted_genuine & is_forgery))
    total = true_positive + false_negative + true_negative + false_positive

    num_genuine = true_positive + false_negative
    num_forgery = true_negative + false_positive

    tpr = true_positive / num_genuine if num_genuine > 0 else 0.0
    tnr = true_negative / num_forgery if num_forgery > 0 else 0.0
    fnr = 1.0 - tpr
    fpr = 1.0 - tnr

    return {
        "tau": float(tau),
        "tp": true_positive, "fn": false_negative, "tn": true_negative, "fp": false_positive,
        "accuracy": float((true_positive + true_negative) / total) if total > 0 else 0.0,
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "tpr": float(tpr), "tnr": float(tnr), "fpr": float(fpr), "fnr": float(fnr),
        "far": float(fpr), "frr": float(fnr),
    }


def sweep_balanced_accuracy_threshold(
    distances: np.ndarray, labels: np.ndarray, step: float = DEFAULT_TAU_STEP,
) -> dict[str, float]:
    """Sweep tau across this draw's own observed distance range, pick the
    cutoff maximizing balanced accuracy. On ties, the smallest tau wins.

    Vectorized via `np.searchsorted` rather than a Python loop over the tau
    grid. The grid (`np.arange(min, max + step, step)`), the metric being
    maximized, and the smallest-tau tie-break are all UNCHANGED - only the
    evaluation is; `np.argmax` returns the first occurrence of the maximum
    over an ascending grid, exactly as the old strict-`>` loop did.

    This matters because the grid is fine (`step = 5e-5`, matching SURDS's
    stated interval) so a realistic distance range gives tens of thousands
    of candidates. Looping in Python cost seconds per draw here, and would
    have been prohibitive for the all-pairs protocol, whose negative set is
    ~43k pairs per dataset. The winning tau is passed back through
    `metrics_at_threshold` so the returned dict is byte-for-byte the same
    shape and values as before.
    """
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if distances.size == 0:
        raise ValueError("Cannot sweep tau over an empty distance array.")

    lowest, highest = float(distances.min()), float(distances.max())
    candidate_taus = np.arange(lowest, highest + step, step)  # +step so "accept everything" is included

    genuine_sorted = np.sort(distances[labels == 1])
    forgery_sorted = np.sort(distances[labels == 0])
    num_genuine, num_forgery = genuine_sorted.size, forgery_sorted.size

    # predicted genuine iff distance <= tau, so 'right' side gives the count at each tau
    true_positive = np.searchsorted(genuine_sorted, candidate_taus, side="right")
    false_positive = np.searchsorted(forgery_sorted, candidate_taus, side="right")
    true_negative = num_forgery - false_positive

    tpr = true_positive / num_genuine if num_genuine > 0 else np.zeros_like(candidate_taus)
    tnr = true_negative / num_forgery if num_forgery > 0 else np.zeros_like(candidate_taus)
    balanced_accuracy = (tpr + tnr) / 2.0

    best_tau = float(candidate_taus[int(np.argmax(balanced_accuracy))])
    return metrics_at_threshold(distances, labels, best_tau)


def _scores_to_arrays(scores: list[QueryScore]) -> tuple[np.ndarray, np.ndarray]:
    distances = np.array([s.distance for s in scores], dtype=np.float64)
    labels = np.array([s.label for s in scores], dtype=int)
    return distances, labels


def select_tau_per_draw_average(
    model: DownstreamVerificationModel,
    writer_ids: list[str],
    dataset_dir: Path,
    dataset_name: str,
    distance_fn: DistanceFn,
    num_references: int,
    seeds: list[int],
    device: torch.device,
    step: float = DEFAULT_TAU_STEP,
    show_progress: bool = True,
    encoded_cache: Optional[dict] = None,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """VALIDATION-side tau selection. For each seed: run one reference
    draw over `writer_ids` (should be validation writers), sweep tau over
    that draw's own distances, take the tau maximizing balanced accuracy.
    Average the per-draw taus into the single operating threshold `tau_star`.

    `encoded_cache` is shared across draws - see `run_verification_draw`
    for why, and for the constraint that it must not outlive a change to
    the model's weights. `batch_distance_fn`/`batch_size` are passed
    straight through to `run_verification_draw` - see its docstring."""
    per_draw_taus: list[float] = []
    per_draw_balanced_accs: list[float] = []

    for seed in seeds:
        scores = run_verification_draw(
            model, writer_ids, dataset_dir, dataset_name, distance_fn,
            num_references, seed, device, show_progress=show_progress,
            encoded_cache=encoded_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
        )
        distances, labels = _scores_to_arrays(scores)
        best = sweep_balanced_accuracy_threshold(distances, labels, step=step)
        per_draw_taus.append(best["tau"])
        per_draw_balanced_accs.append(best["balanced_accuracy"])

    return {
        "tau_star": float(np.mean(per_draw_taus)),
        "tau_std": float(np.std(per_draw_taus)),
        "val_balanced_acc_mean": float(np.mean(per_draw_balanced_accs)),
        "val_balanced_acc_std": float(np.std(per_draw_balanced_accs)),
        "per_draw_taus": per_draw_taus,
    }


def evaluate_test_optimal_tau_per_draw(
    model: DownstreamVerificationModel,
    writer_ids: list[str],
    dataset_dir: Path,
    dataset_name: str,
    distance_fn: DistanceFn,
    num_references: int,
    seeds: list[int],
    device: torch.device,
    step: float = DEFAULT_TAU_STEP,
    show_progress: bool = True,
    encoded_cache: Optional[dict] = None,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """SURDS-CONVENTION evaluation: sweep tau on the TEST scores themselves
    and report the best result. This deliberately does what
    `evaluate_at_fixed_tau_per_draw` deliberately does NOT.

    Why this exists despite being the weaker methodology: SURDS's published
    numbers (BHSig260 Hindi 89.50%, Bengali 87.34% - the figures this
    thesis is compared against) are defined this way. Their Eq. 8 is

        Accuracy = max over tau of (TPR(tau) + TNR(tau)) / 2

    i.e. the threshold is chosen on the very data being scored, swept "from
    the minimum distance to maximum distance value at intervals of 5e-5"
    (their Section IV-C) - which is exactly `DEFAULT_TAU_STEP` and exactly
    the balanced accuracy `sweep_balanced_accuracy_threshold` maximizes.

    Reporting only our stricter validation-selected tau against their
    test-optimal tau would understate this work's results for a purely
    methodological reason, on an identical model. So both conventions get
    reported, always, clearly labelled - the SURDS-matched number for the
    comparison table, the validation-selected number as this thesis's own
    methodological improvement. See
    `docs/claude_response/downstream_supervised_learning_approach.md`,
    "Evaluation protocol - decided, with the reasoning".

    Per-draw, then averaged, for the same reason as everything else in this
    module (see module docstring): each reference draw has its own distance
    scale, so tau is swept inside a draw and only the results are averaged.
    """
    accumulators: dict[str, list[float]] = {
        "roc_auc": [], "eer": [], "tau": [], "accuracy": [], "balanced_accuracy": [],
        "tpr": [], "tnr": [], "far": [], "frr": [],
    }

    for seed in seeds:
        scores = run_verification_draw(
            model, writer_ids, dataset_dir, dataset_name, distance_fn,
            num_references, seed, device, show_progress=show_progress,
            encoded_cache=encoded_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
        )
        distances, labels = _scores_to_arrays(scores)

        roc = compute_roc_metrics(scores)
        best = sweep_balanced_accuracy_threshold(distances, labels, step=step)

        accumulators["roc_auc"].append(roc.roc_auc)
        accumulators["eer"].append(compute_eer(roc)["eer"])
        for key in ("tau", "accuracy", "balanced_accuracy", "tpr", "tnr", "far", "frr"):
            accumulators[key].append(best[key])

    summary: dict[str, float] = {}
    for metric_name, values in accumulators.items():
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
    return summary


def evaluate_at_fixed_tau_per_draw(
    model: DownstreamVerificationModel,
    writer_ids: list[str],
    dataset_dir: Path,
    dataset_name: str,
    distance_fn: DistanceFn,
    num_references: int,
    seeds: list[int],
    tau: float,
    device: torch.device,
    show_progress: bool = True,
    encoded_cache: Optional[dict] = None,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """TEST-side (or validation-monitoring-side) evaluation at a FIXED
    tau. For each seed: run one reference draw, compute ROC-AUC/EER for
    that draw (threshold-free), and apply the fixed `tau` for the
    threshold-dependent metrics. Report mean +/- std across draws."""
    accumulators: dict[str, list[float]] = {
        "roc_auc": [], "eer": [], "accuracy": [], "balanced_accuracy": [],
        "tpr": [], "tnr": [], "far": [], "frr": [],
    }

    for seed in seeds:
        scores = run_verification_draw(
            model, writer_ids, dataset_dir, dataset_name, distance_fn,
            num_references, seed, device, show_progress=show_progress,
            encoded_cache=encoded_cache, batch_distance_fn=batch_distance_fn, batch_size=batch_size,
        )
        distances, labels = _scores_to_arrays(scores)

        roc = compute_roc_metrics(scores)
        eer_info = compute_eer(roc)
        threshold_metrics = metrics_at_threshold(distances, labels, tau)

        accumulators["roc_auc"].append(roc.roc_auc)
        accumulators["eer"].append(eer_info["eer"])
        for key in ("accuracy", "balanced_accuracy", "tpr", "tnr", "far", "frr"):
            accumulators[key].append(threshold_metrics[key])

    summary: dict[str, float] = {"tau": float(tau)}
    for metric_name, values in accumulators.items():
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
    return summary
