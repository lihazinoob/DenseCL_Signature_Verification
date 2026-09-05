"""Step 5, part 3: threshold-free metrics (ROC-AUC, EER) from a scored draw.

Direct port of the SURDS-era metrics math
(`Thesis_Final/downstream_verification/evaluation/metrics.py`), adapted to
operate on `list[QueryScore]` (from `scoring.py`) instead of a pandas
DataFrame.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from scoring import QueryScore  # noqa: E402


@dataclass(frozen=True)
class RocResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray  # on the SCORE scale (-distance), matching sklearn's roc_curve convention
    roc_auc: float


def compute_roc_metrics(scores: list[QueryScore]) -> RocResult:
    labels = np.array([s.label for s in scores])
    score_values = np.array([s.score for s in scores])
    fpr, tpr, thresholds = roc_curve(labels, score_values)
    roc_auc = float(auc(fpr, tpr))
    return RocResult(fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)


def compute_eer(roc: RocResult) -> dict[str, float]:
    """EER at the point where FPR is closest to FNR (1 - TPR)."""
    fnr = 1.0 - roc.tpr
    eer_gap = np.abs(roc.fpr - fnr)
    index = int(np.argmin(eer_gap))
    return {
        "eer": float((roc.fpr[index] + fnr[index]) / 2.0),
        "eer_score_threshold": float(roc.thresholds[index]),
        "eer_distance_threshold": float(-roc.thresholds[index]),
    }


def compute_per_writer_auc(scores: list[QueryScore]) -> dict[str, float]:
    """AUC computed independently per writer. Writers with only one label
    class present (no forgery queries, or no genuine queries left after
    the reference split) are skipped - AUC is undefined for a single
    class."""
    writer_ids = sorted({s.writer_id for s in scores})
    per_writer_auc: dict[str, float] = {}

    for writer_id in writer_ids:
        writer_scores = [s for s in scores if s.writer_id == writer_id]
        labels = {s.label for s in writer_scores}
        if len(labels) < 2:
            print(f"Warning: writer {writer_id} has only one label class - skipping per-writer AUC.")
            continue
        roc = compute_roc_metrics(writer_scores)
        per_writer_auc[writer_id] = roc.roc_auc

    return per_writer_auc
