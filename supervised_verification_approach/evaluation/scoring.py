"""Step 5, part 2: reference/query splitting and per-reference-then-average scoring.

Per `downstream_supervised_learning_approach.md`'s "Inference and
validation protocol": for each writer, K genuine signatures become
references, the writer's remaining genuines become positive queries, and
all of that writer's forgeries become negative queries (never other
writers' genuines - this tests skilled forgeries specifically, the harder
and more meaningful setting).

Critically, a query is scored against its writer's K references
SEPARATELY (K distances), then those K distances are averaged into one
final score - never by averaging the K references into a prototype first.
This is the fix documented at length in the roadmap doc: prototypes don't
exist for Method B (averaging misaligned dense grids destroys the ink
structure being matched on), and using per-reference-then-average for
BOTH methods keeps the aggregation scheme identical across Method A and
Method B, so Step 5's comparison isolates the distance function being
tested rather than confounding it with two different aggregation choices.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

EVALUATION_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(SUPERVISED_DIR / "datasets"))
sys.path.insert(0, str(SUPERVISED_DIR / "models"))

from dual_triplet_dataset import list_genuine_and_forged_images  # noqa: E402
from embedding_model import DownstreamVerificationModel  # noqa: E402
from encoding import EncodedSignature, encode_all_signatures  # noqa: E402

DistanceFn = Callable[[EncodedSignature, EncodedSignature], float]

# A batched sibling of `DistanceFn`: scores many (a, b) pairs in one call
# instead of one pair per call, so an expensive per-pair computation (e.g.
# Step 4's Sinkhorn-based structural distance) can amortize its GPU
# kernel-launch/sync overhead across a batch, the same way training does.
# Optional - callers that only have a cheap `DistanceFn` (Method A's plain
# embedding norm) have nothing to gain from batching and can ignore this.
BatchDistanceFn = Callable[[list[tuple[EncodedSignature, EncodedSignature]]], list[float]]


@dataclass(frozen=True)
class QueryScore:
    writer_id: str
    query_path: Path
    query_type: str  # "genuine" | "forgery"
    label: int        # 1 = genuine, 0 = forgery
    distance: float    # mean distance over the K references
    score: float        # -distance (higher = more genuine, for ROC/AUC)


@dataclass(frozen=True)
class WriterSplit:
    reference_paths: list[Path]
    genuine_query_paths: list[Path]
    forgery_query_paths: list[Path]


def split_references_and_queries(
    writer_dir: Path,
    dataset_name: str,
    num_references: int,
    seed: int,
) -> WriterSplit:
    """One writer's genuines -> K references + remaining genuine queries;
    all forgeries -> forgery queries. A reference path never also appears
    as a query. Deterministic given `seed` (the "one reference draw" the
    caller is currently running)."""
    genuine_paths, forgery_paths = list_genuine_and_forged_images(writer_dir, dataset_name)
    if len(genuine_paths) <= num_references:
        raise ValueError(
            f"Writer {writer_dir.name} has only {len(genuine_paths)} genuine signatures; "
            f"need more than {num_references} for a reference + query split."
        )
    if not forgery_paths:
        raise ValueError(f"Writer {writer_dir.name} has no forgery signatures.")

    rng = random.Random(seed)
    reference_paths = sorted(rng.sample(genuine_paths, num_references))
    reference_set = set(reference_paths)
    genuine_query_paths = [p for p in genuine_paths if p not in reference_set]

    return WriterSplit(
        reference_paths=reference_paths,
        genuine_query_paths=genuine_query_paths,
        forgery_query_paths=forgery_paths,
    )


def score_writer(
    writer_id: str,
    split: WriterSplit,
    encoded: dict[Path, EncodedSignature],
    distance_fn: DistanceFn,
) -> list[QueryScore]:
    """Score every query against `split.reference_paths` SEPARATELY (K
    distances per query), then average - never against a pre-averaged
    prototype (see module docstring)."""
    scores: list[QueryScore] = []
    labeled_queries = (
        [(p, "genuine", 1) for p in split.genuine_query_paths]
        + [(p, "forgery", 0) for p in split.forgery_query_paths]
    )

    for query_path, query_type, label in labeled_queries:
        per_reference_distances = [
            distance_fn(encoded[query_path], encoded[reference_path])
            for reference_path in split.reference_paths
        ]
        distance = float(np.mean(per_reference_distances))
        scores.append(QueryScore(
            writer_id=writer_id, query_path=query_path, query_type=query_type,
            label=label, distance=distance, score=-distance,
        ))

    return scores


def _score_all_writers_batched(
    writer_splits: dict[str, WriterSplit],
    encoded: dict[Path, EncodedSignature],
    batch_distance_fn: BatchDistanceFn,
    batch_size: int,
    show_progress: bool,
) -> list[QueryScore]:
    """Same result as calling `score_writer` per writer, but every
    (query, reference) comparison across ALL writers in this draw is
    flattened into one list first, then scored `batch_size` at a time
    through `batch_distance_fn` - chunk boundaries deliberately ignore
    writer/query boundaries so every chunk (except the last) is full,
    maximizing how much work each batched Sinkhorn call actually covers."""
    rows: list[tuple[str, Path, str, int, Path]] = []
    for writer_id, split in writer_splits.items():
        labeled_queries = (
            [(p, "genuine", 1) for p in split.genuine_query_paths]
            + [(p, "forgery", 0) for p in split.forgery_query_paths]
        )
        for query_path, query_type, label in labeled_queries:
            for reference_path in split.reference_paths:
                rows.append((writer_id, query_path, query_type, label, reference_path))

    if not rows:
        return []

    num_chunks = (len(rows) + batch_size - 1) // batch_size
    chunk_starts = range(0, len(rows), batch_size)
    if show_progress:
        chunk_starts = tqdm(chunk_starts, desc="Scoring pairs (batched)", ncols=100, total=num_chunks)

    all_distances: list[float] = []
    for start in chunk_starts:
        chunk = rows[start : start + batch_size]
        pairs = [(encoded[query_path], encoded[reference_path]) for _, query_path, _, _, reference_path in chunk]
        all_distances.extend(batch_distance_fn(pairs))

    if len(all_distances) != len(rows):
        raise ValueError(
            f"batch_distance_fn returned {len(all_distances)} distances for {len(rows)} pairs - "
            "it must return exactly one distance per input pair, in order."
        )

    per_query_distances: dict[tuple[str, Path], list[float]] = {}
    per_query_meta: dict[tuple[str, Path], tuple[str, int]] = {}
    for (writer_id, query_path, query_type, label, _reference_path), distance in zip(rows, all_distances):
        key = (writer_id, query_path)
        per_query_distances.setdefault(key, []).append(distance)
        per_query_meta[key] = (query_type, label)

    scores: list[QueryScore] = []
    for (writer_id, query_path), distances in per_query_distances.items():
        query_type, label = per_query_meta[(writer_id, query_path)]
        mean_distance = float(np.mean(distances))
        scores.append(QueryScore(
            writer_id=writer_id, query_path=query_path, query_type=query_type,
            label=label, distance=mean_distance, score=-mean_distance,
        ))

    return scores


def run_verification_draw(
    model: DownstreamVerificationModel,
    writer_ids: list[str],
    dataset_dir: Path,
    dataset_name: str,
    distance_fn: DistanceFn,
    num_references: int,
    seed: int,
    device: torch.device,
    show_progress: bool = True,
    encoded_cache: Optional[dict[Path, EncodedSignature]] = None,
    batch_distance_fn: Optional[BatchDistanceFn] = None,
    batch_size: int = 64,
) -> list[QueryScore]:
    """One full reference draw over every writer in `writer_ids`: split
    references/queries per writer (all under the same `seed`, so the draw
    is reproducible as a whole), encode every needed signature exactly
    once, then score every writer's queries.

    `encoded_cache`, if given, is reused and extended ACROSS draws. This
    matters a lot: a draw only changes WHICH signatures are references and
    which are queries - the underlying image set is identical every time,
    so re-encoding it per draw is pure waste. A full protocol run makes
    ~10 draws over the same writers, and the per-epoch verification
    callback makes one per seed, so sharing a cache cuts encoding work by
    roughly that factor.

    CRITICAL: the cache is keyed by image path only, NOT by model state.
    It is therefore valid only while the model's weights are unchanged.
    Callers must create a fresh cache whenever the model changes - e.g.
    the per-epoch callback builds one per invocation, never one that
    outlives the epoch. Passing `None` (the default) preserves the old
    encode-every-time behaviour exactly.

    `batch_distance_fn`, if given, takes over scoring ENTIRELY (`distance_fn`
    is then unused) via `_score_all_writers_batched` - every (query,
    reference) comparison in this draw, across every writer, is scored
    `batch_size` pairs at a time instead of one pair per Python-level call.
    This is the fix for Step 4's verification protocol being far slower
    than training: the per-pair path was re-paying full GPU kernel-launch
    overhead for every single Sinkhorn solve instead of amortizing it
    across a batch. Leaving `batch_distance_fn=None` preserves the exact
    old per-pair behaviour (still correct, just slower) for callers that
    don't have a batched distance function (e.g. Method A alone, where
    batching buys nothing since each call is already just a vector norm).
    """
    writer_splits: dict[str, WriterSplit] = {}
    all_paths: list[Path] = []

    for writer_id in writer_ids:
        writer_dir = dataset_dir / writer_id
        split = split_references_and_queries(writer_dir, dataset_name, num_references, seed)
        writer_splits[writer_id] = split
        all_paths.extend(split.reference_paths)
        all_paths.extend(split.genuine_query_paths)
        all_paths.extend(split.forgery_query_paths)

    if encoded_cache is None:
        encoded = encode_all_signatures(model, all_paths, device, show_progress=show_progress)
    else:
        missing = [p for p in dict.fromkeys(all_paths) if p not in encoded_cache]
        if missing:
            encoded_cache.update(
                encode_all_signatures(model, missing, device, show_progress=show_progress)
            )
        encoded = encoded_cache

    if batch_distance_fn is not None:
        return _score_all_writers_batched(writer_splits, encoded, batch_distance_fn, batch_size, show_progress)

    all_scores: list[QueryScore] = []
    for writer_id, split in writer_splits.items():
        all_scores.extend(score_writer(writer_id, split, encoded, distance_fn))

    return all_scores
