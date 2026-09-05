"""Fixed 4-tuple dataset for downstream dual-triplet VALIDATION loss tracking.

Adapted from the SURDS-era `FixedDualTripletDataset`
(`Thesis_Final/downstream_verification/datasets/FixedDualTripletDataset.py`)
- see `docs/claude_response/downstream_supervised_learning_approach.md`,
"Validation loss curve: fixed sampling, not dynamic" for the full
reasoning. Unlike `DualTripletDataset` (training), the 4-tuples here are
built ONCE with a fixed seed and never change between epochs, so a rise or
fall in validation loss can only mean the model changed, not that this
epoch's random quadruples happened to be easier or harder than last
epoch's. This is a per-epoch training-health diagnostic (mirrors the SSL
phase's train-vs-validation loss columns), not the real target metric -
see the K-reference AUC/EER protocol in the same markdown file for that.

Same two deliberate changes from the old pipeline as `dual_triplet_dataset.py`:
  - Takes `writer_ids` + `dataset_dir` directly, genuine/forgery filename
    parsing folded in rather than a separate inventory step.
  - Uses this project's preprocessing (`preprocess_signature`) and binary
    {0,1} float32 tensor convention, matching the SSL-pretrained encoder's
    training input exactly (not the old pipeline's {-1,+1} normalization).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

DATASETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASETS_DIR))

from dual_triplet_dataset import load_signature_tensor, list_genuine_and_forged_images  # noqa: E402

DEFAULT_TUPLES_PER_ANCHOR = 4
# Distinct from DualTripletDataset's default training seed (42) - keeping
# them different is not load-bearing (the two datasets draw from disjoint
# writer pools anyway) but avoids any appearance of the validation draw
# being derived from the training seed.
DEFAULT_SEED = 314


@dataclass(frozen=True)
class FixedTripletRecord:
    anchor_writer_id: str
    inter_writer_id: str
    anchor_path: Path
    positive_path: Path
    negative_intra_path: Path
    negative_inter_path: Path


def build_fixed_triplet_records(
    writer_ids: list[str],
    dataset_dir: Path,
    dataset_name: str,
    tuples_per_anchor: int = DEFAULT_TUPLES_PER_ANCHOR,
    seed: int = DEFAULT_SEED,
) -> list[FixedTripletRecord]:
    """Deterministically build `tuples_per_anchor` fixed 4-tuples per
    genuine anchor. Positive and negative_intra cycle round-robin through
    each writer's own genuine/forgery pool (`tuple_index % len(...)`);
    negative_inter's writer is chosen via a per-anchor seeded RNG so it's
    reproducible without being the same other-writer every time. Calling
    this twice with the same arguments returns an identical list."""
    writer_to_genuine_paths: dict[str, list[Path]] = {}
    writer_to_forgery_paths: dict[str, list[Path]] = {}
    for writer_id in writer_ids:
        writer_dir = dataset_dir / writer_id
        if not writer_dir.is_dir():
            raise FileNotFoundError(f"Writer directory not found: {writer_dir}")

        genuine_paths, forgery_paths = list_genuine_and_forged_images(writer_dir, dataset_name)
        if len(genuine_paths) < 2:
            raise ValueError(
                f"Writer {writer_id} has fewer than two genuine signatures "
                f"(found {len(genuine_paths)}). Cannot form anchor-positive pairs."
            )
        if len(forgery_paths) < 1:
            raise ValueError(f"Writer {writer_id} has no forgery signatures.")

        writer_to_genuine_paths[writer_id] = genuine_paths
        writer_to_forgery_paths[writer_id] = forgery_paths

    sorted_writer_ids = sorted(writer_to_genuine_paths.keys(), key=int)
    if len(sorted_writer_ids) < 2:
        raise ValueError(
            f"Need at least two writers for inter-negative sampling (found {len(sorted_writer_ids)})."
        )

    records: list[FixedTripletRecord] = []
    for writer_id in sorted_writer_ids:
        inter_writer_candidates = [w for w in sorted_writer_ids if w != writer_id]
        genuine_paths = writer_to_genuine_paths[writer_id]
        forgery_paths = writer_to_forgery_paths[writer_id]

        for anchor_index, anchor_path in enumerate(genuine_paths):
            rng = random.Random(seed + (int(writer_id) * 1009) + anchor_index)
            positive_candidates = [p for p in genuine_paths if p != anchor_path]

            for tuple_index in range(tuples_per_anchor):
                positive_path = positive_candidates[tuple_index % len(positive_candidates)]
                negative_intra_path = forgery_paths[tuple_index % len(forgery_paths)]

                inter_writer_id = inter_writer_candidates[rng.randrange(len(inter_writer_candidates))]
                inter_genuine_paths = writer_to_genuine_paths[inter_writer_id]
                negative_inter_path = inter_genuine_paths[tuple_index % len(inter_genuine_paths)]

                records.append(FixedTripletRecord(
                    anchor_writer_id=writer_id,
                    inter_writer_id=inter_writer_id,
                    anchor_path=anchor_path,
                    positive_path=positive_path,
                    negative_intra_path=negative_intra_path,
                    negative_inter_path=negative_inter_path,
                ))

    return records


class FixedDualTripletDataset(Dataset):
    """Validation/test dataset backed by a pre-built, deterministic list of
    4-tuple records (`build_fixed_triplet_records`). Unlike
    `DualTripletDataset`, there is no `set_epoch()` - the same records are
    served every epoch, on purpose (see module docstring)."""

    def __init__(self, records: list[FixedTripletRecord]) -> None:
        super().__init__()
        if not records:
            raise ValueError("records is empty. Cannot build FixedDualTripletDataset.")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "anchor": load_signature_tensor(record.anchor_path),
            "positive": load_signature_tensor(record.positive_path),
            "negative_intra": load_signature_tensor(record.negative_intra_path),
            "negative_inter": load_signature_tensor(record.negative_inter_path),
            "anchor_writer_id": record.anchor_writer_id,
            "inter_writer_id": record.inter_writer_id,
        }
