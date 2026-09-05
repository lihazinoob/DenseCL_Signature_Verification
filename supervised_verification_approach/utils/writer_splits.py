"""Per-dataset writer split for the downstream supervised verification phase.

Design decision (see `docs/claude_response/downstream_supervised_learning_approach.md`,
"Writer split for the downstream phase"): reuse the existing SSL split files
directly instead of creating new ones.

  - TEST writers      = `data/test_set_writer_split/*.json` (already frozen).
  - VALIDATION writers = `data/validation_set_writer_split/*.json` (already
    frozen), reused as-is for downstream supervised validation too - these
    writers never receive a gradient update in either phase (SSL or
    downstream), by construction.
  - TRAINING writers  = every writer left over (all writers on disk minus
    test minus validation) - exactly the pool DenseCL already trained on
    unlabeled. Safe to reuse with labels now: SSL pretraining never touched
    a writer label or a genuine/forgery label, so there is no leakage in
    using the same writers, now labeled, for supervised training.

Mendeley is deliberately unsupported here - it has no genuine/forged
labels at all (plain numbered files only, see `test_set_creation.py`'s
module docstring), so it cannot participate in a supervised verification
task in any role (train, validation, or test).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = UTILS_DIR.parent  # DenseCL_approach/supervised_verification_approach
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent  # DenseCL_approach
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))

from test_set_creation import list_writer_ids, load_test_writer_ids  # noqa: E402
from validation_set_creation import load_validation_writer_ids  # noqa: E402

DATA_ROOT = DENSECL_APPROACH_DIR / "data" / "all"

# Every dataset with usable genuine/forged labels for supervised
# verification - Mendeley is excluded (see module docstring).
SUPPORTED_DATASETS: tuple[str, ...] = ("CEDAR", "BHSig260_Bengali", "BHSig260_Hindi")


@dataclass(frozen=True)
class WriterSplit:
    dataset: str
    train_writer_ids: list[str]
    validation_writer_ids: list[str]
    test_writer_ids: list[str]

    @property
    def total_writers(self) -> int:
        return len(self.train_writer_ids) + len(self.validation_writer_ids) + len(self.test_writer_ids)


def get_writer_split(dataset_name: str) -> WriterSplit:
    """Derive the downstream train/validation/test writer split for one
    dataset. Training writers are computed, not stored - they're simply
    whichever writers are on disk and not claimed by the test or
    validation split files."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"'{dataset_name}' has no genuine/forged labels and cannot be used for "
            f"supervised verification (see module docstring). Supported datasets: {SUPPORTED_DATASETS}"
        )

    dataset_dir = DATA_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    all_writer_ids = set(list_writer_ids(dataset_dir))
    test_writer_ids = load_test_writer_ids(dataset_name)
    validation_writer_ids = load_validation_writer_ids(dataset_name)

    if not test_writer_ids:
        raise ValueError(
            f"No test split found for '{dataset_name}' - run test_set_creation.py first."
        )
    if not validation_writer_ids:
        raise ValueError(
            f"No validation split found for '{dataset_name}' - run validation_set_creation.py first."
        )

    overlap = test_writer_ids & validation_writer_ids
    if overlap:
        raise ValueError(
            f"'{dataset_name}' has {len(overlap)} writer(s) in BOTH the test and validation "
            f"splits - this must never happen: {sorted(overlap, key=int)}"
        )

    unknown_test = test_writer_ids - all_writer_ids
    unknown_validation = validation_writer_ids - all_writer_ids
    if unknown_test or unknown_validation:
        raise ValueError(
            f"'{dataset_name}' split file(s) reference writer(s) not found on disk under "
            f"{dataset_dir}: test={sorted(unknown_test, key=int)}, validation={sorted(unknown_validation, key=int)}"
        )

    train_writer_ids = all_writer_ids - test_writer_ids - validation_writer_ids

    return WriterSplit(
        dataset=dataset_name,
        train_writer_ids=sorted(train_writer_ids, key=int),
        validation_writer_ids=sorted(validation_writer_ids, key=int),
        test_writer_ids=sorted(test_writer_ids, key=int),
    )


def main() -> None:
    print(f"{'Dataset':<20} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    for dataset_name in SUPPORTED_DATASETS:
        split = get_writer_split(dataset_name)
        print(
            f"{dataset_name:<20} {split.total_writers:>6} "
            f"{len(split.train_writer_ids):>6} {len(split.validation_writer_ids):>6} {len(split.test_writer_ids):>6}"
        )


if __name__ == "__main__":
    main()
