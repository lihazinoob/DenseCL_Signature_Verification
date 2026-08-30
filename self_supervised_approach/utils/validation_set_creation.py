"""Writer-level SSL validation split, created once and frozen.

Companion to `test_set_creation.py`, but for a DIFFERENT purpose and with a
DIFFERENT inclusion rule:

- `test_set_creation.py`'s split is a labeled, held-out benchmark for the
  eventual downstream verification stage - Mendeley is excluded from it
  entirely because it has no genuine/forged labels to test with.
- THIS split is for tracking the unlabeled SSL contrastive loss (global +
  dense) on writers the pretraining loop never trains on, purely to catch
  the SSL-specific overfitting failure mode discussed in chat (the encoder
  exploiting training-image-specific quirks rather than learning
  generalizable stroke structure) - since it never touches labels, Mendeley
  is fully eligible and IS included here.

Per the user's explicit decision: no live labeled verification probe (that
idea was dropped after checking that DenseCL's own authors never monitored
representation quality during pretraining either - see chat), and no images
kept around as a "sanity checker" - only these loss numbers, tracked every
epoch, train vs. validation.

Validation writers are drawn from each dataset's pool AFTER test-writer
exclusion (via `test_set_creation.load_test_writer_ids`) - a validation
writer and a test writer can never be the same writer, by construction, not
by luck. Same overwrite-refusal safeguard as `test_set_creation.py`, for the
same reason: once used in any training run, a validation split must not
silently change.

VALIDATION_WRITER_COUNTS defaults (chosen as a small, ~10-14% slice of each
dataset's REMAINING pool after test exclusion - "small" per the user's own
framing, not a formal benchmark that needs a large sample):
  CEDAR:            5  of 40  remaining (55 total - 15 test)
  BHSig260_Bengali: 10 of 70  remaining (100 total - 30 test)
  BHSig260_Hindi:   15 of 130 remaining (160 total - 30 test)
  Mendeley:         20 of 200 remaining (200 total - 0 test)
Editable in place, same convention as `test_set_creation.py`'s
TEST_WRITER_COUNTS and `augment.py`'s `AugmentConfig`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from test_set_creation import (
    DATA_ROOT,
    SEED,
    list_writer_ids,
    load_test_writer_ids,
)
from test_set_creation import select_test_writers as _select_random_writers

SCRIPT_DIR = Path(__file__).resolve().parent
SELF_SUPERVISED_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SELF_SUPERVISED_DIR.parent / "data" / "validation_set_writer_split"

VALIDATION_WRITER_COUNTS: dict[str, int] = {
    "CEDAR": 5,
    "BHSig260_Bengali": 10,
    "BHSig260_Hindi": 15,
    "Mendeley": 20,
}


def build_validation_split(dataset_name: str, num_validation: int, seed: int = SEED) -> dict:
    dataset_dir = DATA_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    all_writer_ids = list_writer_ids(dataset_dir)
    test_writer_ids = load_test_writer_ids(dataset_name)  # empty set for Mendeley
    eligible_writer_ids = [w for w in all_writer_ids if w not in test_writer_ids]

    validation_writer_ids = _select_random_writers(eligible_writer_ids, num_validation, seed)

    return {
        "dataset": dataset_name,
        "total_writers": len(all_writer_ids),
        "num_test_writers_excluded": len(test_writer_ids),
        "num_eligible_writers": len(eligible_writer_ids),
        "num_validation_writers": len(validation_writer_ids),
        "seed": seed,
        "validation_writer_ids": validation_writer_ids,
    }


def save_validation_split(split: dict, output_dir: Path = OUTPUT_DIR, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split['dataset']}_validation_writers.json"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Refusing to overwrite by default - a validation "
            f"split, once used for any training run, must never silently change (see module "
            f"docstring). Pass overwrite=True / --overwrite only if you are certain no "
            f"training has happened against the existing split yet."
        )

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return out_path


def load_validation_writer_ids(dataset_name: str, split_dir: Path = OUTPUT_DIR) -> set[str]:
    """Read back a previously-created split. Returns an empty set if no
    split file exists for this dataset yet."""
    split_path = split_dir / f"{dataset_name}_validation_writers.json"
    if not split_path.exists():
        return set()
    with open(split_path) as f:
        return set(json.load(f)["validation_writer_ids"])


def main(overwrite: bool = False) -> None:
    for dataset_name, num_validation in VALIDATION_WRITER_COUNTS.items():
        split = build_validation_split(dataset_name, num_validation)
        out_path = save_validation_split(split, overwrite=overwrite)
        print(f"[{dataset_name}] {split['num_validation_writers']}/{split['num_eligible_writers']} eligible "
              f"writers held out for validation (of {split['total_writers']} total, "
              f"{split['num_test_writers_excluded']} already test-excluded) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create writer-level SSL validation splits per dataset.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting an existing split file. Only use this if no "
                              "training has happened against the existing split yet.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
