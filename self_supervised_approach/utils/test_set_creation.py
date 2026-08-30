"""Writer-level test-set split, created once and frozen.

Per-dataset writer counts confirmed on disk: BHSig260_Bengali=100,
BHSig260_Hindi=160, CEDAR=55, Mendeley=200 writer directories. Bengali,
Hindi, and CEDAR all have genuine/forged-labeled files per writer
(`B-S-<id>-G/F-*.tif`, `original_*`/`forgeries_*.png` respectively), so they
can support the downstream supervised (labeled) stage. Mendeley's writer
folders (e.g. `a_(1)`) hold plain numbered files with no genuine/forged
distinction - it has no usable label for the downstream stage at all, so
(per the user's explicit decision) it contributes NO test writers and is
used entirely for SSL pretraining; it deliberately gets no JSON file here.

TEST_WRITER_COUNTS below is the only place the held-out writer counts are
defined - CEDAR 15, Bengali 30, Hindi 30 (user-specified). Which specific
writer IDs end up in that count is chosen at random (`random.Random(seed)`,
not "the first N" or any hand-picked set), per the user's requirement.

Held-out writers must never appear in EITHER pipeline stage - not the SSL
pretraining pool, not the downstream supervised pool - only in final
testing. Since this file is a writer-independent verifier's whole
leakage guarantee, `save_test_split` REFUSES to silently overwrite an
existing split file: once a split has been used for even one training run,
regenerating a different random sample under the same filename would
silently move some previously-held-out test writers into the training
pool (or vice versa) without anyone noticing. Pass `overwrite=True`
explicitly (or `--overwrite` on the CLI) only when no training has
happened yet against the existing split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SELF_SUPERVISED_DIR = SCRIPT_DIR.parent
DATA_ROOT = SELF_SUPERVISED_DIR.parent / "data" / "all"
OUTPUT_DIR = SELF_SUPERVISED_DIR.parent / "data" / "test_set_writer_split"

SEED = 42

# dataset folder name (under data/all/) -> number of writers to hold out for testing.
# Mendeley is intentionally absent - see module docstring.
TEST_WRITER_COUNTS: dict[str, int] = {
    "CEDAR": 15,
    "BHSig260_Bengali": 30,
    "BHSig260_Hindi": 30,
}


def list_writer_ids(dataset_dir: Path) -> list[str]:
    """Every writer directory name under a dataset folder, as strings (writer
    IDs are numeric-looking but not treated as ints - Mendeley's writer
    folders, e.g. `a_(1)`, are not numeric at all)."""
    return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())


def select_test_writers(writer_ids: list[str], num_test: int, seed: int) -> list[str]:
    """Randomly choose `num_test` writer IDs to hold out, using a fixed seed
    so the choice is reproducible if the split is intentionally rebuilt from
    scratch. Returned sorted purely for a readable, diff-friendly JSON file -
    sorting the OUTPUT does not affect the randomness of the selection."""
    if num_test > len(writer_ids):
        raise ValueError(
            f"Requested {num_test} test writers but only {len(writer_ids)} writers exist"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(writer_ids, num_test))


def build_test_split(dataset_name: str, num_test: int, seed: int = SEED) -> dict:
    dataset_dir = DATA_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    writer_ids = list_writer_ids(dataset_dir)
    test_writer_ids = select_test_writers(writer_ids, num_test, seed)

    return {
        "dataset": dataset_name,
        "total_writers": len(writer_ids),
        "num_test_writers": len(test_writer_ids),
        "seed": seed,
        "test_writer_ids": test_writer_ids,
    }


def save_test_split(split: dict, output_dir: Path = OUTPUT_DIR, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split['dataset']}_test_writers.json"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Refusing to overwrite by default - a test split, "
            f"once used for any training run, must never silently change (see module "
            f"docstring). Pass overwrite=True / --overwrite only if you are certain no "
            f"training has happened against the existing split yet."
        )

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return out_path


def load_test_writer_ids(dataset_name: str, split_dir: Path = OUTPUT_DIR) -> set[str]:
    """Read back a previously-created split - the function any SSL/downstream
    dataset-building code should call to exclude held-out test writers.
    Returns an empty set for a dataset with no split file (e.g. Mendeley),
    meaning "hold out nothing," which is the correct behavior for it."""
    split_path = split_dir / f"{dataset_name}_test_writers.json"
    if not split_path.exists():
        return set()
    with open(split_path) as f:
        return set(json.load(f)["test_writer_ids"])


def main(overwrite: bool = False) -> None:
    for dataset_name, num_test in TEST_WRITER_COUNTS.items():
        split = build_test_split(dataset_name, num_test)
        out_path = save_test_split(split, overwrite=overwrite)
        print(f"[{dataset_name}] {split['num_test_writers']}/{split['total_writers']} writers "
              f"held out for testing -> {out_path}")

    print("[Mendeley] no usable genuine/forged labels - 0 writers held out, "
          "entire dataset used for SSL pretraining only (no split file created).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create writer-level test splits per dataset.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting an existing split file. Only use this if no "
                              "training has happened against the existing split yet.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
