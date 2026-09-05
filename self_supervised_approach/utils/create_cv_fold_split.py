"""Creates one new K-fold cross-validation fold's writer split for a given
dataset, guaranteed disjoint from every already-created fold's held-out
(test + validation) writers for that dataset.

Only meant for datasets that actually rotate across CV folds. In this
project that is BHSig260_Hindi ONLY - Bengali/CEDAR/Mendeley are not part
of the downstream CV experiment (the ladder in
`docs/claude_response/downstream_supervised_learning_approach.md` only
ever evaluates Hindi), and CEDAR's pool (55 writers) is too small to
support 3 disjoint 20-writer folds anyway (55 / 20 = 2.75). Bengali/CEDAR/
Mendeley's splits should instead be copied byte-identical from fold_0 into
every new fold's directories, so every fold's SSL pretraining excludes the
same writers for those datasets - keeping them valid for the still-pending
Step 6 cross-dataset generalization check regardless of which fold's
encoder is used.

Same sampling method as `test_set_creation.select_test_writers`
(random.Random(seed).sample, sorted for a readable diff) - reused directly,
not reimplemented, so a new fold's writers are chosen exactly the same way
fold_0's were, just restricted to the still-unused pool and a new seed.

Usage:
    python create_cv_fold_split.py BHSig260_Hindi fold_1 --seed 43
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from test_set_creation import (  # noqa: E402
    DATA_ROOT,
    TEST_WRITER_COUNTS,
    OUTPUT_DIR as TEST_SPLIT_ROOT,
    list_writer_ids,
    select_test_writers,
)
from validation_set_creation import (  # noqa: E402
    VALIDATION_WRITER_COUNTS,
    OUTPUT_DIR as VALIDATION_SPLIT_ROOT,
)


def _existing_fold_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_"))


def already_used_writers(dataset_name: str) -> set[str]:
    """Union of every already-created fold's held-out (test + validation)
    writer IDs for this dataset - the pool a new fold must avoid."""
    used: set[str] = set()
    for fold_dir in _existing_fold_dirs(TEST_SPLIT_ROOT):
        split_path = fold_dir / f"{dataset_name}_test_writers.json"
        if split_path.exists():
            used |= set(json.loads(split_path.read_text())["test_writer_ids"])
    for fold_dir in _existing_fold_dirs(VALIDATION_SPLIT_ROOT):
        split_path = fold_dir / f"{dataset_name}_validation_writers.json"
        if split_path.exists():
            used |= set(json.loads(split_path.read_text())["validation_writer_ids"])
    return used


def build_fold_split(dataset_name: str, fold_name: str, seed: int) -> tuple[dict, dict]:
    num_test = TEST_WRITER_COUNTS[dataset_name]
    num_validation = VALIDATION_WRITER_COUNTS[dataset_name]

    dataset_dir = DATA_ROOT / dataset_name
    all_writer_ids = list_writer_ids(dataset_dir)
    used = already_used_writers(dataset_name)

    still_used_but_missing = used - set(all_writer_ids)
    if still_used_but_missing:
        raise ValueError(
            f"{dataset_name}: {len(still_used_but_missing)} previously-used writer ID(s) "
            f"no longer exist on disk: {sorted(still_used_but_missing)} - refusing to "
            f"proceed, something is inconsistent."
        )

    eligible_for_test = [w for w in all_writer_ids if w not in used]
    if num_test > len(eligible_for_test):
        raise ValueError(
            f"{dataset_name}: need {num_test} test writers for {fold_name}, but only "
            f"{len(eligible_for_test)} of {len(all_writer_ids)} writers remain unused by "
            f"prior folds ({len(used)} already used). Pool exhausted for further folds."
        )
    test_writer_ids = select_test_writers(eligible_for_test, num_test, seed)

    eligible_for_validation = [w for w in eligible_for_test if w not in test_writer_ids]
    if num_validation > len(eligible_for_validation):
        raise ValueError(
            f"{dataset_name}: need {num_validation} validation writers for {fold_name}, but "
            f"only {len(eligible_for_validation)} remain after this fold's own test draw."
        )
    validation_writer_ids = select_test_writers(eligible_for_validation, num_validation, seed)

    test_split = {
        "dataset": dataset_name,
        "total_writers": len(all_writer_ids),
        "num_test_writers": len(test_writer_ids),
        "seed": seed,
        "fold": fold_name,
        "excluded_prior_fold_writers": len(used),
        "test_writer_ids": test_writer_ids,
    }
    validation_split = {
        "dataset": dataset_name,
        "total_writers": len(all_writer_ids),
        "num_test_writers_excluded": len(test_writer_ids),
        "num_eligible_writers": len(eligible_for_validation) + len(validation_writer_ids),
        "num_validation_writers": len(validation_writer_ids),
        "seed": seed,
        "fold": fold_name,
        "excluded_prior_fold_writers": len(used),
        "validation_writer_ids": validation_writer_ids,
    }

    overlap_within_fold = set(test_split["test_writer_ids"]) & set(validation_split["validation_writer_ids"])
    overlap_with_prior = (set(test_split["test_writer_ids"]) | set(validation_split["validation_writer_ids"])) & used
    if overlap_within_fold or overlap_with_prior:
        raise ValueError(
            f"{dataset_name}: disjointness check failed - within-fold overlap="
            f"{overlap_within_fold}, overlap-with-prior-folds={overlap_with_prior}. "
            f"This should be impossible given the sampling above; refusing to write output."
        )

    return test_split, validation_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(TEST_WRITER_COUNTS.keys()))
    parser.add_argument("fold_name", help="e.g. fold_1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print the split without writing files.")
    args = parser.parse_args()

    test_split, validation_split = build_fold_split(args.dataset, args.fold_name, args.seed)

    print(f"[{args.dataset} / {args.fold_name}] seed={args.seed}")
    print(f"  Already used by prior folds : {test_split['excluded_prior_fold_writers']}")
    print(f"  New test writers ({test_split['num_test_writers']}): {test_split['test_writer_ids']}")
    print(f"  New validation writers ({validation_split['num_validation_writers']}): {validation_split['validation_writer_ids']}")

    if args.dry_run:
        print("  [dry run] no files written.")
        return

    test_out_dir = TEST_SPLIT_ROOT / args.fold_name
    validation_out_dir = VALIDATION_SPLIT_ROOT / args.fold_name
    test_out_dir.mkdir(parents=True, exist_ok=True)
    validation_out_dir.mkdir(parents=True, exist_ok=True)

    test_out_path = test_out_dir / f"{args.dataset}_test_writers.json"
    validation_out_path = validation_out_dir / f"{args.dataset}_validation_writers.json"

    for path in (test_out_path, validation_out_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists - refusing to overwrite an existing fold split.")

    test_out_path.write_text(json.dumps(test_split, indent=2))
    validation_out_path.write_text(json.dumps(validation_split, indent=2))
    print(f"  Wrote -> {test_out_path}")
    print(f"  Wrote -> {validation_out_path}")


if __name__ == "__main__":
    main()
