"""Final test-set evaluation for one trained downstream checkpoint.

Run this ONCE, after training finishes, on the run's `best_model.pt`. It is
the only thing in this project that touches the TEST writers - everything
during training sees validation writers only.

Reports BOTH threshold conventions side by side, which is not optional
bookkeeping (see `threshold.evaluate_test_optimal_tau_per_draw`):

  - STRICT: tau chosen on validation writers, applied unchanged to test.
    This thesis's own methodology, and the honest number.
  - SURDS CONVENTION: tau swept on the test scores themselves (their
    Eq. 8). Optimistic, but it is what SURDS's published 89.50% (Hindi) /
    87.34% (Bengali) actually mean, so it is the only number that can
    legitimately be placed next to theirs.

Reporting only the strict number against their optimistic one understates
this work by 1-2 accuracy points for a purely methodological reason - as
measured on a fixed model on 2026-09-03.

K defaults to 8, matching SURDS's own protocol (their Sec. III-D: "we
randomly select 8 genuine samples from each of the unseen writers and
consider them as reference signatures, and the mean of the distance metric
between each of the references with the queried sample is considered for
comparison with the threshold").

Usage:
    python evaluate_checkpoint.py <run_tag> [--dataset BHSig260_Hindi] [--k 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

EVALUATION_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = EVALUATION_DIR.parent
for sub in ("utils", "models", "datasets", "matching", "evaluation"):
    sys.path.insert(0, str(SUPERVISED_DIR / sub))

from writer_splits import DATA_ROOT, get_writer_split  # noqa: E402
from embedding_model import load_downstream_model  # noqa: E402
from protocol import (  # noqa: E402
    make_combined_batch_distance_fn,
    make_combined_distance_fn,
    make_method_a_distance_fn,
    run_full_protocol,
)

# SSL run name defaults to the pooled encoder, derived per-fold at call
# time (`f"all_data_ssl/{args.fold}"`) unless overridden by --ssl_run_name
# - see main(). fold_0's weights are byte-identical to the original
# "densecl_pretrain_v1" (same run, relocated into the fold-scoped layout).
SSL_CHECKPOINT_EPOCH = 50

# More draws than the per-epoch monitoring budget - this runs once, so the
# extra reference draws are cheap and tighten the reported spread.
VAL_SEEDS = [101, 202, 303, 404, 505]
TEST_SEEDS = [11, 22, 33, 44, 55]

# Step 4's combined-distance hyperparameters (must match trainer.py's
# LAMBDA_0/SINKHORN_EPSILON/SINKHORN_ITERATIONS - these are fixed, not
# learned, so evaluation must use the same values training did).
# VERIFICATION_BATCH_SIZE mirrors trainer.py's constant of the same name -
# see downstream_supervised_learning_approach.md SS9, "Verification
# protocol speed - fixed".
LAMBDA_0 = 1.0
SINKHORN_EPSILON = 0.05
SINKHORN_ITERATIONS = 50
VERIFICATION_BATCH_SIZE = 64

# For orientation in the printout only - never used in any computation.
PUBLISHED_REFERENCE = {
    "BHSig260_Hindi": "SURDS: Acc 89.50, FAR 12.01, FRR 8.98",
    "BHSig260_Bengali": "SURDS: Acc 87.34, FAR 19.89, FRR 5.42",
    "CEDAR": "(SURDS does not report CEDAR)",
}


def _show(title: str, summary: dict) -> None:
    print(f"\n  {title}")
    for key in ("roc_auc", "eer", "accuracy", "balanced_accuracy", "far", "frr", "tpr", "tnr"):
        mean = summary.get(f"{key}_mean")
        if mean is not None:
            print(f"    {key:<19} {mean:.4f} +/- {summary[f'{key}_std']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag", help="results/<ssl_run_name>/<dataset>/<run_tag>/")
    parser.add_argument("--dataset", default="BHSig260_Hindi")
    parser.add_argument("--fold", default="fold_0",
                         help="K-fold CV fold - selects the writer split "
                              "(data/*_writer_split/<fold>/); also feeds the default "
                              "--ssl_run_name if that is not given explicitly.")
    parser.add_argument(
        "--ssl_run_name", default=None,
        help="Which SSL pretraining run's encoder produced this checkpoint - MUST match "
             "trainer.py's RUN_NAME for the run being evaluated, or this loads the wrong "
             "encoder weights silently (the checkpoint's own trained heads would still load "
             "correctly, but every distance computed by the shared/frozen encoder layers "
             "would be wrong). Defaults to the pooled 'all_data_ssl/<fold>' run, matching "
             "every trainer.py run before SS16; pass e.g. 'Hindi_data_ssl/fold_0' for a "
             "single-dataset SSL run's checkpoint. Also determines the results directory "
             "(results/<ssl_run_name>/<dataset>/<run_tag>/), matching trainer.py's RESULTS_DIR.",
    )
    parser.add_argument("--k", type=int, default=8, help="number of reference signatures (SURDS uses 8)")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument(
        "--checkpoint_dataset", default=None,
        help="Zero-shot cross-dataset evaluation: which dataset the checkpoint was TRAINED "
             "on (selects where to load the checkpoint from). Defaults to --dataset, so every "
             "in-domain call is unaffected. Pass e.g. --checkpoint_dataset BHSig260_Hindi "
             "--dataset BHSig260_Bengali to load a Hindi-trained checkpoint's weights and "
             "evaluate them on Bengali's own validation/test writers and images (--dataset "
             "controls the evaluation data throughout; --checkpoint_dataset only controls "
             "where the checkpoint file is read from). Results save to a separate "
             "results/<ssl_run_name>/zero_shot_OSV/<run_tag>/test_results_K<k>_<dataset>_<fold>.json "
             "- never the checkpoint's own in-domain result file - so a zero-shot run can never "
             "overwrite or be confused with that checkpoint's in-domain evaluation.",
    )
    args = parser.parse_args()
    ssl_run_name = args.ssl_run_name or f"all_data_ssl/{args.fold}"
    checkpoint_dataset = args.checkpoint_dataset or args.dataset
    zero_shot = checkpoint_dataset != args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = get_writer_split(args.dataset, fold=args.fold)
    dataset_dir = DATA_ROOT / args.dataset

    checkpoint_run_dir = SUPERVISED_DIR / "results" / ssl_run_name / checkpoint_dataset / args.run_tag
    checkpoint_path = checkpoint_run_dir / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    if zero_shot:
        run_dir = SUPERVISED_DIR / "results" / ssl_run_name / "zero_shot_OSV" / args.run_tag
    else:
        run_dir = checkpoint_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset    : {args.dataset}  "
          f"({len(split.train_writer_ids)} train / {len(split.validation_writer_ids)} val / "
          f"{len(split.test_writer_ids)} test writers)")
    print(f"Fold       : {args.fold}")
    print(f"Run tag    : {args.run_tag}")
    print(f"SSL run    : {ssl_run_name}")
    print(f"K          : {args.k}")
    if zero_shot:
        print(f"Checkpoint dataset: {checkpoint_dataset}  [ZERO-SHOT - evaluating on a dataset the checkpoint never trained on]")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Auto-detect Step 4's combined distance from the checkpoint itself
    # (presence of local_projection weights) rather than requiring a CLI
    # flag - eliminates the failure mode of evaluating a Step 4 checkpoint
    # with the wrong (Method A only) distance function by accident.
    uses_combined_distance = "local_projection.weight" in state_dict
    local_embedding_dim = state_dict["local_projection.weight"].shape[0] if uses_combined_distance else None

    model = load_downstream_model(ssl_run_name, SSL_CHECKPOINT_EPOCH, device, local_embedding_dim=local_embedding_dim)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Checkpoint : {args.checkpoint} from epoch {checkpoint['epoch']} "
          f"(selection score {checkpoint.get('best_score', float('nan')):.4f})")

    if uses_combined_distance:
        print(f"Distance   : Step 4 combined (lambda_0={LAMBDA_0}, local_dim={local_embedding_dim}) "
              f"[batched, batch_size={VERIFICATION_BATCH_SIZE}]")
        distance_fn = make_combined_distance_fn(
            model, lambda_0=LAMBDA_0, sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        batch_distance_fn = make_combined_batch_distance_fn(
            model, lambda_0=LAMBDA_0, sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
    else:
        print("Distance   : Method A only (no local_projection in checkpoint)")
        distance_fn = make_method_a_distance_fn()
        batch_distance_fn = None

    start = time.time()
    result = run_full_protocol(
        model, dataset_dir, args.dataset, distance_fn,
        split.validation_writer_ids, split.test_writer_ids,
        num_references=args.k, val_seeds=VAL_SEEDS, test_seeds=TEST_SEEDS,
        device=device, show_progress=False,
        batch_distance_fn=batch_distance_fn, batch_size=VERIFICATION_BATCH_SIZE,
    )
    elapsed = time.time() - start

    print(f"\n===== TEST RESULTS ({elapsed:.0f}s, {len(TEST_SEEDS)} reference draws) =====")
    print(f"  tau selected on validation writers: {result.val_tau_star:.5f}")
    _show("STRICT - tau from validation, applied to test (our methodology):", result.test_summary)
    _show("SURDS CONVENTION - tau swept on test itself (their published basis):",
          result.test_summary_surds_convention)

    strict = result.test_summary["balanced_accuracy_mean"]
    surds = result.test_summary_surds_convention["balanced_accuracy_mean"]
    print(f"\n  convention gap: {(surds - strict) * 100:.2f} accuracy points on an identical model")
    if not zero_shot:
        # Suppressed in zero-shot mode: SURDS's published numbers are in-domain,
        # not zero-shot - printing them here would invite a misleading comparison.
        print(f"  published reference: {PUBLISHED_REFERENCE.get(args.dataset, 'n/a')}")

    per_writer = result.per_writer_auc
    worst = sorted(per_writer.items(), key=lambda kv: kv[1])[:5]
    print(f"\n  per-writer AUC over {len(per_writer)} test writers - 5 worst:")
    for writer_id, auc in worst:
        print(f"    writer {writer_id:>4}: {auc:.4f}")

    if zero_shot:
        out_path = run_dir / f"test_results_K{args.k}_{args.dataset}_{args.fold}.json"
    else:
        out_path = run_dir / f"test_results_K{args.k}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "checkpoint_dataset": checkpoint_dataset,
        "zero_shot": zero_shot,
        "fold": args.fold,
        "run_tag": args.run_tag,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "distance_method": "combined" if uses_combined_distance else "method_a",
        "lambda_0": LAMBDA_0 if uses_combined_distance else None,
        "local_embedding_dim": local_embedding_dim,
        "num_references": args.k,
        "val_seeds": VAL_SEEDS,
        "test_seeds": TEST_SEEDS,
        "val_tau_star": result.val_tau_star,
        "val_balanced_acc_mean": result.val_balanced_acc_mean,
        "test_strict": result.test_summary,
        "test_surds_convention": result.test_summary_surds_convention,
        "per_writer_auc": per_writer,
    }, indent=2))
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
