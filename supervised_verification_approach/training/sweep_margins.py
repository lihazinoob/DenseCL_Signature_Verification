"""One-time margin sweep for the plain global-embedding double-margin loss
(`loss/double_margin_loss.py`, `LOSS_TYPE="double_margin"` in `trainer.py`
- Steps 1-3, and now the global-only-head ablation cells, ICCIT §9.2 Tier
1 run (A)).

`m`/`n` aren't published anywhere DetailSemNet reports (see that module's
docstring) - this script picks them empirically instead of guessing, by
looking at where genuine-pair and negative-pair distances actually land
for a given SSL-pretrained encoder, measured on VALIDATION writers only.

Not a training loop: no gradients, one forward pass per validation image
(cached by path), one Euclidean distance per pair - only the embedding
space is being *read*, not changed.

REWRITTEN 2026-09-10 (ICCIT §9.2 Tier 1, run A): the original version of
this script measured margins against an EXISTING trained checkpoint
(`--run_tag`, defaulting to Step 1's), because at the time every
global-only run was the next rung of a ladder that already had a prior
rung's checkpoint to warm-measure from. The global-only-head-at-Cell-C
ablation this script now also serves has no such prior checkpoint for
the per-dataset in-domain encoders (`Hindi_data_ssl/fold_0` etc. never
had a plain `double_margin` run at all - every in-domain run so far used
`double_margin_combined`). So this now follows the SAME proxy convention
`sweep_margins_combined.py` already uses for exactly this situation: the
raw SSL-pretrained encoder (whatever `--ssl_run_name` names) + a FRESHLY
initialized global projector - exactly the state a real Cell C run starts
from (the encoder's `stem`/`stage1-4` are all about to become trainable,
so measuring with `trainable_encoder_stages=("stage4",)` here vs. the
real run's full-unfreeze tuple makes no difference to which WEIGHTS are
loaded - only which ones get gradients later - so this proxy is exact,
not approximate, unlike the combined script's local-branch caveat).

The old `--run_tag`/`--checkpoint` behavior (measure an existing trained
checkpoint directly) is kept as an option (`--run_tag`, opt-in) for any
future use that still has a prior checkpoint to warm-measure from - e.g.
re-deriving Step 2's original margins for the record. Omit it (the
default) for a from-scratch sweep against a raw SSL checkpoint.

ADDED 2026-09-10 (ICCIT §9.1/§9.2, the random-init control, §4.3): a
`--random_init` flag for measuring a RANDOMLY initialized encoder instead
of a pretrained one - a pretrained encoder's distance distribution says
nothing about a random one's (no learned structure at all), so reusing
any prior sweep's margins here would be a real mismatch, not just an
approximation. Uses `models.embedding_model.load_random_init_downstream_model`,
which already existed but was never wired into a training or sweep
script until now. `--seed` also seeds the random encoder's OWN weight
initialization here (not just the writer/tuple sampling as in every other
mode), so this sweep measures the exact same random weights the real
training run will start from if it uses the same seed - see
`trainer.py`'s `RANDOM_INIT_SEED` for where that must match.

Usage:
    python sweep_margins.py --dataset BHSig260_Hindi --ssl_run_name Hindi_data_ssl/fold_0
    python sweep_margins.py --dataset BHSig260_Bengali --ssl_run_name Bengali_data_ssl/fold_0
    python sweep_margins.py --dataset CEDAR --ssl_run_name CEDAR_data_ssl/fold_0
    # random-init control (ICCIT §4.3) - no SSL checkpoint at all:
    python sweep_margins.py --dataset BHSig260_Hindi --random_init --seed 42
    # legacy: measure an existing checkpoint directly instead of a raw SSL encoder
    python sweep_margins.py --dataset BHSig260_Hindi --run_tag step1_baseline_dualtriplet_frozen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = TRAINING_DIR.parent
for sub in ("utils", "models", "datasets", "evaluation"):
    sys.path.insert(0, str(SUPERVISED_DIR / sub))

from writer_splits import DATA_ROOT, get_writer_split  # noqa: E402
from embedding_model import load_downstream_model, load_random_init_downstream_model  # noqa: E402
from fixed_dual_triplet_dataset import build_fixed_triplet_records  # noqa: E402
from encoding import encode_all_signatures, method_a_distance_cached  # noqa: E402

SSL_RUN_NAME_DEFAULT = "all_data_ssl/fold_0"  # legacy default - pass --ssl_run_name explicitly for any per-dataset in-domain sweep
SSL_CHECKPOINT_EPOCH = 50
PERCENTILES = (10, 25, 50, 75, 90)
CANDIDATE_PERCENTILES = (30, 50, 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BHSig260_Hindi")
    parser.add_argument("--fold", default="fold_0", help="Fold subdirectory the writer split (and, for --run_tag, the checkpoint) lives under.")
    parser.add_argument(
        "--ssl_run_name", default=SSL_RUN_NAME_DEFAULT,
        help="Which SSL pretraining run's raw encoder to measure against (fresh global "
             "projector on top - see module docstring). Pass the matched per-dataset run "
             "for an in-domain sweep, e.g. 'Hindi_data_ssl/fold_0'. Ignored if --run_tag is given.",
    )
    parser.add_argument(
        "--run_tag", default=None,
        help="LEGACY: measure an existing trained checkpoint directly instead of a raw SSL "
             "encoder + fresh projector (e.g. 'step1_baseline_dualtriplet_frozen'). Looked up "
             "under results/all_data_ssl/<dataset>/<run_tag>/checkpoints/. Omit for the default, "
             "from-scratch raw-encoder proxy that every current use of this script wants.",
    )
    parser.add_argument(
        "--random_init", action="store_true",
        help="Measure a RANDOMLY initialized encoder (ICCIT §4.3's random-init control) instead "
             "of a pretrained SSL checkpoint - no learned structure at all, so this is NOT "
             "expected to land near any pretrained encoder's margins. Takes priority over "
             "--ssl_run_name/--run_tag if given. --seed also seeds the encoder's own random "
             "weights here (see module docstring) - use the SAME --seed value that the real "
             "training run will use, or this sweep measures different random weights than the "
             "ones actually trained.",
    )
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--tuples_per_anchor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="Force a device instead of auto-detecting cuda. This script only does inference "
             "(encoding a few hundred images + a distance computation) - cheap enough to run "
             "on CPU in a couple minutes rather than contend with a concurrent GPU job.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = get_writer_split(args.dataset, fold=args.fold)
    dataset_dir = DATA_ROOT / args.dataset

    if args.random_init:
        # No SSL checkpoint at all - the exact random weights depend on
        # torch's global RNG state at construction time, so seed it
        # explicitly with --seed (default 314, but pass the SAME seed the
        # real training run will use, e.g. 42, or this sweep measures
        # different random weights than the ones actually trained).
        torch.manual_seed(args.seed)
        model = load_random_init_downstream_model(
            trainable_encoder_stages=("stage4",), local_embedding_dim=None, device=device,
        )
        model.eval()
        print(
            f"Using a RANDOMLY initialized encoder (seed={args.seed}) + a freshly initialized "
            f"global projector - no SSL pretraining, no local branch (ICCIT §4.3 control)."
        )
    elif args.run_tag is not None:
        run_dir = SUPERVISED_DIR / "results" / "all_data_ssl" / args.dataset / args.run_tag
        checkpoint_path = run_dir / "checkpoints" / args.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
        model = load_downstream_model(SSL_RUN_NAME_DEFAULT, SSL_CHECKPOINT_EPOCH, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        print(f"[legacy mode] Loaded {args.run_tag}/{args.checkpoint} (epoch {checkpoint['epoch']})")
    else:
        # Raw SSL-pretrained encoder + FRESH global projector - exactly the
        # state a real Cell A/B/C run starts from, since load_downstream_model
        # only ever loads the encoder from the SSL checkpoint; the projector
        # head is always freshly initialized regardless of which encoder
        # stages will later be trainable. `trainable_encoder_stages` here
        # only controls which weights get gradients LATER during actual
        # training - it does not change which weights are loaded now, so
        # this measurement is valid for Cell A, B, or C equally.
        model = load_downstream_model(
            args.ssl_run_name, SSL_CHECKPOINT_EPOCH, device,
            trainable_encoder_stages=("stage4",), local_embedding_dim=None,
        )
        model.to(device).eval()
        print(
            f"Using the raw SSL-pretrained encoder ({args.ssl_run_name}, epoch "
            f"{SSL_CHECKPOINT_EPOCH}) + a freshly initialized global projector - no local branch."
        )

    records = build_fixed_triplet_records(
        split.validation_writer_ids, dataset_dir, args.dataset,
        tuples_per_anchor=args.tuples_per_anchor, seed=args.seed,
    )
    print(f"Validation writers: {len(split.validation_writer_ids)} | pair records: {len(records) * 3}")

    all_paths = sorted({
        p for r in records
        for p in (r.anchor_path, r.positive_path, r.negative_intra_path, r.negative_inter_path)
    })
    encoded = encode_all_signatures(model, all_paths, device, show_progress=True)

    positive_distances: list[float] = []
    negative_distances: list[float] = []
    for record in records:
        positive_distances.append(
            method_a_distance_cached(encoded[record.anchor_path], encoded[record.positive_path])
        )
        negative_distances.append(
            method_a_distance_cached(encoded[record.anchor_path], encoded[record.negative_intra_path])
        )
        negative_distances.append(
            method_a_distance_cached(encoded[record.anchor_path], encoded[record.negative_inter_path])
        )

    positive_distances_arr = np.array(positive_distances)
    negative_distances_arr = np.array(negative_distances)

    print(f"\nGenuine-genuine pairs (n={len(positive_distances_arr)}): mean={positive_distances_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(positive_distances_arr, p):.4f}")

    print(f"\nNegative pairs (n={len(negative_distances_arr)}): mean={negative_distances_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(negative_distances_arr, p):.4f}")

    print(
        "\nCandidate (m, n) margins, from percentiles of the observed distributions, "
        "with the resulting active fraction (share of pairs that would still produce "
        "a nonzero gradient under that margin):"
    )
    print(f"{'m %ile':>8} {'n %ile':>8} {'m':>8} {'n':>8} {'pos_active':>11} {'neg_active':>11}")
    for m_pct in CANDIDATE_PERCENTILES:
        for n_pct in CANDIDATE_PERCENTILES:
            m = float(np.percentile(positive_distances_arr, m_pct))
            n = float(np.percentile(negative_distances_arr, n_pct))
            if m >= n:
                continue
            pos_active = float((positive_distances_arr > m).mean())
            neg_active = float((negative_distances_arr < n).mean())
            print(f"{m_pct:>7}% {n_pct:>7}% {m:>8.4f} {n:>8.4f} {pos_active:>10.1%} {neg_active:>10.1%}")


if __name__ == "__main__":
    main()
