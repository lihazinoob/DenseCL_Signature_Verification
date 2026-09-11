"""One-time margin sweep for Step 4's combined-distance double-margin loss
(`loss/double_margin_distance_loss.py`).

Step 2/3's margins (0.46/0.96) were swept on pure global-embedding
distance and do NOT transfer here - the combined distance
(`lambda_0 * dis_global + dis_struct`) has a different composition and
scale. Unlike that earlier sweep, there is no already-trained checkpoint
to measure directly: the local branch (`local_projection`) has never been
trained, so its contribution can only be approximated. The proxy used
here (per `downstream_supervised_learning_approach.md` SS9/SS10): Step
3a's FINE-TUNED encoder (its `stage4` weights, already adapted by
supervised training) + a FRESHLY initialized global projector + a
FRESHLY initialized local projection - exactly the state a real Step 4
`stage4`-unfrozen run starts from (no warm start on either head).

Treat the result as a starting guess, not a settled value - watch
`positive_active_rate`/`negative_active_rate` in the first 1-2 epochs of
the real run and adjust if either sits near 0% or 100%.

`--skip_step3_encoder` uses the raw SSL-pretrained encoder instead of a
Step 3a checkpoint, for datasets with no Step 3a run (e.g. Bengali, which
went straight from SSL pretraining to Step 4). trainer.py's real Step 4
run always starts stage4 from the raw SSL checkpoint anyway, so this is
not a lesser proxy for those datasets.

`--ssl_run_name` selects WHICH SSL pretraining run's encoder to measure
against - defaults to the pooled `all_data_ssl/fold_0` run every prior
sweep used. Pass a single-dataset SSL run (e.g. `Hindi_data_ssl/fold_0`)
for the matched-domain/in-domain study: the margins must be measured on
the SAME encoder `trainer.py`'s `RUN_NAME` will actually load downstream,
or the sweep is measuring the wrong model's distance distribution.

ADDED 2026-09-10 (ICCIT §4.3/§9.1, the random-init control): a
`--random_init` flag for measuring a RANDOMLY initialized encoder instead
of any SSL checkpoint - takes priority over `--ssl_run_name`/
`--skip_step3_encoder`/the Step-3 override (none of those apply when
there is no pretrained encoder to start from). This is the control for
"did the SSL pretraining matter at all", so it deliberately keeps the
REAL reported architecture (combined distance, local branch included) and
changes only the encoder's starting weights - the axis this script's
`--skip_step3_encoder`/`--ssl_run_name` flags vary is a DIFFERENT
question (which pretrained encoder) that no longer applies once there is
no pretraining at all. `--seed` also seeds the random encoder's OWN
weight initialization here (not just the writer/tuple sampling) - pass
the SAME seed the real training run will use, or this sweep measures
different random weights than the ones actually trained.

ADDED 2026-09-10 (Stage 2 of the random-init control): a `--full_checkpoint`
flag for measuring against a REAL, already-trained checkpoint (encoder +
projector + local_projection all loaded together) instead of building any
proxy. This is for the random-init control's two-stage margin fix: Stage 1
(`--random_init`, t=0, before any gradient has flowed) turned out to be an
inadequate proxy for this specific architecture - a randomly-initialized,
fully-unfrozen encoder's raw/local feature scale is not stable at
initialization and drifts fast (~5.5x within one epoch) once training
starts, unlike a pretrained encoder's already-mature scale. Stage 2 fixes
this by re-sweeping against a REAL post-gradient-flow checkpoint (e.g. a
genuine `epoch1.pt` from one real epoch of training) instead of a t=0
snapshot. Takes priority over `--random_init`/`--ssl_run_name`/
`--skip_step3_encoder` (all irrelevant once there is a full checkpoint to
load directly).

Usage:
    python sweep_margins_combined.py [--dataset BHSig260_Hindi] [--step3_run_tag step3a_doublemargin_stage4] [--local_dim 128]
    python sweep_margins_combined.py --dataset BHSig260_Bengali --skip_step3_encoder
    python sweep_margins_combined.py --dataset BHSig260_Hindi --skip_step3_encoder --ssl_run_name Hindi_data_ssl/fold_0
    # random-init control (ICCIT §4.3) - no SSL checkpoint at all:
    python sweep_margins_combined.py --dataset BHSig260_Hindi --random_init --seed 42
    # random-init control, STAGE 2 - re-sweep against a real epoch-1 checkpoint:
    python sweep_margins_combined.py --dataset BHSig260_Hindi --full_checkpoint "../results/random_init_control/fold_0/BHSig260_Hindi/step4_full_unfrozen_combined_randominit/checkpoints/epoch1.pt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = TRAINING_DIR.parent
for sub in ("utils", "models", "datasets", "matching", "evaluation"):
    sys.path.insert(0, str(SUPERVISED_DIR / sub))

from writer_splits import DATA_ROOT, get_writer_split  # noqa: E402
from embedding_model import load_downstream_model, load_random_init_downstream_model  # noqa: E402
from fixed_dual_triplet_dataset import build_fixed_triplet_records  # noqa: E402
from encoding import encode_all_signatures, combined_distance_cached  # noqa: E402

SSL_RUN_NAME = "all_data_ssl/fold_0"  # relocated from "densecl_pretrain_v1" by the fold-scoped layout - byte-identical weights, see trainer.py's RUN_NAME comment
SSL_CHECKPOINT_EPOCH = 50
PERCENTILES = (10, 25, 50, 75, 90)
CANDIDATE_PERCENTILES = (30, 50, 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BHSig260_Hindi")
    parser.add_argument("--fold", default="fold_0", help="Fold subdirectory the Step 3 checkpoint lives under.")
    parser.add_argument(
        "--ssl_run_name", default=SSL_RUN_NAME,
        help="Which SSL pretraining run's encoder to use as the proxy's starting point (before "
             "any --skip_step3_encoder / Step-3 override). Defaults to the pooled all_data_ssl "
             "run every prior sweep used; pass a single-dataset run (e.g. 'Hindi_data_ssl/fold_0') "
             "for a matched-domain/in-domain sweep - the margins must come from the SAME encoder "
             "that trainer.py's RUN_NAME will actually load, or the proxy measurement is for the "
             "wrong model.",
    )
    parser.add_argument("--step3_run_tag", default="step3a_doublemargin_stage4")
    parser.add_argument("--step3_checkpoint", default="best_model.pt")
    parser.add_argument(
        "--skip_step3_encoder", action="store_true",
        help="Use the raw SSL-pretrained encoder as-is instead of overriding stage4 with a "
             "Step 3a fine-tuned checkpoint. Use this when no Step 3a run exists for the "
             "dataset (e.g. Bengali, which went straight from SSL pretraining to Step 4 - "
             "Step 3a was a Hindi-only exploratory stage). Note that trainer.py's real Step 4 "
             "run always starts stage4 from this same raw SSL checkpoint regardless of "
             "dataset, so this is not a lesser proxy - if anything it matches the real start "
             "state more closely than borrowing another dataset's fine-tuned weights would.",
    )
    parser.add_argument(
        "--random_init", action="store_true",
        help="Measure a RANDOMLY initialized encoder (ICCIT §4.3's random-init control) instead "
             "of any SSL checkpoint - no learned structure at all. Takes priority over "
             "--ssl_run_name/--skip_step3_encoder/--step3_run_tag (none apply without a "
             "pretrained encoder to start from). --seed also seeds the encoder's own random "
             "weights here - use the SAME --seed value the real training run will use.",
    )
    parser.add_argument(
        "--full_checkpoint", default=None,
        help="Path to a FULL DownstreamVerificationModel checkpoint (encoder + projector + "
             "local_projection all saved together, e.g. a mid-training epochN.pt) to load "
             "in full and measure directly - the random-init control's Stage 2 fix (see "
             "module docstring). Takes priority over --random_init/--ssl_run_name/"
             "--skip_step3_encoder/--step3_run_tag (all irrelevant once there is a real "
             "checkpoint to load).",
    )
    parser.add_argument("--local_dim", type=int, default=128)
    parser.add_argument("--lambda0", type=float, default=1.0)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iterations", type=int, default=50)
    parser.add_argument("--tuples_per_anchor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="Force a device instead of auto-detecting cuda. Useful when the GPU is already busy "
             "with a real training run (e.g. an SSL pretraining job) - this script only does "
             "inference (encoding a few hundred images + a distance computation), cheap enough to "
             "run on CPU in a couple minutes rather than contend with or slow down a concurrent "
             "GPU job. Defaults to cuda if available, matching every prior sweep's behavior.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = get_writer_split(args.dataset, fold=args.fold)
    dataset_dir = DATA_ROOT / args.dataset

    if args.full_checkpoint:
        # Real, already-trained checkpoint (Stage 2 of the random-init
        # control - see module docstring). The architecture built here is
        # discarded immediately by the full state_dict load below; only
        # local_embedding_dim needs to match (determines whether
        # local_projection exists at all).
        checkpoint_path = Path(args.full_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
        model = load_random_init_downstream_model(
            trainable_encoder_stages=(), local_embedding_dim=args.local_dim, device=device,
        )
        state = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=True)
        print(
            f"Loaded FULL model state (encoder + projector + local_projection) from "
            f"{checkpoint_path}. Missing: {len(missing)}, unexpected: {len(unexpected)}"
        )
    elif args.random_init:
        # No SSL checkpoint anywhere in this model - encoder, projector,
        # AND local_projection are all fresh random init. Seed explicitly
        # so this measures the exact same random weights the real
        # training run starts from, if it uses the same seed.
        torch.manual_seed(args.seed)
        model = load_random_init_downstream_model(
            trainable_encoder_stages=("stage4",), local_embedding_dim=args.local_dim, device=device,
        )
        print(
            f"Using a RANDOMLY initialized encoder (seed={args.seed}) + fresh projector + fresh "
            f"local_projection - no SSL pretraining at all (ICCIT §4.3 control, combined distance "
            f"architecture kept identical to the real reported system)."
        )
    else:
        # Fresh model: encoder = original SSL checkpoint (stage4 trainable),
        # projector + local_projection = fresh random init - see module
        # docstring for why the encoder gets overwritten next but the two
        # heads deliberately do not.
        model = load_downstream_model(
            args.ssl_run_name, SSL_CHECKPOINT_EPOCH, device,
            trainable_encoder_stages=("stage4",), local_embedding_dim=args.local_dim,
        )

        if args.skip_step3_encoder:
            print(
                "Using the raw SSL-pretrained encoder as-is (--skip_step3_encoder) - stage4 not "
                "yet fine-tuned by any supervised stage. Projector + local_projection are fresh "
                "random init, same as the Step 3a proxy path."
            )
        else:
            step3_checkpoint_path = (
                SUPERVISED_DIR / "results" / "all_data_ssl" / args.fold / args.dataset / args.step3_run_tag
                / "checkpoints" / args.step3_checkpoint
            )
            if not step3_checkpoint_path.exists():
                raise FileNotFoundError(f"No Step 3 checkpoint at {step3_checkpoint_path}")
            step3_state = torch.load(step3_checkpoint_path, map_location=device)["model_state_dict"]
            encoder_state = {k[len("encoder."):]: v for k, v in step3_state.items() if k.startswith("encoder.")}
            missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=True)
            print(
                f"Loaded {args.step3_run_tag}'s fine-tuned encoder weights over the SSL checkpoint's "
                f"(projector + local_projection left at fresh random init). Missing: {len(missing)}, "
                f"unexpected: {len(unexpected)}"
            )
    model.eval()

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
        positive_distances.append(combined_distance_cached(
            encoded[record.anchor_path], encoded[record.positive_path], model.local_projection,
            args.lambda0, args.sinkhorn_epsilon, args.sinkhorn_iterations,
        ))
        negative_distances.append(combined_distance_cached(
            encoded[record.anchor_path], encoded[record.negative_intra_path], model.local_projection,
            args.lambda0, args.sinkhorn_epsilon, args.sinkhorn_iterations,
        ))
        negative_distances.append(combined_distance_cached(
            encoded[record.anchor_path], encoded[record.negative_inter_path], model.local_projection,
            args.lambda0, args.sinkhorn_epsilon, args.sinkhorn_iterations,
        ))

    positive_arr = np.array(positive_distances)
    negative_arr = np.array(negative_distances)

    print(f"\nGenuine-genuine pairs (n={len(positive_arr)}): mean={positive_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(positive_arr, p):.4f}")
    print(f"\nNegative pairs (n={len(negative_arr)}): mean={negative_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(negative_arr, p):.4f}")

    print(
        "\nCandidate (m, n) margins, from percentiles of the observed distributions, "
        "with the resulting active fraction:"
    )
    print(f"{'m %ile':>8} {'n %ile':>8} {'m':>8} {'n':>8} {'pos_active':>11} {'neg_active':>11}")
    for m_pct in CANDIDATE_PERCENTILES:
        for n_pct in CANDIDATE_PERCENTILES:
            m = float(np.percentile(positive_arr, m_pct))
            n = float(np.percentile(negative_arr, n_pct))
            if m >= n:
                continue
            pos_active = float((positive_arr > m).mean())
            neg_active = float((negative_arr < n).mean())
            print(f"{m_pct:>7}% {n_pct:>7}% {m:>8.4f} {n:>8.4f} {pos_active:>10.1%} {neg_active:>10.1%}")


if __name__ == "__main__":
    main()
