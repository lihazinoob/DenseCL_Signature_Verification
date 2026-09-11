"""Downstream supervised verification trainer - entry point.

Adapted from the SURDS-era `trainer.py`
(`Thesis_Final/downstream_verification/scripts/trainer.py`), config
constants at top per that convention, but parameterized by `DATASET_NAME`
instead of one hardcoded script per dataset - `writer_splits.py` already
handles all three supported datasets (CEDAR, BHSig260_Bengali,
BHSig260_Hindi) uniformly, so a separate near-identical script per dataset
would just duplicate this file three times.

`LOSS_TYPE` picks the training objective, and is the axis the experiment
ladder in `downstream_supervised_learning_approach.md` steps through:
  - `"dual_triplet"` (Step 1 only): the original quadruple/triplet path,
    unchanged - `DualTripletDataset` + `DualTripletLoss` + `run_one_epoch`
    (or the blended variants below when `ALPHA > 0.0`).
  - `"double_margin"` (Step 2+, i.e. also Step 3's unfrozen-encoder runs):
    `PairDataset` (wraps the same
    `DualTripletDataset`/`FixedDualTripletDataset` objects, decomposing
    each quadruple into its 3 labeled pairs) + `DoubleMarginLoss` +
    `run_one_epoch_pairs`. Only supports `ALPHA = 0.0` for now - combining
    it with the local/blended distance is Step 4 work that needs the
    trainable local projection and combined distance, neither of which
    exist yet (see the roadmap doc's "REMAINING in Step 0").
    `MARGIN_M`/`MARGIN_N` were chosen by `sweep_margins.py` against Step
    1's trained checkpoint (see that script's docstring and the comment
    at their definition below) - not published anywhere, since
    DetailSemNet doesn't report theirs either.
  - `"double_margin_combined"` (Step 4): `PairDataset` (same as above) +
    `DoubleMarginDistanceLoss` (the precomputed-scalar-distance variant,
    since the combined distance isn't a per-image embedding - see that
    module's docstring) + `run_one_epoch_combined`, which computes
    `dis = LAMBDA_0 * dis_global + dis_struct` per batch via
    `matching/combined_distance.py`'s batched Sinkhorn path. Requires
    `LOCAL_EMBEDDING_DIM` to be set (adds `model.local_projection`) and
    uses its own margins, `MARGIN_M_COMBINED`/`MARGIN_N_COMBINED` -
    Step 2/3's `MARGIN_M`/`MARGIN_N` do NOT transfer (different distance,
    different scale). Also only supports `ALPHA = 0.0` for now - `ALPHA`
    belongs to the retired blended-distance path (see below), not to this
    one; this loss type is its own, separate combined-distance mechanism.

`ALPHA` controls Method A vs. Method B vs. the blend, end to end (only
meaningful when `LOSS_TYPE = "dual_triplet"`):
  - `ALPHA = 0.0` (default): the original, already-validated Method-A-only
    path, byte-for-byte - `DualTripletLoss` (embeddings, unnormalized) +
    `run_one_epoch` + `make_method_a_distance_fn()` for checkpoint
    selection. No scale estimation happens; nothing about this path
    changes from before `ALPHA` existed.
  - `ALPHA > 0.0` (including `1.0`, pure Method B): the full blended
    machinery - `DistanceDualTripletLoss` (takes precomputed distances,
    see that module's docstring for why `DualTripletLoss` can't be reused
    here) + `run_one_epoch_blended` (wrapped via `functools.partial` to
    bind `alpha`/`scales`/the Sinkhorn settings so it matches
    `run_one_epoch`'s call signature - see `train_validation.py`'s
    `run_epoch_fn` extension point) + `make_blended_distance_fn(ALPHA, ...)`
    for checkpoint selection. Distance scales
    (`blended_distance.estimate_distance_scales`) are estimated once from
    training writers before training starts. Training and evaluation
    always use the SAME `ALPHA` and the SAME scales, so checkpoint
    selection measures exactly the objective being optimized - even at
    `ALPHA=1.0` this goes through the normalized blended path (not
    `make_method_b_distance_fn()`'s unnormalized version), since the
    margin hyperparameters below were chosen assuming scale-normalized
    distances.

Checkpoint selection uses the real Step 5 K-reference protocol
(`evaluation/protocol.py`'s `build_verification_callback`), not the
validation-triplet-loss placeholder `train_and_validate_model` falls back
to when no `verification_callback` is given - see that function's
docstring for why the placeholder is wrong to use permanently.

Model/optimizer defaults below (projector_hidden_dim=256, embedding_dim=256,
weight_decay=1e-4) match the SURDS-era run that produced the published v4
baseline (87.97% BHSig260 Bengali) and carried over unchanged into Step 1
- kept identical so that run was a like-for-like comparison, changing only
the pretraining (DenseCL vs. reconstruction).

`LOSS_TYPE` now defaults to `"double_margin_combined"` (Step 4, Cell A:
frozen encoder + the new local branch) - `RUN_TAG="step4_frozen_combined"`.
To reproduce earlier steps: `"dual_triplet"` + `INTRA_MARGIN`/
`INTER_MARGIN`=0.2 + `TRAINABLE_ENCODER_STAGES=()` is Step 1;
`"double_margin"` + `TRAINABLE_ENCODER_STAGES=()` is Step 2;
`"double_margin"` + `TRAINABLE_ENCODER_STAGES=("stage4",)` is Step 3a.

`TRAINABLE_ENCODER_STAGES` defaults to `()` here (Step 4 Cell A, matching
Step 1-2's frozen setting) - switch to `("stage4",)` + `RUN_TAG=
"step4_finetuned_combined"` for Cell B, the fine-tuned/headline run (Step
3a's setting). Whenever any encoder stage is trainable,
`_get_optimizer_param_groups` gives encoder parameters their own,
separate learning rate (`ENCODER_LEARNING_RATE`, ~10x lower than the
head's `LEARNING_RATE`) - these weights already encode 50 epochs of
DenseCL pretraining, and a careless gradient step on a 115-writer
supervised set can undo that faster than it can improve it.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = TRAINING_DIR.parent  # DenseCL_approach/supervised_verification_approach

sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(SUPERVISED_DIR / "utils"))
sys.path.insert(0, str(SUPERVISED_DIR / "datasets"))
sys.path.insert(0, str(SUPERVISED_DIR / "models"))
sys.path.insert(0, str(SUPERVISED_DIR / "loss"))
sys.path.insert(0, str(SUPERVISED_DIR / "matching"))
sys.path.insert(0, str(SUPERVISED_DIR / "evaluation"))

from writer_splits import DATA_ROOT, get_writer_split  # noqa: E402
from dual_triplet_dataset import DualTripletDataset  # noqa: E402
from fixed_dual_triplet_dataset import FixedDualTripletDataset, build_fixed_triplet_records  # noqa: E402
from pair_dataset import PairDataset  # noqa: E402
from embedding_model import load_downstream_model, load_random_init_downstream_model  # noqa: E402
from dual_triplet_loss import DualTripletLoss  # noqa: E402
from double_margin_loss import DoubleMarginLoss  # noqa: E402
from double_margin_distance_loss import DoubleMarginDistanceLoss  # noqa: E402
from distance_triplet_loss import DistanceDualTripletLoss  # noqa: E402
from blended_distance import estimate_distance_scales  # noqa: E402
from train_validation import train_and_validate_model  # noqa: E402
from run_one_epoch_blended import run_one_epoch_blended  # noqa: E402
from run_one_epoch_pairs import run_one_epoch_pairs  # noqa: E402
from run_one_epoch_combined import run_one_epoch_combined  # noqa: E402
from protocol import (  # noqa: E402
    build_verification_callback,
    make_blended_distance_fn,
    make_combined_batch_distance_fn,
    make_combined_distance_fn,
    make_method_a_distance_fn,
)

# -- Dataset / SSL checkpoint --------------------------------------------
# Which K-fold CV fold's writer split AND SSL encoder to use - both the
# downstream writer split (`get_writer_split(..., fold=FOLD)`) and the SSL
# checkpoint (`RUN_NAME` below) must come from the SAME fold, or the
# encoder would have been pretrained on writers this run then treats as
# held-out test/validation (or vice versa) - see `create_cv_fold_split.py`
# and downstream_supervised_learning_approach.md's K-fold CV section.
FOLD = "fold_0"

DATASET_NAME = "BHSig260_Bengali"
# ICCIT §4.3/§9.1/§9.2 Tier 1, run (B) - the RANDOM-INIT CONTROL, second
# dataset (Hindi's run is DONE - AUC 0.9548 vs. 0.9911 pretrained,
# writer-level-bootstrap-confirmed real gap, results.md §10 - this repeats
# it on Bengali to check the finding isn't a one-dataset fluke). Question:
# did the DenseCL pretraining do anything at all? Keeps the REAL reported
# architecture unchanged - combined distance (global + local branch), Cell
# C (full unfreeze) - identical to
# results/Bengali_data_ssl/fold_0/BHSig260_Bengali/step4_full_unfrozen_combined_indomain
# (AUC 0.9902, EER 5.18%) - the ONLY thing this run changes is
# RANDOM_INIT below: no SSL checkpoint is loaded at all, the encoder
# starts at random initialization (seeded, RANDOM_INIT_SEED) instead.
# This is a DIFFERENT axis from §9.2's global-only-head ablation (Tier 1
# run A, complete on all 3 datasets, results.md §9) - that one asks "does
# the local branch help," holding pretraining fixed; this one asks "does
# the pretraining help," holding the architecture (local branch included)
# fixed.
RUN_NAME = "random_init_control/fold_0"  # pseudo run-name, used only to name RESULTS_DIR below (matches every other run's directory shape) - there is no real SSL run here, nothing is ever loaded from this path
CHECKPOINT_EPOCH: int | None = None  # unused when RANDOM_INIT=True below
RANDOM_INIT = True  # <-- THE control switch: no SSL checkpoint loaded, encoder starts random (see model-construction call below)
RANDOM_INIT_SEED = 42  # seeds the encoder's OWN random weights (not just data sampling) - MUST match the seed used for MARGIN_M_COMBINED/MARGIN_N_COMBINED's sweep below, or the margins were measured against different random weights than the ones actually trained

# Names this run's own results folder, so each rung of the experiment
# ladder in downstream_supervised_learning_approach.md / the ICCIT doc
# gets its own directory instead of overwriting the previous one. Change
# this for every new configuration.
# THIS RUN = Cell C (full unfreeze), combined head (real architecture),
# random-init encoder, Bengali (repeating ICCIT §4.3's control - already
# DONE on Hindi - on a second dataset). "_randominit" distinguishes this
# from every "_indomain"/"_crossdomain" run, none of which apply here
# (there is no SSL source at all).
RUN_TAG = "step4_full_unfrozen_combined_randominit"

TRAIN_SEED = 42
VAL_TUPLES_PER_ANCHOR = 4
VAL_SEED = 314

# -- Verification (checkpoint-selection metric, Step 5) ----------------------
VERIFICATION_NUM_REFERENCES = 8  # K=8 matches SURDS's own protocol (their Sec. III-D), which is what final reporting uses
VERIFICATION_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)  # 5 draws/epoch - cheap at alpha=0.0 (Method A only)
VERIFICATION_BATCH_SIZE = 64  # pairs per batched Sinkhorn call in the verification protocol (LOSS_TYPE="double_margin_combined" only) - unrelated to training's BATCH_SIZE, no gradients/activations to keep here so this can be much larger

# -- Distance blend (Step 4) --------------------------------------------------
ALPHA = 0.0  # 0.0 = Method A only (original path); 1.0 = pure Method B; between = blended
SINKHORN_EPSILON = 0.05
SINKHORN_ITERATIONS = 50
SCALE_ESTIMATION_NUM_QUADRUPLES = 200  # only used when ALPHA > 0.0
SCALE_ESTIMATION_SEED = 42

# -- Model -----------------------------------------------------------------
# THIS RUN: Cell C - full unfreeze, ALL five encoder stages trainable, the
# same capacity as the combined-head headline result this run is being
# compared against (ICCIT §9.2 Tier 1, run A). No intermediate cells (A/B)
# needed for this ablation - the theme only needs to be shown to hold at
# the BEST configuration each dataset actually uses.
TRAINABLE_ENCODER_STAGES: tuple[str, ...] = ("stem", "stage1", "stage2", "stage3", "stage4")
PROJECTOR_HIDDEN_DIM = 256
EMBEDDING_DIM = 256
NORM_TYPE = "batch"

# Step 4's local/structural branch (`model.local_projection`, a single
# linear layer - see embedding_model.py). THIS RUN keeps it (128, same as
# every combined-head run) - the random-init control changes the
# encoder's starting weights only, not the architecture.
LOCAL_EMBEDDING_DIM: int | None = 128
LAMBDA_0 = 1.0  # dis = LAMBDA_0 * dis_global + dis_struct (DetailSemNet Eq. 2, lambda_0=1.0 optimal per their Table A4)

# -- Loss --------------------------------------------------------------------
# THIS RUN: "double_margin_combined" - the REAL architecture (local branch
# included), kept identical to the reported system. RANDOM_INIT above is
# the only thing that makes this a control rather than a normal Cell C run.
LOSS_TYPE = "double_margin_combined"  # "dual_triplet" (Step 1) / "double_margin" (Step 2-3) / "double_margin_combined" (Step 4, and this control)

# Only used when LOSS_TYPE == "dual_triplet".
INTRA_MARGIN = 0.2
INTER_MARGIN = 0.2
INTER_LOSS_WEIGHT = 1.0

# Only used when LOSS_TYPE == "double_margin". Not used by this run
# (LOSS_TYPE="double_margin_combined" above) - kept for when LOSS_TYPE is
# switched back to a global-only ablation run.
MARGIN_M = 0.16
MARGIN_N = 0.36
# ORIGINAL (Step 1-checkpoint proxy, pooled encoder - do not reuse either):
# MARGIN_M = 0.46
# MARGIN_N = 0.96

# Only used when LOSS_TYPE == "double_margin_combined".
#
# PRIOR RUNS' HISTORICAL RECORD (all pretrained-encoder sweeps - none
# apply to THIS run, kept only for reference): in-domain Hindi 0.34/0.71,
# in-domain Bengali 0.27/0.53, in-domain CEDAR 0.28/0.53, cross-domain
# Hindi->Bengali 0.33/0.61, cross-domain Hindi->CEDAR 0.32/0.57 - every
# one measured against a REAL DenseCL-pretrained encoder, which is exactly
# what THIS run does not have. Full per-sweep detail in git history / the
# chat record if ever needed again.
#
# RANDOM-INIT CONTROL MARGINS - BHSig260_Bengali (2026-09-12, repeating
# Hindi's already-complete random-init control on a second dataset - ICCIT
# §4.3/§9.1, Tier 1 run B). Same two-stage process Hindi needed, NOT a
# single sweep, because a random encoder's distance scale is not stable at
# t=0 the way a pretrained one's is (see Hindi's record above this run
# replaced, or the chat record, for the full diagnosis - same mechanism:
# `dis_global` is L2-normalized/bounded, `dis_struct`'s raw dense features
# are not, and a randomly initialized, fully-unfrozen 5-stage encoder's
# raw feature scale is far less stable early on than a pretrained one's).
#
# STAGE 1 (done on Kaggle, 1-epoch probe only): swept directly against the
# untrained random encoder - `sweep_margins_combined.py --dataset
# BHSig260_Bengali --random_init --seed 42 --device cpu` gave genuine-pair
# median 0.0061, negative-pair median 0.0068 (50.0%/49.9% active,
# n=960/1920 pairs). These were the margins the Kaggle notebook's 1-epoch
# probe run trained with (NUM_EPOCHS capped at 1, purely to get a real
# checkpoint out, not trusted for training quality) - NOT used below;
# expected to be stale by epoch 1 exactly like Hindi's were.
#
# STAGE 2 (DONE 2026-09-12): re-swept against the REAL post-gradient-flow
# checkpoint the probe produced - `checkpoints/epoch1.pt`, downloaded
# locally, then `sweep_margins_combined.py --dataset BHSig260_Bengali
# --full_checkpoint <path to epoch1.pt> --device cpu` loaded it in full
# (encoder + projector + local_projection together, not just the encoder)
# and measured where the distance distribution had actually settled after
# one real epoch: genuine-pair median 0.0219, negative-pair median 0.0229
# (50.0%/50.0% active, n=960/1920 pairs) - about 3.6x/3.4x Stage 1's t=0
# values, an even bigger drift than Hindi's ~2.2x but the same direction
# and cause. These are the margins THIS run actually trains with.
# `epoch1.pt` itself is now discarded - the real run below starts from a
# FRESH random init (same seed=42), not a warm start from it, or the "no
# pretraining at all" control would be contaminated by one epoch of head
# start.
MARGIN_M_COMBINED = 0.0219  # STAGE 2 (real) value - genuine-pair P50 against Bengali's epoch1.pt, see above
MARGIN_N_COMBINED = 0.0229  # STAGE 2 (real) value - negative-pair P50 against Bengali's epoch1.pt, see above

# -- Optimizer -----------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # head (projector) learning rate - unchanged from Steps 1-2
# Only used when TRAINABLE_ENCODER_STAGES is non-empty. ~10x lower than
# LEARNING_RATE: these weights already encode 50 epochs of DenseCL
# pretraining, and a large step on a small supervised set (115 writers for
# Hindi, 60 for Bengali, only 35 for CEDAR) risks undoing that faster than
# it improves verification. Kept unchanged across all three datasets for
# comparability, even though CEDAR's training pool - the smallest yet -
# would arguably justify going lower still.
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50  # generous ceiling; PATIENCE is what actually ends the run
PERIODIC_SAVE_FREQUENCY = 1  # save every epoch's checkpoint, not just the best one

# Early stopping. The previous run was cut off at epoch 5 with epoch 5 still
# the best score - i.e. stopped mid-climb, so where this design actually
# plateaus is unknown. PATIENCE ends the run once the verification score has
# not improved for that many consecutive epochs. MIN_EPOCHS blocks stopping
# before then regardless, because the score is noisy early (the 5-epoch run
# went 0.930 / 0.926 / 0.908 / 0.927 / 0.956 - that epoch-3 dip is noise, and
# without a floor a run of unlucky bounces could end training prematurely).
PATIENCE = 10
MIN_EPOCHS = 15

# Keyed on RUN_NAME (not a hardcoded "all_data_ssl" segment) so results
# land under whichever SSL run actually produced the encoder - for every
# pooled-encoder run RUN_NAME == "all_data_ssl/<fold>" already, so this is
# byte-identical to the old hardcoded path for all of them; for a
# single-dataset SSL run (e.g. "Hindi_data_ssl/fold_0", SS16) it correctly
# lands under that run's own name instead of the misleading "all_data_ssl"
# label.
RESULTS_DIR = SUPERVISED_DIR / "results" / RUN_NAME / DATASET_NAME / RUN_TAG
HISTORY_CSV_PATH = RESULTS_DIR / "training_history.csv"
MODEL_DIR = RESULTS_DIR / "checkpoints"


def _get_optimizer_param_groups(
    model: nn.Module, head_lr: float, encoder_lr: float, weight_decay: float,
) -> list[dict]:
    """Splits trainable parameters into up to 4 groups: {encoder, head} x
    {weight-decayed, not}. `DownstreamVerificationModel`'s submodules are
    named `encoder`/`projector`, so `param_name.startswith("encoder.")`
    cleanly separates the two - the encoder gets `encoder_lr` (see
    `ENCODER_LEARNING_RATE`'s definition for why it must be much lower
    than the head's), the projector gets `head_lr`. When
    `TRAINABLE_ENCODER_STAGES=()` (Steps 1-2), no encoder params have
    `requires_grad=True`, so the encoder groups are simply empty and
    dropped - identical behavior to the single-`head_lr` optimizer those
    steps used."""
    section_lr = {"encoder": encoder_lr, "head": head_lr}
    grouped_params: dict[tuple[str, bool], list] = {
        (section, is_decay): [] for section in section_lr for is_decay in (True, False)
    }
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        section = "encoder" if param_name.startswith("encoder.") else "head"
        is_decay = not (param.ndim == 1 or param_name.endswith(".bias"))
        grouped_params[(section, is_decay)].append(param)

    return [
        {"params": params, "lr": section_lr[section], "weight_decay": weight_decay if is_decay else 0.0}
        for (section, is_decay), params in grouped_params.items()
        if params
    ]


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = get_writer_split(DATASET_NAME, fold=FOLD)
    print(
        f"Writers - train: {len(split.train_writer_ids)} | "
        f"val: {len(split.validation_writer_ids)} | "
        f"test: {len(split.test_writer_ids)}"
    )

    dataset_dir = DATA_ROOT / DATASET_NAME

    train_dataset = DualTripletDataset(
        split.train_writer_ids, dataset_dir, DATASET_NAME, seed=TRAIN_SEED,
    )
    val_dataset = FixedDualTripletDataset(
        build_fixed_triplet_records(
            split.validation_writer_ids, dataset_dir, DATASET_NAME,
            tuples_per_anchor=VAL_TUPLES_PER_ANCHOR, seed=VAL_SEED,
        )
    )
    if LOSS_TYPE in ("double_margin", "double_margin_combined"):
        # Decomposes each quadruple into its 3 labeled pairs - same
        # underlying sampling/resampling discipline, different shape out.
        train_dataset = PairDataset(train_dataset)
        val_dataset = PairDataset(val_dataset)
    print(f"Dataset sizes - train: {len(train_dataset)} | val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Batches per epoch - train: {len(train_loader)} | val: {len(val_loader)}")

    if RANDOM_INIT:
        # THE control: no SSL checkpoint loaded anywhere. Seed explicitly
        # so the encoder's random weights are reproducible - this MUST
        # match the seed MARGIN_M_COMBINED/MARGIN_N_COMBINED were swept
        # with (RANDOM_INIT_SEED's own comment above), or the margins were
        # measured against different random weights than the ones
        # actually trained here.
        torch.manual_seed(RANDOM_INIT_SEED)
        model = load_random_init_downstream_model(
            trainable_encoder_stages=TRAINABLE_ENCODER_STAGES,
            projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            norm_type=NORM_TYPE,
            local_embedding_dim=LOCAL_EMBEDDING_DIM if LOSS_TYPE == "double_margin_combined" else None,
            device=device,
        )
        print(f"Random-init control: encoder seeded with RANDOM_INIT_SEED={RANDOM_INIT_SEED}, no SSL checkpoint loaded.")
    else:
        model = load_downstream_model(
            RUN_NAME, CHECKPOINT_EPOCH, device,
            trainable_encoder_stages=TRAINABLE_ENCODER_STAGES,
            projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            norm_type=NORM_TYPE,
            local_embedding_dim=LOCAL_EMBEDDING_DIM if LOSS_TYPE == "double_margin_combined" else None,
        )

    optimizer = torch.optim.AdamW(
        _get_optimizer_param_groups(
            model, head_lr=LEARNING_RATE, encoder_lr=ENCODER_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        ),
    )

    print(f"Run tag: {RUN_TAG}")
    print(f"Results:  {RESULTS_DIR}")
    print(
        f"Batch size: {BATCH_SIZE} | Head LR: {LEARNING_RATE}"
        + (f" | Encoder LR: {ENCODER_LEARNING_RATE}" if TRAINABLE_ENCODER_STAGES else " (encoder frozen)")
    )
    print(
        f"Epochs: {NUM_EPOCHS} (patience {PATIENCE}, min {MIN_EPOCHS}) | "
        f"Loss: {LOSS_TYPE} | Alpha: {ALPHA}"
    )

    eval_batch_distance_fn = None  # only "double_margin_combined" has a batched verification path so far
    if LOSS_TYPE == "double_margin_combined":
        if LOCAL_EMBEDDING_DIM is None:
            raise ValueError("LOSS_TYPE='double_margin_combined' requires LOCAL_EMBEDDING_DIM to be set.")
        print(
            f"Margins (combined): m={MARGIN_M_COMBINED}, n={MARGIN_N_COMBINED} "
            f"(from sweep_margins_combined.py) | lambda_0={LAMBDA_0} | local_dim={LOCAL_EMBEDDING_DIM}"
        )
        loss_function = DoubleMarginDistanceLoss(margin_m=MARGIN_M_COMBINED, margin_n=MARGIN_N_COMBINED)
        run_epoch_fn = functools.partial(
            run_one_epoch_combined, lambda_0=LAMBDA_0,
            sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        eval_distance_fn = make_combined_distance_fn(
            model, lambda_0=LAMBDA_0, sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        # Batched verification path (see downstream_supervised_learning_approach.md,
        # "Verification protocol speed"): the per-pair `eval_distance_fn` above is
        # kept only as the DistanceFn `build_verification_callback` still requires
        # positionally - `eval_batch_distance_fn` is what actually scores every
        # (query, reference) pair, VERIFICATION_BATCH_SIZE at a time, once it's
        # passed to `build_verification_callback` below.
        eval_batch_distance_fn = make_combined_batch_distance_fn(
            model, lambda_0=LAMBDA_0, sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
    elif LOSS_TYPE == "double_margin":
        if ALPHA > 0.0:
            raise NotImplementedError(
                "double_margin + ALPHA>0 (the local/blended distance) is the retired path - "
                "use LOSS_TYPE='double_margin_combined' for Step 4's local branch, or "
                "LOSS_TYPE='dual_triplet' for ALPHA>0 runs."
            )
        print(f"Margins: m={MARGIN_M}, n={MARGIN_N} (from sweep_margins.py)")
        loss_function = DoubleMarginLoss(margin_m=MARGIN_M, margin_n=MARGIN_N)
        run_epoch_fn = run_one_epoch_pairs
        eval_distance_fn = make_method_a_distance_fn()
    elif ALPHA <= 0.0:
        # Original Method-A-only path, unchanged - no scale estimation, no
        # blended machinery, byte-for-byte the pipeline that already
        # produced verified results.
        loss_function = DualTripletLoss(
            intra_margin=INTRA_MARGIN, inter_margin=INTER_MARGIN, inter_loss_weight=INTER_LOSS_WEIGHT,
        )
        run_epoch_fn = None  # train_and_validate_model defaults this to run_one_epoch
        eval_distance_fn = make_method_a_distance_fn()
    else:
        print(f"Estimating distance scales ({SCALE_ESTIMATION_NUM_QUADRUPLES} quadruples, training writers only)...")
        scales = estimate_distance_scales(
            model, train_dataset, device,
            num_quadruples=SCALE_ESTIMATION_NUM_QUADRUPLES,
            sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
            seed=SCALE_ESTIMATION_SEED,
        )
        print(f"Distance scales: method_a_scale={scales.method_a_scale:.4f}, method_b_scale={scales.method_b_scale:.4f}")

        loss_function = DistanceDualTripletLoss(
            intra_margin=INTRA_MARGIN, inter_margin=INTER_MARGIN, inter_loss_weight=INTER_LOSS_WEIGHT,
        )
        run_epoch_fn = functools.partial(
            run_one_epoch_blended, alpha=ALPHA, scales=scales,
            sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        eval_distance_fn = make_blended_distance_fn(
            ALPHA, scales.method_a_scale, scales.method_b_scale,
            sinkhorn_epsilon=SINKHORN_EPSILON, sinkhorn_iterations=SINKHORN_ITERATIONS,
        )

    verification_callback = build_verification_callback(
        dataset_dir, DATASET_NAME, split.validation_writer_ids,
        eval_distance_fn,
        num_references=VERIFICATION_NUM_REFERENCES,
        seeds=VERIFICATION_SEEDS,
        batch_distance_fn=eval_batch_distance_fn,
        batch_size=VERIFICATION_BATCH_SIZE,
    )
    print(
        f"Checkpoint selection: K-reference validation AUC "
        f"(K={VERIFICATION_NUM_REFERENCES}, {len(VERIFICATION_SEEDS)} draws/epoch, alpha={ALPHA})"
        + (f" [batched, batch_size={VERIFICATION_BATCH_SIZE}]" if eval_batch_distance_fn is not None else "")
    )

    history_df = train_and_validate_model(
        model=model,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
        history_csv_path=HISTORY_CSV_PATH,
        model_dir=MODEL_DIR,
        periodic_save_frequency=PERIODIC_SAVE_FREQUENCY,
        verification_callback=verification_callback,
        run_epoch_fn=run_epoch_fn,
        patience=PATIENCE,
        min_epochs=MIN_EPOCHS,
    )

    print("Training complete.")
    print(history_df)


if __name__ == "__main__":
    train()
