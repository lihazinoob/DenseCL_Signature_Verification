"""DenseCL pretraining loop.

Wires together everything built so far - `utils/dataset.py`,
`model/encoder.py`, `model/heads.py`, `model/momentum.py`,
`model/memory_queue.py`, `model/losses.py` - into an actual training run.
See `docs/claude_response/progress_so_far.md` Section 2 for the full
architecture this implements, and the "what would you do now" chat
discussion for the reasoning behind the choices below.

One training step, in order (the order matters - see the module-level
comment in `train_one_step`):
  1. pull a batch (view_a, view_b, mask_a, mask_b)
  2. forward View A through the online encoder + online heads (grad on)
  3. forward View B through the momentum encoder + momentum heads (no grad)
  4. fetch the queue's CURRENT contents (before this step touches it)
  5. compute DenseCLLoss
  6. backward + optimizer step (updates only the online encoder/heads)
  7. EMA-update the momentum encoder/heads toward the just-updated online weights
  8. enqueue this batch's momentum global vectors for FUTURE steps' negatives

RESUME PROTOCOL (power-outage resilience, added for cross-machine training):
Every epoch now unconditionally saves BOTH an encoder-only checkpoint
(`encoder_epoch<N>.pt`, for downstream/analysis use) and a full resumable
state (`train_state_epoch<N>.pt`, everything needed to continue training
exactly where it left off: optimizer momentum buffers, momentum encoder/
heads, memory queue contents, LR scheduler's internal step counter, and the
`TrainConfig` that produced it). `train()` auto-detects an existing
checkpoint for `config.run_name` and resumes from it automatically - no
flag to remember. Calling `train()` again with the same `run_name` after an
interruption (or on an entirely different machine, given the same `data/`
contents copied over) just continues; calling it on a `run_name` with no
existing checkpoint starts fresh. To deliberately restart a `run_name` from
scratch, delete its `results/training/<run_name>/` folder first (or use a
new `run_name`) - mirrors this project's existing "never silently
overwrite" convention (`test_set_creation.py`/`validation_set_creation.py`).

Three things this protocol specifically guards against - see the resume
logic in `train()` and the helpers below for where each is handled:
  - A resumed run silently using a DIFFERENT `TrainConfig` than the one
    that produced the checkpoint would desync the learning-rate schedule
    (which is keyed on step COUNT, not epoch count - if `batch_size`
    changes, "epoch 23" no longer means the same step count) or quietly
    change the training objective mid-run. `_validate_resumed_config`
    checks every field (except `num_workers`, a pure performance knob)
    matches exactly, and raises rather than guessing.
  - A power cut can land exactly DURING a checkpoint or CSV write, leaving
    a half-written, corrupt file - the one moment this whole feature exists
    to survive. `_atomic_torch_save`/`_atomic_csv_save` write to a temp
    file and rename it into place only once the write is complete, so a
    save is always all-or-nothing.
  - A power cut can also land between the CSV write and the checkpoint
    save at the end of an epoch, leaving a CSV row for an epoch that has
    no matching saved weights. Resume always trusts the checkpoint's own
    recorded epoch number as ground truth and truncates any reloaded CSV
    history to it, so a stray row can never survive into a resumed run.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

DRIVER_DIR = Path(__file__).resolve().parent
SELF_SUPERVISED_DIR = DRIVER_DIR.parent
sys.path.insert(0, str(SELF_SUPERVISED_DIR / "model"))
sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))
sys.path.insert(0, str(DRIVER_DIR))

from dataset import SignatureSSLDataset, list_specific_writer_signature_paths  # noqa: E402
from encoder import Encoder  # noqa: E402
from heads import DenseHead, GlobalHead  # noqa: E402
from losses import DenseCLLoss, LossConfig  # noqa: E402
from lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from memory_queue import DEFAULT_QUEUE_SIZE, MemoryQueue  # noqa: E402
from momentum import DEFAULT_MOMENTUM, EMAModule, MomentumEncoder  # noqa: E402
from test_set_creation import OUTPUT_DIR as TEST_SPLIT_ROOT  # noqa: E402
from validation_set_creation import VALIDATION_WRITER_COUNTS, load_validation_writer_ids  # noqa: E402
from validation_set_creation import OUTPUT_DIR as VALIDATION_SPLIT_ROOT  # noqa: E402

DATA_ROOT = SELF_SUPERVISED_DIR.parent / "data" / "all"
RESULTS_DIR = SELF_SUPERVISED_DIR / "results" / "training"

# TrainConfig fields allowed to differ across a resume without raising -
# purely a performance knob (dataloader worker count), never touches the
# optimization trajectory or the saved tensors' shapes.
CONFIG_FIELDS_IGNORED_ON_RESUME = {"num_workers"}


@dataclass(frozen=True)
class TrainConfig:
    """Defaults are starting points, not final - see the "what would you do
    now" chat discussion for the reasoning behind each one. Meant to be
    edited directly here (matching this project's existing `AugmentConfig`/
    `LossConfig` convention) rather than exposed via a CLI, since a first
    correctness-focused run doesn't need one yet.

    Every field here (except `num_workers`) must stay IDENTICAL for the
    whole lifetime of one `run_name` - see this module's docstring's RESUME
    PROTOCOL section for why, and `_validate_resumed_config` for the check
    that enforces it."""

    run_name: str = "all_data_ssl/fold_1"

    # Which K-fold CV fold's writer split to pretrain under (see
    # `create_cv_fold_split.py` and `build_dataloaders` below) - `None`
    # (the original default) means "use the top-level, non-fold split
    # files" (`densecl_pretrain_v1`'s original behavior). A fold value
    # selects `data/test_set_writer_split/<fold>/` and
    # `data/validation_set_writer_split/<fold>/` instead. New field, so it
    # is simply absent from any pre-fold checkpoint's saved config and
    # never checked on those runs' resumes (`_validate_resumed_config`
    # only iterates the SAVED config's own keys).
    fold: str | None = "fold_1"

    # Which dataset(s) to pretrain on - `None` (the original, and still
    # default, behavior) pools every dataset under `data/all/` together,
    # matching every prior SSL run (`densecl_pretrain_v1`, `all_data_ssl/
    # fold_0`, `all_data_ssl/fold_1`). A tuple like `("BHSig260_Hindi",)`
    # restricts pretraining to just that dataset - the matched-domain
    # control (pretrain and finetune on the SAME dataset, no cross-dataset/
    # cross-script pooling) that isolates domain-transfer effects from the
    # pretext objective itself. Forwarded to `list_all_signature_paths`/
    # `SignatureSSLDataset` via `dataset_names` in `build_dataloaders`
    # below. Tuple (not set) so it stays hashable/comparable for
    # `_validate_resumed_config`'s equality check and `dataclasses.asdict`'s
    # CSV-friendly serialization.
    dataset_names: tuple[str, ...] | None = None

    # Real 50-epoch/5-warmup-epoch budget - confirmed via the saved
    # TrainConfig inside fold_0's own checkpoint (`densecl_pretrain_v1`,
    # copied into `all_data_ssl/fold_0`) to be the exact configuration
    # that produced it, so every fold trains under an identical schedule
    # and only the excluded writers differ.
    num_epochs: int = 50
    warmup_epochs: int = 5

    # Empirically measured on the actual target GPU (RTX 3050 Laptop, 4.29GB):
    # batch=16 peaked at ~6.15GB and spilled into slow paged memory (one step
    # took 16,452s instead of ~1s) rather than raising a clean CUDA OOM.
    # batch=4 -> 1.57GB peak, batch=8 -> 3.08GB peak, both clean and fast
    # (0.34s/0.67s per step) - 8 chosen over 4 despite similar throughput
    # because the dense loss's negative pool is drawn only from the current
    # batch (no queue), so a larger batch gives it more diverse negatives.
    batch_size: int = 8
    num_workers: int = 4  # matches fold_0's confirmed config; explicitly a no-effect-on-results knob (CONFIG_FIELDS_IGNORED_ON_RESUME)

    # SGD + linear LR scaling rule, both matching MoCo's own standard
    # recipe (base_lr=0.03 @ batch=256) and this project's existing thesis
    # training script's optimizer choice (SGD, momentum=0.9, wd=1e-4) -
    # NOT copying the old script's actual lr=0.01, since that was tuned for
    # a reconstruction loss, not a contrastive one.
    base_lr: float = 0.03
    base_lr_batch_size: int = 256
    sgd_momentum: float = 0.9
    weight_decay: float = 1e-4

    encoder_momentum: float = DEFAULT_MOMENTUM  # EMA momentum for the momentum encoder/heads
    queue_size: int = DEFAULT_QUEUE_SIZE
    loss_config: LossConfig = LossConfig()

    seed: int = 42

    @property
    def learning_rate(self) -> float:
        return self.base_lr * self.batch_size / self.base_lr_batch_size


@dataclass
class Models:
    online_encoder: Encoder
    momentum_encoder: MomentumEncoder
    global_head: GlobalHead
    dense_head: DenseHead
    momentum_global_head: EMAModule
    momentum_dense_head: EMAModule
    memory_queue: MemoryQueue
    loss_fn: DenseCLLoss

    def to(self, device: torch.device) -> "Models":
        for module in (
            self.online_encoder, self.momentum_encoder, self.global_head, self.dense_head,
            self.momentum_global_head, self.momentum_dense_head, self.memory_queue, self.loss_fn,
        ):
            module.to(device)
        return self

    def train(self) -> None:
        self.online_encoder.train()
        self.global_head.train()
        self.dense_head.train()
        # momentum modules stay in eval-equivalent behavior via their own
        # torch.no_grad() forward passes regardless of .train()/.eval(),
        # but BatchNorm still needs an explicit mode - keep them in train()
        # mode so their running stats keep updating from the momentum path's
        # own inputs too (matches standard MoCo practice).
        self.momentum_encoder.train()
        self.momentum_global_head.train()
        self.momentum_dense_head.train()

    def eval(self) -> None:
        self.online_encoder.eval()
        self.global_head.eval()
        self.dense_head.eval()
        self.momentum_encoder.eval()
        self.momentum_global_head.eval()
        self.momentum_dense_head.eval()

    def trainable_parameters(self):
        return list(self.online_encoder.parameters()) + list(self.global_head.parameters()) + list(self.dense_head.parameters())

    def load_resumable_state(self, checkpoint: dict) -> None:
        """Restore every stateful piece a resume needs, besides the
        optimizer/scheduler (loaded separately in `train()`, since they're
        not owned by `Models`)."""
        self.online_encoder.load_state_dict(checkpoint["online_encoder"])
        self.momentum_encoder.load_state_dict(checkpoint["momentum_encoder"])
        self.global_head.load_state_dict(checkpoint["global_head"])
        self.dense_head.load_state_dict(checkpoint["dense_head"])
        self.momentum_global_head.load_state_dict(checkpoint["momentum_global_head"])
        self.momentum_dense_head.load_state_dict(checkpoint["momentum_dense_head"])
        self.memory_queue.load_state_dict(checkpoint["memory_queue"])


def build_models(config: TrainConfig) -> Models:
    online_encoder = Encoder()
    momentum_encoder = MomentumEncoder(online_encoder)
    global_head = GlobalHead()
    dense_head = DenseHead()
    momentum_global_head = EMAModule(global_head)
    momentum_dense_head = EMAModule(dense_head)
    memory_queue = MemoryQueue(size=config.queue_size)
    loss_fn = DenseCLLoss(config=config.loss_config)
    return Models(
        online_encoder, momentum_encoder, global_head, dense_head,
        momentum_global_head, momentum_dense_head, memory_queue, loss_fn,
    )


def build_dataloaders(config: TrainConfig, starting_epoch: int = 1) -> tuple[DataLoader, DataLoader]:
    """Writer-level train/validation split (not the old image-level random
    holdout): validation writers come from `validation_set_creation.py`'s
    fixed, pre-created split (already guaranteed disjoint from the test
    split by construction); the training pool is every remaining writer
    after excluding BOTH test and validation writers. No images are kept
    around for a separate "sanity" mechanism - see `evaluate_validation_loss`.

    `starting_epoch` seeds the train loader's shuffle generator as
    `config.seed + starting_epoch` rather than always `config.seed` - a
    fresh run and a run resumed mid-way both get a shuffle order that
    depends only on which epoch they're starting at, never on how many
    times the process happened to restart. Without this, a resumed run's
    first epoch would silently replay whichever earlier epoch's shuffle
    order happened to share `config.seed`'s starting point, instead of
    getting its own distinct permutation.

    `config.fold`, if set, selects that fold's writer splits
    (`data/test_set_writer_split/<fold>/`, `data/validation_set_writer_split/<fold>/`)
    instead of the top-level, non-fold split files - see `create_cv_fold_split.py`
    for how a fold's splits are generated (K-fold CV, roadmap doc SS on
    generalization evidence).

    `config.dataset_names`, if set, restricts both the training and
    validation pools to only those dataset(s) instead of pooling every
    dataset under `data/all/` - the single-dataset SSL pretraining case."""
    test_split_dir = TEST_SPLIT_ROOT / config.fold if config.fold else TEST_SPLIT_ROOT
    validation_split_dir = VALIDATION_SPLIT_ROOT / config.fold if config.fold else VALIDATION_SPLIT_ROOT
    wanted_datasets = set(config.dataset_names) if config.dataset_names else None

    validation_writer_ids = {
        dataset_name: load_validation_writer_ids(dataset_name, split_dir=validation_split_dir)
        for dataset_name in VALIDATION_WRITER_COUNTS
        if wanted_datasets is None or dataset_name in wanted_datasets
    }

    train_dataset = SignatureSSLDataset(
        DATA_ROOT, extra_exclude_writer_ids=validation_writer_ids, test_split_dir=test_split_dir,
        dataset_names=wanted_datasets,
    )

    validation_paths = list_specific_writer_signature_paths(DATA_ROOT, validation_writer_ids)
    validation_dataset = SignatureSSLDataset(DATA_ROOT, image_paths_override=validation_paths)

    # Note: SignatureSSLDataset.__getitem__ draws a fresh, UNSEEDED
    # np.random.default_rng() per call, by design (see dataset.py's
    # docstring - re-augments every epoch, View A != View B). That's left
    # untouched here; only the DataLoader's shuffling order is made
    # reproducible via `loader_generator`.
    loader_generator = torch.Generator().manual_seed(config.seed + starting_epoch)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, generator=loader_generator, drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader


def train_one_step(
    batch: dict[str, torch.Tensor],
    models: Models,
    optimizer: SGD,
    scheduler: LinearWarmupCosineAnnealingLR,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    view_a = batch["view_a"].to(device, non_blocking=True)
    view_b = batch["view_b"].to(device, non_blocking=True)
    mask_a = batch["mask_a"].to(device, non_blocking=True)
    mask_b = batch["mask_b"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    # Step 2: online path (View A), gradients on.
    pooled_a = models.online_encoder(view_a, pool=True)
    dense_a = models.online_encoder(view_a, pool=False)
    z_a_global = models.global_head(pooled_a)
    z_a_dense = models.dense_head(dense_a)

    # Step 3: momentum path (View B), no gradients (enforced inside MomentumEncoder/EMAModule).
    pooled_b = models.momentum_encoder(view_b, pool=True)
    dense_b = models.momentum_encoder(view_b, pool=False)
    z_b_global = models.momentum_global_head(pooled_b)
    z_b_dense = models.momentum_dense_head(dense_b)

    # Step 4: read the queue's CURRENT contents, before this step's enqueue().
    queue_vectors = models.memory_queue.get()

    # Step 5: loss.
    losses = models.loss_fn(z_a_global, z_b_global, queue_vectors, z_a_dense, z_b_dense, mask_a, mask_b)

    # Step 6: backward + optimizer step - updates only the online encoder/heads.
    losses["total"].backward()
    optimizer.step()
    scheduler.step()

    # Step 7: EMA-update the momentum encoder/heads toward the just-updated online weights.
    models.momentum_encoder.update(models.online_encoder, momentum=config.encoder_momentum)
    models.momentum_global_head.update(models.global_head, momentum=config.encoder_momentum)
    models.momentum_dense_head.update(models.dense_head, momentum=config.encoder_momentum)

    # Step 8: only now enqueue this batch's momentum keys, for future steps' negatives.
    models.memory_queue.enqueue(z_b_global.detach())

    return {key: float(value.item()) for key, value in losses.items()}


@torch.no_grad()
def evaluate_validation_loss(validation_loader: DataLoader, models: Models, device: torch.device) -> dict[str, float]:
    """Forward-only loss on held-out WRITERS the training loop never trains
    on - the classical overfitting check the user asked for, tracked as
    numbers only (no images kept). This catches the SSL-specific failure
    mode discussed in chat: the encoder exploiting training-image-specific
    quirks rather than learning generalizable stroke structure, which a
    well-behaved-looking training loss alone wouldn't reveal.

    Not a substitute for the real downstream validation AUC
    (progress_so_far.md Section 5, step 7), which needs labels this stage
    never uses - this is unlabeled, same as the training loss itself. A
    live labeled probe was considered and deliberately skipped (see chat:
    DenseCL's own authors never monitored representation quality during
    pretraining either, only via full downstream fine-tuning after the
    fact - this loss-curve check is the field-appropriate substitute at
    pretraining time, not a downgrade from some standard practice).

    Never touches the queue's contents (no enqueue) - validation batches
    must not contaminate the negative pool training steps rely on."""
    models.eval()
    totals, globals_, denses = [], [], []

    progress_bar = tqdm(validation_loader, desc="  Validation", leave=False)
    for batch in progress_bar:
        view_a = batch["view_a"].to(device, non_blocking=True)
        view_b = batch["view_b"].to(device, non_blocking=True)
        mask_a = batch["mask_a"].to(device, non_blocking=True)
        mask_b = batch["mask_b"].to(device, non_blocking=True)

        z_a_global = models.global_head(models.online_encoder(view_a, pool=True))
        z_a_dense = models.dense_head(models.online_encoder(view_a, pool=False))
        z_b_global = models.momentum_global_head(models.momentum_encoder(view_b, pool=True))
        z_b_dense = models.momentum_dense_head(models.momentum_encoder(view_b, pool=False))

        losses = models.loss_fn(
            z_a_global, z_b_global, models.memory_queue.get(),
            z_a_dense, z_b_dense, mask_a, mask_b,
        )
        totals.append(losses["total"].item())
        globals_.append(losses["global"].item())
        denses.append(losses["dense"].item())
        progress_bar.set_postfix({
            "total": f"{sum(totals) / len(totals):.4f}",
            "global": f"{sum(globals_) / len(globals_):.4f}",
            "dense": f"{sum(denses) / len(denses):.4f}",
        })

    models.train()
    return {
        "validation_total": sum(totals) / len(totals),
        "validation_global": sum(globals_) / len(globals_),
        "validation_dense": sum(denses) / len(denses),
    }


def _config_to_dict(config: TrainConfig) -> dict:
    return dataclasses.asdict(config)


def _validate_resumed_config(saved_config: dict, current_config: TrainConfig) -> None:
    """Refuse to resume if any schedule/objective-affecting field differs
    from the config that produced the checkpoint - see this module's
    docstring's RESUME PROTOCOL section for why a silent mismatch here is
    dangerous (a desynced LR schedule, or an unnoticed mid-run change to
    the training objective) rather than merely inconvenient."""
    current = _config_to_dict(current_config)
    mismatches = [
        (key, saved_value, current.get(key))
        for key, saved_value in saved_config.items()
        if key not in CONFIG_FIELDS_IGNORED_ON_RESUME and current.get(key) != saved_value
    ]
    if mismatches:
        lines = "\n".join(f"  - {key}: checkpoint has {saved!r}, current config has {now!r}" for key, saved, now in mismatches)
        raise ValueError(
            f"Cannot resume run '{current_config.run_name}': the following TrainConfig field(s) "
            f"differ from the checkpoint that produced them, which would desync the learning-rate "
            f"schedule and/or silently change the training objective mid-run:\n{lines}\n"
            f"Match these fields exactly to resume, or start a new run under a different run_name."
        )


def _validate_resumed_schedule(saved_scheduler_state: dict, warmup_steps: int, total_steps: int) -> None:
    """A second, independent check beyond `_validate_resumed_config`: even
    with identical TrainConfig fields, `warmup_steps`/`total_steps` are
    ALSO a function of the training pool's size (`steps_per_epoch =
    pool_size // batch_size`). If the copied-over `data/` folder on the new
    machine doesn't exactly match the original (a missing file, a
    regenerated-instead-of-copied split JSON, ...), the pool size shifts
    and this recomputes a different schedule than the one actually saved -
    exactly the silent corruption this whole protocol exists to prevent."""
    saved_warmup_steps = saved_scheduler_state.get("warmup_epochs")  # LinearWarmupCosineAnnealingLR's own field name for "warmup steps"
    saved_total_steps = saved_scheduler_state.get("max_epochs")      # ... and for "total steps"
    if saved_warmup_steps != warmup_steps or saved_total_steps != total_steps:
        raise ValueError(
            f"Cannot resume: the recomputed schedule (warmup_steps={warmup_steps}, "
            f"total_steps={total_steps}) does not match the checkpoint's saved schedule "
            f"(warmup_steps={saved_warmup_steps}, total_steps={saved_total_steps}), even though "
            f"the TrainConfig fields matched. This means the training pool's SIZE changed between "
            f"sessions - almost always because the data/ folder (images and/or the test/validation "
            f"split JSON files) on this machine isn't byte-identical to the one the checkpoint was "
            f"produced with. Copy the exact same data/ contents over before resuming."
        )


def _atomic_torch_save(obj, path: Path) -> None:
    """Write-to-temp-then-rename: a power cut mid-write leaves either the
    old file intact or the new one fully written, never a corrupt partial
    file - the exact failure mode this feature is built to survive."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _atomic_csv_save(rows: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """The full resumable `train_state_epoch<N>.pt` with the highest N
    under `run_dir/checkpoints/`, or `None` if this run has never
    checkpointed (a brand new run_name)."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidates = list(checkpoint_dir.glob("train_state_epoch*.pt"))
    if not candidates:
        return None

    def epoch_of(path: Path) -> int:
        return int(path.stem.replace("train_state_epoch", ""))

    return max(candidates, key=epoch_of)


def save_checkpoint(models: Models, optimizer: SGD, scheduler: LinearWarmupCosineAnnealingLR,
                     epoch: int, global_step: int, config: TrainConfig, run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Full state, for resuming a run - includes the config that produced it
    # (see `_validate_resumed_config`) so a future resume can check it's
    # actually being continued under matching settings, not guessing.
    _atomic_torch_save({
        "epoch": epoch,
        "global_step": global_step,
        "config": _config_to_dict(config),
        "online_encoder": models.online_encoder.state_dict(),
        "momentum_encoder": models.momentum_encoder.state_dict(),
        "global_head": models.global_head.state_dict(),
        "dense_head": models.dense_head.state_dict(),
        "momentum_global_head": models.momentum_global_head.state_dict(),
        "momentum_dense_head": models.momentum_dense_head.state_dict(),
        "memory_queue": models.memory_queue.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_dir / f"train_state_epoch{epoch}.pt")

    # Encoder-only, the actual artifact Stage B (downstream) and the
    # analysis tools reuse - kept for EVERY epoch (not just a periodic
    # subset), since that's exactly what the analyzer/interactive tool's
    # "compare across epochs" workflow already relies on. Cheap: ~11MB per
    # epoch, so a full 50-epoch run costs well under 1GB for these alone.
    _atomic_torch_save(models.online_encoder.state_dict(), checkpoint_dir / f"encoder_epoch{epoch}.pt")

    print(f"  [Checkpoint] Saved train_state_epoch{epoch}.pt and encoder_epoch{epoch}.pt")


def train(config: TrainConfig = TrainConfig(), time_budget_seconds: float | None = None) -> None:
    """`time_budget_seconds` is a session-runtime knob, not a training-schedule
    field - deliberately NOT part of `TrainConfig` (so it's never compared by
    `_validate_resumed_config`; a fresh session naturally gets its own fresh
    budget regardless of what a previous session used). When set, checked only
    at epoch BOUNDARIES, after that epoch's checkpoint has already saved - this
    can only ever stop the run at an already-safely-persisted point, never
    mid-epoch, matching the granularity the whole resume protocol is built
    around. Built for time-limited hosted environments (e.g. Kaggle's session
    runtime cap) - pass something safely under the actual limit (leaving room
    for the last epoch's own duration plus checkpoint-save time), not the
    limit itself.
    """
    start_time = time.monotonic()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = RESULTS_DIR / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    step_csv = run_dir / "step_summary.csv"
    epoch_csv = run_dir / "epoch_summary.csv"

    models = build_models(config).to(device)
    models.train()

    optimizer = SGD(
        models.trainable_parameters(),
        lr=config.learning_rate,
        momentum=config.sgd_momentum,
        weight_decay=config.weight_decay,
    )

    # ---- Resume detection: auto-continue this run_name if it already has
    # a checkpoint, otherwise start fresh. No flag to remember - see this
    # module's docstring's RESUME PROTOCOL section. ----
    starting_epoch = 1
    global_step = 0
    step_logs: list[dict] = []
    epoch_logs: list[dict] = []
    checkpoint = None

    latest_checkpoint_path = find_latest_checkpoint(run_dir)
    if latest_checkpoint_path is not None:
        print(f"Found existing checkpoint for '{config.run_name}': {latest_checkpoint_path.name} - resuming.")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        saved_config = checkpoint.get("config")
        if saved_config is not None:
            _validate_resumed_config(saved_config, config)
        else:
            print("  [Warning] This checkpoint predates resume support (no saved config) - "
                  "skipping the config-consistency check. Proceed only if you're sure the "
                  "current TrainConfig matches what produced it.")

        models.load_resumable_state(checkpoint)
        # NOT optimizer.load_state_dict(...) here yet - see the note below,
        # by the scheduler construction, for why that has to happen AFTER
        # the scheduler exists.

        starting_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

        # Trust the checkpoint's own recorded epoch as ground truth - if a
        # crash happened between the CSV write and the checkpoint save for
        # some epoch, this drops that epoch's now-orphaned CSV row(s)
        # rather than double-counting them once it's redone.
        ckpt_epoch = checkpoint["epoch"]
        if epoch_csv.exists():
            epoch_logs = pd.read_csv(epoch_csv).query("epoch <= @ckpt_epoch").to_dict("records")
        if step_csv.exists():
            step_logs = pd.read_csv(step_csv).query("epoch <= @ckpt_epoch").to_dict("records")

        if starting_epoch > config.num_epochs:
            print(f"'{config.run_name}' already completed all {config.num_epochs} epochs - nothing to do.")
            return
    else:
        print(f"No existing checkpoint for '{config.run_name}' - starting fresh.")

    train_loader, validation_loader = build_dataloaders(config, starting_epoch)

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, config.num_epochs * steps_per_epoch)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    # `LRScheduler.__init__` ALWAYS calls `_initial_step()` (a PyTorch
    # internal, unconditional on `last_epoch`), which writes a fresh
    # `get_lr()` value straight into `optimizer.param_groups[i]['lr']` - for
    # this scheduler, that's `warmup_start_lr` (0.0), since `_initial_step`
    # runs with `last_epoch == 0`. So the scheduler construction above
    # ALWAYS clobbers whatever LR was in the optimizer to 0.0, checkpoint or
    # not. `optimizer.load_state_dict(...)` must run AFTER this line (never
    # before it) so its restored `lr` is the last thing written - otherwise
    # the resumed run silently trains at LR 0.0 forever after (this
    # scheduler's `get_lr()` is recursive - it multiplies the CURRENT
    # `group["lr"]` by a ratio each step, so once it's 0 it can never
    # recover on its own). `scheduler.load_state_dict(...)` does NOT have
    # this problem - it's just `self.__dict__.update(...)`, restoring the
    # scheduler's own attributes (`last_epoch` etc.), never touching the
    # optimizer.
    if checkpoint is not None:
        _validate_resumed_schedule(checkpoint["scheduler"], warmup_steps, total_steps)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Run             : {config.run_name}")
    print(f"CV fold         : {config.fold if config.fold else '(none - top-level split)'}")
    print(f"Dataset(s)      : {', '.join(config.dataset_names) if config.dataset_names else 'all four, pooled'}")
    print(f"Device          : {device}")
    print(f"Dataset         : {len(train_loader.dataset)} train / {len(validation_loader.dataset)} validation (writer-level, fixed split)")
    print(f"Epochs          : {config.num_epochs} (starting at epoch {starting_epoch})")
    print(f"Steps per epoch : {steps_per_epoch}")
    print(f"Warmup steps    : {warmup_steps}")
    print(f"Total steps     : {total_steps}")
    print(f"Batch size      : {config.batch_size}")
    print(f"Learning rate   : {config.learning_rate:.5f} (base {config.base_lr} scaled for batch {config.batch_size})")
    print(f"Queue size      : {config.queue_size}")
    print(f"Encoder momentum: {config.encoder_momentum}")
    print("=" * 60)

    for epoch in range(starting_epoch, config.num_epochs + 1):
        epoch_totals: list[float] = []
        epoch_globals: list[float] = []
        epoch_denses: list[float] = []

        progress_bar = tqdm(train_loader, desc=f"[{config.run_name}] Epoch {epoch}/{config.num_epochs}", leave=True)
        for batch_index, batch in enumerate(progress_bar):
            losses = train_one_step(batch, models, optimizer, scheduler, config, device)
            global_step += 1

            epoch_totals.append(losses["total"])
            epoch_globals.append(losses["global"])
            epoch_denses.append(losses["dense"])

            step_logs.append({
                "run_name": config.run_name,
                "epoch": epoch,
                "epoch_step": batch_index + 1,
                "global_step": global_step,
                "loss_total": losses["total"],
                "loss_global": losses["global"],
                "loss_dense": losses["dense"],
                "lr": optimizer.param_groups[0]["lr"],
            })
            progress_bar.set_postfix({
                "total": f"{losses['total']:.4f}",
                "global": f"{losses['global']:.4f}",
                "dense": f"{losses['dense']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
            })

        validation_metrics = evaluate_validation_loss(validation_loader, models, device)
        epoch_log = {
            "run_name": config.run_name,
            "epoch": epoch,
            "epoch_avg_loss_total": sum(epoch_totals) / len(epoch_totals),
            "epoch_avg_loss_global": sum(epoch_globals) / len(epoch_globals),
            "epoch_avg_loss_dense": sum(epoch_denses) / len(epoch_denses),
            **validation_metrics,
        }
        epoch_logs.append(epoch_log)
        print(f"  [Epoch {epoch}] train_total={epoch_log['epoch_avg_loss_total']:.4f}  "
              f"validation_total={validation_metrics['validation_total']:.4f}")

        _atomic_csv_save(step_logs, step_csv)
        _atomic_csv_save(epoch_logs, epoch_csv)

        # Every epoch, unconditionally - see this module's docstring's
        # RESUME PROTOCOL section for why (power outages can strike after
        # any epoch, not just ones divisible by some frequency).
        save_checkpoint(models, optimizer, scheduler, epoch, global_step, config, run_dir)

        if time_budget_seconds is not None and time.monotonic() - start_time >= time_budget_seconds:
            print(f"  [Time budget] {time_budget_seconds / 3600:.2f}h budget reached after epoch {epoch} - "
                  f"stopping here (epoch {epoch} is fully checkpointed). Re-run train() with the same "
                  f"run_name once its results/checkpoints are back in place to continue from epoch {epoch + 1}.")
            return


if __name__ == "__main__":
    # CEDAR-only, fold_0: the third and final dataset in the matched-domain
    # SSL pretraining control's ladder ([[in-domain-ssl-pretraining-study]]),
    # after Hindi-only/fold_0 and Bengali-only/fold_0 both completed cleanly.
    # Same reasoning as those runs - see this module's docstring and
    # downstream_supervised_learning_approach.md SS16 for why this control is
    # needed, not just TrainConfig()'s current default (fold_1, all four
    # datasets pooled - left untouched since fold_1's pretraining is a
    # separate, currently in-flight run on a different machine; changing the
    # dataclass's own defaults risks desyncing that run if this file is ever
    # synced there).
    # `fold="fold_0"` selects the SAME writer split already used by every
    # downstream fold_0 result so far (Hindi/Bengali/CEDAR Step 4): CEDAR's
    # fold_0 has 55 total writers, 15 test, 5 validation, 35 train - so this
    # run's training pool is exactly those 35 writers (byte-identical to the
    # pooled fold_0 encoder's CEDAR portion), only the OTHER THREE datasets
    # are removed from the pool. This is the SMALLEST of the three
    # single-dataset SSL pools by a wide margin (35 vs Bengali's 60 vs
    # Hindi's 115 training writers) - worth keeping firmly in mind when
    # judging this run's loss curves and any later downstream result: CEDAR
    # was already the weakest of the three datasets even with pooled SSL
    # pretraining (SS15.2, largely attributed to its smaller writer pool),
    # so an even smaller CEDAR-only SSL pool is the hardest test yet of
    # whether this pipeline's SSL stage needs a certain data volume to work
    # well at all.
    #
    # NOT smoke-tested on this machine (a separate SSL run - fold_1, pooled -
    # is already using its GPU); intended to be launched on a different
    # machine, same as the command handed to the user. Every input path here
    # (DATA_ROOT, fold-scoped split dirs, dataset_names filter) is identical
    # in kind to what Hindi-only/fold_0 and Bengali-only/fold_0 already
    # exercised successfully, only the dataset name and run_name differ.
    train(config=TrainConfig(
        run_name="CEDAR_data_ssl/fold_0",
        fold="fold_0",
        dataset_names=("CEDAR",),
    ))
