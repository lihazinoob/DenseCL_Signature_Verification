"""Dense correspondence visualization - DenseCL Figure 4/5 style, adapted for signatures.

For a sampled signature, generate two independently augmented views (the
same `generate_view` used to build training pairs), run each through the
encoder's dense (pre-projection-head) output, and find mutual
nearest-neighbor matches between the two views' foreground grid cells
(`correspondence.compute_dense_correspondence`). The result is drawn as a
gallery of side-by-side views with lines connecting matched patches,
color-coded by similarity (`visualize.plot_correspondence_gallery`).

Two encoders are compared per dataset, matching the paper's own Figure 5
framing (random-init vs. trained, rather than Figure 4's "compare two
different fully-converged methods" framing - more honest for an
early-training checkpoint):
  - the actual trained checkpoint (`results/training/<run_name>/checkpoints/
    encoder_epoch<N>.pt`)
  - a freshly-initialized, untrained `Encoder()` - the baseline "is this
    better than doing nothing" comparison.

Samples are drawn ONLY from each dataset's held-out VALIDATION writers
(`validation_set_creation.py`'s split - writers the SSL training loop never
trains on), not an arbitrary writer sample - the same principle
`evaluate_validation_loss` already follows in `driver/train.py`: judging a
trained model must not use data it was trained on, or the check proves
nothing about generalization.

Output: `DenseCL_approach/analysis/DenseCL_analysis/<run_name>/
dense_correspondence/<dataset>_{trained_epoch<N>,random_init}.png`.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ANALYZER_DIR = Path(__file__).resolve().parent
SELF_SUPERVISED_DIR = ANALYZER_DIR.parent
DENSECL_APPROACH_DIR = SELF_SUPERVISED_DIR.parent

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "model"))
sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))
sys.path.insert(0, str(ANALYZER_DIR))

from augment import DEFAULT_CONFIG, generate_view  # noqa: E402
from correspondence import compute_dense_correspondence  # noqa: E402
from encoder import Encoder  # noqa: E402
from mask import compute_foreground_mask  # noqa: E402
from orchestration import list_images, list_writer_dirs  # noqa: E402
from validation_set_creation import load_validation_writer_ids  # noqa: E402
from visualize import GalleryRow, plot_correspondence_gallery  # noqa: E402

DATA_ROOT = DENSECL_APPROACH_DIR / "data" / "all"
RESULTS_TRAINING_DIR = SELF_SUPERVISED_DIR / "results" / "training"
ANALYSIS_ROOT = DENSECL_APPROACH_DIR / "analysis" / "DenseCL_analysis"


@dataclass(frozen=True)
class AnalyzerConfig:
    run_name: str = "densecl_pretrain_v1"
    checkpoint_epoch: int | None = None  # None = use the latest checkpoint found for run_name
    num_samples_per_dataset: int = 5  # CEDAR has only 5 validation writers - the natural ceiling
    seed: int = 42
    grid_size: int = 32


def find_encoder_checkpoint(run_dir: Path, epoch: int | None) -> tuple[Path, int]:
    """Locate the encoder-only checkpoint to load. `epoch=None` picks the
    highest-numbered `encoder_epoch<N>.pt` found (the most-trained one)."""
    checkpoint_dir = run_dir / "checkpoints"
    if epoch is not None:
        path = checkpoint_dir / f"encoder_epoch{epoch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return path, epoch

    candidates = sorted(checkpoint_dir.glob("encoder_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No encoder checkpoints found under {checkpoint_dir}")

    def epoch_of(path: Path) -> int:
        return int(path.stem.replace("encoder_epoch", ""))

    best = max(candidates, key=epoch_of)
    return best, epoch_of(best)


def load_trained_encoder(checkpoint_path: Path, device: torch.device) -> Encoder:
    encoder = Encoder()
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder


def build_random_init_encoder(seed: int, device: torch.device) -> Encoder:
    """A freshly-initialized, never-trained encoder - the baseline
    correspondence quality should clearly beat (paper's own Figure 5
    framing: random init vs. trained)."""
    torch.manual_seed(seed)
    encoder = Encoder()
    encoder.to(device)
    encoder.eval()
    return encoder


def sample_validation_signatures(
    dataset_dir: Path, validation_writer_ids: set[str], num_samples: int, rng: random.Random
) -> list[Path]:
    """Pick one signature each from `num_samples` distinct, randomly chosen
    VALIDATION writers (never a training writer) - mirrors
    `orchestration.sample_signatures`, but restricted to the fixed
    validation-writer pool instead of the whole dataset."""
    writer_dirs = [p for p in list_writer_dirs(dataset_dir) if p.name in validation_writer_ids]
    if not writer_dirs:
        raise ValueError(f"{dataset_dir.name} has no validation writers on disk under {dataset_dir}")

    actual_num_samples = min(num_samples, len(writer_dirs))
    if actual_num_samples < num_samples:
        print(
            f"  [{dataset_dir.name}] only {len(writer_dirs)} validation writer(s) available, "
            f"requested {num_samples} - using {actual_num_samples}"
        )

    chosen_writers = rng.sample(writer_dirs, actual_num_samples)
    samples = []
    for writer_dir in chosen_writers:
        images = list_images(writer_dir)
        if not images:
            continue
        samples.append(rng.choice(images))
    return samples


@torch.no_grad()
def extract_dense_features(encoder: Encoder, view: np.ndarray, device: torch.device) -> np.ndarray:
    """Binary 256x256 view -> `(grid_size, grid_size, feature_dim)` dense
    backbone features (NOT the dense projection head's output - see this
    module's and `correspondence.py`'s docstrings for why)."""
    tensor = torch.from_numpy((view > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    dense = encoder(tensor, pool=False)  # (1, feature_dim, grid_size, grid_size)
    return dense.squeeze(0).permute(1, 2, 0).cpu().numpy()


def analyze_dataset(
    dataset_dir: Path,
    config: AnalyzerConfig,
    trained_encoder: Encoder,
    random_encoder: Encoder,
    checkpoint_epoch: int,
    aug_rng: np.random.Generator,
    sample_rng: random.Random,
    device: torch.device,
    output_dir: Path,
) -> tuple[Path, Path]:
    validation_writer_ids = load_validation_writer_ids(dataset_dir.name)
    samples = sample_validation_signatures(dataset_dir, validation_writer_ids, config.num_samples_per_dataset, sample_rng)

    trained_rows: list[GalleryRow] = []
    random_rows: list[GalleryRow] = []

    for image_path in samples:
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        view_a = generate_view(original, aug_rng, DEFAULT_CONFIG)
        view_b = generate_view(original, aug_rng, DEFAULT_CONFIG)
        mask_a, _ = compute_foreground_mask(view_a, config.grid_size)
        mask_b, _ = compute_foreground_mask(view_b, config.grid_size)
        label = f"writer {image_path.parent.name} / {image_path.name}"

        for encoder, rows in ((trained_encoder, trained_rows), (random_encoder, random_rows)):
            feat_a = extract_dense_features(encoder, view_a, device)
            feat_b = extract_dense_features(encoder, view_b, device)
            matches = compute_dense_correspondence(feat_a, feat_b, mask_a, mask_b)
            rows.append(GalleryRow(view_a=view_a, view_b=view_b, matches=matches, label=label))

    trained_path = plot_correspondence_gallery(
        trained_rows,
        output_dir / f"{dataset_dir.name}_trained_epoch{checkpoint_epoch}.png",
        suptitle=f"{dataset_dir.name}: dense correspondence - trained encoder (epoch {checkpoint_epoch}, validation writers)",
        grid_size=config.grid_size,
    )
    random_path = plot_correspondence_gallery(
        random_rows,
        output_dir / f"{dataset_dir.name}_random_init.png",
        suptitle=f"{dataset_dir.name}: dense correspondence - random-init encoder (baseline, validation writers)",
        grid_size=config.grid_size,
    )
    return trained_path, random_path


def main(config: AnalyzerConfig = AnalyzerConfig()) -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RESULTS_TRAINING_DIR / config.run_name
    checkpoint_path, checkpoint_epoch = find_encoder_checkpoint(run_dir, config.checkpoint_epoch)

    print(f"Run             : {config.run_name}")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_path} (epoch {checkpoint_epoch})")

    trained_encoder = load_trained_encoder(checkpoint_path, device)
    random_encoder = build_random_init_encoder(config.seed, device)

    output_dir = ANALYSIS_ROOT / config.run_name / "dense_correspondence"

    dataset_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders found under {DATA_ROOT}")

    # One shared, continuously-advancing rng for augmentation randomness and
    # one for writer/file sampling - both seeded once, not reset per
    # dataset, matching orchestration.py's convention for a single pass.
    aug_rng = np.random.default_rng(config.seed)
    sample_rng = random.Random(config.seed)

    for dataset_dir in dataset_dirs:
        trained_path, random_path = analyze_dataset(
            dataset_dir, config, trained_encoder, random_encoder, checkpoint_epoch,
            aug_rng, sample_rng, device, output_dir,
        )
        print(f"Saved {trained_path}")
        print(f"Saved {random_path}")


if __name__ == "__main__":
    main()
