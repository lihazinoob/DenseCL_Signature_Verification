"""Generate sanity-check comparison plots for every dataset.

For each dataset under DenseCL_approach/data/all, this script randomly
picks 10 distinct writers (so the sample spans the writer population
instead of clustering on one writer or the first N files) and takes one
signature from each. Two galleries are produced per dataset, one file
each:

  1. results/preprocessing/<dataset>_preprocessing_comparison.png
     original signature next to its preprocessed version.
  2. results/masking/<dataset>_mask_comparison.png
     preprocessed signature next to the same image with its foreground
     (ink) grid mask overlaid, so the mask can be checked visually
     against the actual strokes before it's used in training.

  3. results/augmentation/<variant>/<dataset>_augmentation_comparison.png
     preprocessed signature next to one augmented view, one such gallery
     per augmentation variant (affine, elastic, dilate, erode, noise, and
     the full combined pipeline), so each augmentation's effect can be
     inspected in isolation before they're all used together to build
     View A / View B for the DenseCL pretext.

All galleries use the same random seed, so the same 10 writers/images
are shown throughout for easy side-by-side comparison.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augment import (
    DEFAULT_CONFIG,
    AugmentConfig,
    add_boundary_noise,
    cutout_strokes,
    dilate_strokes,
    elastic_warp,
    erode_strokes,
    generate_view,
    random_affine,
)
from mask import compute_foreground_mask
from preprocess import preprocess_signature

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent / "data" / "all"
PREPROCESSING_OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "preprocessing"
MASKING_OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "masking"
AUGMENTATION_OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "augmentation"


# Each variant is (raw_original, rng, config) -> final 256x256 binary
# image, fully self-contained (each does its own preprocess_signature
# call at the right point), so they can all be driven by the same loop.
def _variant_affine(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return preprocess_signature(random_affine(image, rng, config))


def _variant_elastic(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return preprocess_signature(elastic_warp(image, rng, config))


def _variant_dilate(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return preprocess_signature(dilate_strokes(image, config))


def _variant_erode(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return preprocess_signature(erode_strokes(image, config))


def _variant_noise(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    # Noise is a post-preprocess step (see add_boundary_noise's docstring
    # for why), so this variant skips augment_view entirely.
    return add_boundary_noise(preprocess_signature(image), rng, config)


def _variant_cutout(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return preprocess_signature(cutout_strokes(image, rng, config))


def _variant_combined(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    return generate_view(image, rng, config)


AUGMENTATION_VARIANTS: dict[str, Callable[[np.ndarray, np.random.Generator, AugmentConfig], np.ndarray]] = {
    "affine": _variant_affine,
    "elastic": _variant_elastic,
    "dilate": _variant_dilate,
    "erode": _variant_erode,
    "noise": _variant_noise,
    "cutout": _variant_cutout,
    "combined": _variant_combined,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUM_SAMPLES = 10
SEED = 42


def list_writer_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.iterdir() if p.is_dir())


def list_images(writer_dir: Path) -> list[Path]:
    return sorted(p for p in writer_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def sample_signatures(dataset_dir: Path, num_samples: int, rng: random.Random) -> list[Path]:
    """Pick one signature each from `num_samples` distinct, randomly chosen writers."""
    writer_dirs = list_writer_dirs(dataset_dir)
    if len(writer_dirs) < num_samples:
        raise ValueError(
            f"{dataset_dir.name} has only {len(writer_dirs)} writers; "
            f"need at least {num_samples} to sample one signature per writer."
        )

    chosen_writers = rng.sample(writer_dirs, num_samples)

    samples = []
    for writer_dir in chosen_writers:
        images = list_images(writer_dir)
        if not images:
            continue
        samples.append(rng.choice(images))
    return samples


def _new_comparison_figure(num_rows: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(num_rows, 2, figsize=(6, 3 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, 2)
    return fig, axes


def plot_preprocessing_comparison(
    dataset_dir: Path, output_dir: Path, num_samples: int, rng: random.Random
) -> Path:
    """Save one figure: original vs. preprocessed, for `num_samples` writers."""
    samples = sample_signatures(dataset_dir, num_samples, rng)
    fig, axes = _new_comparison_figure(len(samples))

    for row, image_path in enumerate(samples):
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        preprocessed = preprocess_signature(original)

        label = f"writer {image_path.parent.name} / {image_path.name}"

        axes[row, 0].imshow(original, cmap="gray")
        axes[row, 0].set_title(f"Original\n{label}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(preprocessed, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Preprocessed", fontsize=8)
        axes[row, 1].axis("off")

    fig.suptitle(f"{dataset_dir.name}: original vs. preprocessed ({len(samples)} writers)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_dir.name}_preprocessing_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _upsample_mask(foreground_mask: np.ndarray, image_size: int) -> np.ndarray:
    """Expand a (grid, grid) boolean mask to a pixel-aligned (image_size, image_size) mask."""
    cell = image_size // foreground_mask.shape[0]
    return np.repeat(np.repeat(foreground_mask, cell, axis=0), cell, axis=1)


def plot_masking_comparison(
    dataset_dir: Path, output_dir: Path, num_samples: int, rng: random.Random
) -> Path:
    """Save one figure: preprocessed vs. preprocessed-with-mask-overlay, for `num_samples` writers."""
    samples = sample_signatures(dataset_dir, num_samples, rng)
    fig, axes = _new_comparison_figure(len(samples))

    for row, image_path in enumerate(samples):
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        preprocessed = preprocess_signature(original)
        foreground_mask, _ = compute_foreground_mask(preprocessed)
        pixel_mask = _upsample_mask(foreground_mask, preprocessed.shape[0])
        overlay = np.ma.masked_where(~pixel_mask, np.ones_like(pixel_mask, dtype=float))

        label = f"writer {image_path.parent.name} / {image_path.name}"

        axes[row, 0].imshow(preprocessed, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Preprocessed\n{label}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(preprocessed, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(overlay, cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].set_title("Foreground mask overlay", fontsize=8)
        axes[row, 1].axis("off")

    fig.suptitle(
        f"{dataset_dir.name}: preprocessed vs. foreground-mask overlay ({len(samples)} writers)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_dir.name}_mask_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_augmentation_comparison(
    dataset_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_rng: random.Random,
    variant_name: str,
    augment_fn: Callable[[np.ndarray, np.random.Generator, AugmentConfig], np.ndarray],
    aug_rng: np.random.Generator,
    config: AugmentConfig,
) -> Path:
    """Save one figure: preprocessed vs. one augmented view, for `num_samples` writers.

    `augment_fn` runs on the raw original and returns the final,
    already-preprocessed 256x256 binary image (see `augment.py`'s module
    docstring for why augmentation has to happen before, not after,
    preprocessing's tight crop), so both panels are directly comparable.
    """
    samples = sample_signatures(dataset_dir, num_samples, sample_rng)
    fig, axes = _new_comparison_figure(len(samples))

    for row, image_path in enumerate(samples):
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        preprocessed = preprocess_signature(original)
        augmented = augment_fn(original, aug_rng, config)

        label = f"writer {image_path.parent.name} / {image_path.name}"

        axes[row, 0].imshow(preprocessed, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Preprocessed\n{label}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(augmented, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Augmented ({variant_name})", fontsize=8)
        axes[row, 1].axis("off")

    fig.suptitle(
        f"{dataset_dir.name}: preprocessed vs. '{variant_name}' augmentation ({len(samples)} writers)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    out_path = variant_dir / f"{dataset_dir.name}_augmentation_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    dataset_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders found under {DATA_ROOT}")

    # Each gallery gets its own rng seeded identically, so both stages
    # sample the exact same 10 writers/images per dataset.
    preprocessing_rng = random.Random(SEED)
    for dataset_dir in dataset_dirs:
        out_path = plot_preprocessing_comparison(
            dataset_dir, PREPROCESSING_OUTPUT_DIR, NUM_SAMPLES, preprocessing_rng
        )
        print(f"Saved {out_path}")

    masking_rng = random.Random(SEED)
    for dataset_dir in dataset_dirs:
        out_path = plot_masking_comparison(
            dataset_dir, MASKING_OUTPUT_DIR, NUM_SAMPLES, masking_rng
        )
        print(f"Saved {out_path}")

    # One shared, continuously-advancing generator for all augmentation
    # randomness; a fresh sample_rng per variant keeps the same 10
    # writers/images picked for every variant (and for the two galleries
    # above), so all outputs are directly comparable.
    aug_rng = np.random.default_rng(SEED)
    for variant_name, augment_fn in AUGMENTATION_VARIANTS.items():
        sample_rng = random.Random(SEED)
        for dataset_dir in dataset_dirs:
            out_path = plot_augmentation_comparison(
                dataset_dir, AUGMENTATION_OUTPUT_DIR, NUM_SAMPLES,
                sample_rng, variant_name, augment_fn, aug_rng, DEFAULT_CONFIG,
            )
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
