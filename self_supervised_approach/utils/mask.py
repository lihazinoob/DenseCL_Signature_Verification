"""Foreground (ink) mask computation for the encoder's dense feature grid.

The dense DenseCL correspondence loss operates on the encoder's 32x32
feature grid, not on raw pixels, so the pixel-level binary mask already
present in a preprocessed signature (ink=255, background=0) has to be
summarized down to one "how much ink is in this cell" number per grid
cell, and then turned into a foreground/background decision per cell.
"""

from __future__ import annotations

import numpy as np

GRID_SIZE = 32


def compute_ink_coverage(binary_image: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Fraction of ink pixels in each non-overlapping grid cell.

    Parameters
    ----------
    binary_image:
        A square binary image (e.g. the output of `preprocess_signature`),
        ink pixels > 0, background == 0.
    grid_size:
        Number of cells per side. Must evenly divide the image side length
        (32 for a 256x256 image, matching the encoder's three stride-2
        downsamples: 256 / 8 = 32).

    Returns
    -------
    np.ndarray
        `(grid_size, grid_size)` float32 array; each value is the fraction
        (0.0-1.0) of ink pixels inside that cell's pixel block.
    """
    h, w = binary_image.shape
    if h != w:
        raise ValueError(f"Expected a square image, got shape {binary_image.shape}")
    if h % grid_size != 0:
        raise ValueError(
            f"Image side length {h} is not evenly divisible by grid_size {grid_size}"
        )

    cell = h // grid_size
    ink = (binary_image > 0).astype(np.float32)
    coverage = ink.reshape(grid_size, cell, grid_size, cell).mean(axis=(1, 3))
    return coverage


def foreground_mask_from_coverage(coverage: np.ndarray, min_coverage: float = 0.0) -> np.ndarray:
    """Turn per-cell ink coverage into a foreground/background decision.

    A cell counts as foreground if its ink coverage is strictly greater
    than `min_coverage`. The default (0.0) means "any ink pixel in the
    cell counts" - the grid-cell equivalent of the pixel-level rule
    already used for the reconstruction pretext's foreground-weighted MSE
    (Eq. 3.5 of the thesis report: w(p) = w_fg if t(p) > 0 else w_bg).
    """
    return coverage > min_coverage


def compute_foreground_mask(
    binary_image: np.ndarray, grid_size: int = GRID_SIZE, min_coverage: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: binary image -> (foreground_mask, ink_coverage)."""
    coverage = compute_ink_coverage(binary_image, grid_size=grid_size)
    mask = foreground_mask_from_coverage(coverage, min_coverage=min_coverage)
    return mask, coverage


if __name__ == "__main__":
    import sys

    from preprocess import load_and_preprocess

    if len(sys.argv) != 2:
        print("Usage: python mask.py <path-to-signature-image>")
        raise SystemExit(1)

    _, preprocessed = load_and_preprocess(sys.argv[1])
    fg_mask, coverage = compute_foreground_mask(preprocessed)
    print(f"Grid shape: {fg_mask.shape}")
    print(f"Foreground cells: {fg_mask.sum()} / {fg_mask.size} "
          f"({100 * fg_mask.sum() / fg_mask.size:.1f}%)")
    print(f"Coverage min/mean/max: {coverage.min():.3f}/{coverage.mean():.3f}/{coverage.max():.3f}")
