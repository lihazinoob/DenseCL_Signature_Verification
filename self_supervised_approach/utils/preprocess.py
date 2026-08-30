"""Deterministic preprocessing pipeline for offline signature images.

Pipeline: grayscale -> Otsu binarize -> crop to ink bounding box ->
square-pad -> resize to 256x256 -> Otsu again -> clean binary image.

The output is a 256x256 uint8 binary image with ink strokes at maximum
intensity (255) against a uniform black (0) background, matching the
preprocessing described for the reconstruction-SSL baseline (Fig 3.2 of
the thesis report) and reused unchanged for the DenseCL pretext.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE = 256


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold with inversion so ink pixels become the foreground (255).

    Public because `augment.py` reuses this exact step to re-binarize every
    augmented view after geometric/intensity perturbations, keeping the
    "always clean binary in, clean binary out" invariant throughout.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _crop_to_ink_bbox(gray: np.ndarray, ink_mask: np.ndarray) -> np.ndarray:
    """Crop `gray` to the tight bounding box of the non-zero pixels in `ink_mask`."""
    ys, xs = np.nonzero(ink_mask)
    if ys.size == 0 or xs.size == 0:
        # No ink detected (blank/corrupt scan) - fall back to the full image
        # rather than crashing on an empty crop.
        return gray
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return gray[y0:y1, x0:x1]


def _square_pad(gray: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """Center `gray` on a square canvas, padding with `fill_value` (white)."""
    h, w = gray.shape
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        gray, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=fill_value,
    )


def preprocess_signature(gray: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Run the full deterministic preprocessing pipeline on one signature image.

    Parameters
    ----------
    gray:
        Single-channel (grayscale) signature image, as returned by
        ``cv2.imread(path, cv2.IMREAD_GRAYSCALE)``.
    target_size:
        Side length of the final square output image.

    Returns
    -------
    np.ndarray
        A ``target_size x target_size`` uint8 binary image with ink
        strokes at maximum intensity (255) against a black (0) background.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a single-channel grayscale image, got shape {gray.shape}")

    # Step 1: locate ink strokes via Otsu binarization. This mask is used
    # only to find the crop region and is discarded afterwards.
    ink_mask = otsu_binarize(gray)

    # Step 2: crop to the tight bounding box around the ink so writer- and
    # scanner-dependent margins don't affect downstream scale.
    cropped = _crop_to_ink_bbox(gray, ink_mask)

    # Step 3: pad to a square canvas (centered) to preserve aspect ratio
    # and avoid the anisotropic distortion a direct resize would introduce.
    squared = _square_pad(cropped, fill_value=255)

    # Step 4: resize to the fixed resolution the encoder consumes.
    resized = cv2.resize(
        squared, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    # Step 5: re-binarize to remove grayscale interpolation artifacts
    # introduced by resizing, yielding a clean binary signature.
    clean_binary = otsu_binarize(resized)

    return clean_binary


def load_and_preprocess(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image from disk and return `(original_grayscale, preprocessed)`."""
    path = Path(path)
    original_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    preprocessed = preprocess_signature(original_gray)
    return original_gray, preprocessed


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path-to-signature-image>")
        raise SystemExit(1)

    orig, clean = load_and_preprocess(sys.argv[1])
    print(f"Original shape:     {orig.shape}")
    print(f"Preprocessed shape: {clean.shape}, dtype={clean.dtype}, "
          f"unique values={np.unique(clean)}")
