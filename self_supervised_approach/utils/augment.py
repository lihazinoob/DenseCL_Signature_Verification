"""Stroke-safe augmentations for producing DenseCL's two independent views.

Augmentation runs on the RAW grayscale original (dark ink, light/white
background - ink near 0, paper near 255), *before* the tight
crop-to-ink-bbox step in `preprocess_signature`, not after it. Warping an
already tightly-cropped 256x256 image was the original design here, but
it has no room: a tightly-bound signature has strokes sitting right at
the canvas edge, and rotating it pushes those strokes past a fixed-size
output frame with nowhere to go, permanently clipping real content -
padding the *output* canvas doesn't fix this, since cropping back down to
the original footprint afterward throws the displaced content away
regardless of how much blank padding surrounded it during the warp.

Augmenting the raw original first, keeping the canvas generously padded
throughout (never cropping back down inside this module), and only
letting the *existing* `preprocess_signature` crop tightly *after*
augmentation is done fixes this: the tight crop is always computed from
the post-augmentation content, so it can never cut anything off.

Two stages, split at exactly the one point where binarization is safe to
do (see `add_boundary_noise` for why):

  Stage 1 (raw grayscale, ink=low/dark, background=high/light - matches
  a raw scan): `augment_view` chains random occlusion (`random_cutout`),
  geometric warps (`random_affine`, `elastic_warp`), and stroke-width
  jitter (`dilate_strokes`, `erode_strokes`, `random_morphology`). Output
  is unbinarized and usually larger than the input, since the geometric
  warps enlarge the canvas rather than cropping back down - nothing is
  thresholded or cropped tight until `preprocess_signature` runs on it.

  Stage 2 (post `preprocess_signature`, ink=255/background=0 - the
  opposite polarity from stage 1): `add_boundary_noise` jitters stroke
  edges on the final, already tightly-cropped 256x256 binary image.

`generate_view` chains all three (`augment_view` -> `preprocess_signature`
-> `add_boundary_noise`) into the one function that produces a complete
training view from a raw original.

Deliberately excludes anything meant for continuous-tone photos (color
jitter, solarization, blur, aggressive random crop) since those are
either meaningless or destructive on a signature scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from preprocess import otsu_binarize, preprocess_signature

PAPER = 255  # background fill value for a raw grayscale scan (light paper)


def _pad_canvas(image: np.ndarray, pad: int) -> np.ndarray:
    """Add a `pad`-pixel blank (paper-white) border on every side."""
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=PAPER)


@dataclass(frozen=True)
class AugmentConfig:
    """Tunable parameters for stroke-safe augmentation.

    Defaults are starting points, not final values - meant to be swept
    the same way `w_fg` (Experiment 1) and `min_coverage` were, once a
    downstream validation signal exists to judge them against.

    Strengthened once already (see chat): the first-pass defaults below
    (rotation 8 deg, scale +-8%, shear 6 deg, elastic alpha 8/sigma 6,
    morph_kernel 3, noise_std 25) turned out to be too gentle - visually,
    only dilation produced a clearly "hard" example; rotation/shear/elastic/
    noise were barely visible after the eventual resize-to-256. The goal
    of this augmentation is NOT to mimic how much a real signature
    naturally varies between two genuine samples - it's to force the
    encoder to discover invariant stroke structure by making the pretext
    task genuinely hard, the same logic DenseCL/MoCo/SimCLR use on natural
    images (whose augmentations go well beyond a single photo's natural
    variation). The only real ceiling is "don't distort so far the
    signature becomes a different topology" (loops merging, strokes
    shattering) - see `morph_kernel`'s note.
    """

    rotation_deg: float = 18.0         # max abs rotation, degrees (was 8.0)
    scale_range: tuple[float, float] = (0.85, 1.15)  # min/max scale factor (was 0.92-1.08)
    shear_deg: float = 11.0            # max abs shear angle, degrees (was 6.0)

    elastic_alpha: float = 18.0        # max pixel displacement magnitude (was 8.0)
    elastic_sigma: float = 9.0         # smoothness of the displacement field (was 6.0)
    # alpha and sigma raised TOGETHER, not alpha alone: sigma controls how
    # spatially smooth the displacement field is, and signature strokes are
    # only ~2-4px wide at 256 resolution - a bigger alpha without a
    # correspondingly bigger sigma risks a jagged, high-frequency warp that
    # tears strokes apart rather than flexing them, the same failure mode
    # erosion had (see morph_prob's note below).

    morph_kernel: int = 5              # structuring element side length, odd (was 3)
    # Ceiling to watch for: a kernel this size can fuse separate nearby
    # strokes together (e.g. closing a loop that should stay open) rather
    # than just thickening them - a topology change, not just a thickness
    # change. Re-verified visually (gallery) that this doesn't happen at 5;
    # would need to drop back to 3 if a stronger effect were needed and
    # fusing appeared instead.
    morph_prob: float = 0.35           # probability dilation (thickening) is applied
    # (0.35, not the original 0.7: that was the chance of *either* thicken
    # or thin firing, 50/50 - now that random_morphology only thickens,
    # keeping 0.7 would silently double how often it fires. 0.35 keeps
    # the per-direction rate the same as before erosion was dropped.)

    noise_std: float = 35.0            # gaussian noise std, 0-255 scale (was 25.0)
    # Deliberately still the weakest lever here, by design, not an
    # oversight: this runs last, on the already-clean binary image, and
    # gets partially undone by its own immediate re-threshold - its job is
    # fine boundary jitter (mimicking scan/print roughness), not a primary
    # difficulty driver. Rotation/shear/elastic/morphology do that work.

    cutout_prob: float = 0.3           # probability a cutout pass is applied at all
    cutout_count: int = 1              # number of erased patches per application
    cutout_size_range: tuple[float, float] = (0.05, 0.15)  # patch side, as a fraction of the image's shorter side
    # New augmentation (see chat): small random rectangular patches erased
    # to paper-white, simulating pen skips / faded strokes / overlapping
    # ink - a well-established general technique (often called "Cutout" /
    # "Random Erasing" in the wider computer-vision literature), not
    # previously used in this pipeline. Runs FIRST in `augment_view` (see
    # that function), before any geometric warp, so the erased patch
    # itself gets naturally rotated/warped along with everything else
    # rather than looking like an axis-aligned rectangle stamped on top of
    # an already-warped image. Sized relative to the RAW original's own
    # dimensions (not the later padded/warped canvas), so it reliably lands
    # somewhere near the actual signature content rather than in blank
    # padding by chance. Specifically targets the dense branch: forces
    # patch-level correspondence to hold even when part of a stroke's
    # local neighborhood is missing, a more targeted "hard example" for
    # DenseCL's actual matching mechanism than the whole-image geometric
    # transforms are.


DEFAULT_CONFIG = AugmentConfig()


def _affine_padding(h: int, w: int, config: AugmentConfig) -> int:
    """Border (px) needed so rotation/scale/shear can't push ink off-canvas.

    Sized from the worst-case combined displacement of the farthest
    corner point under the configured rotation/scale/shear ranges, for
    *this specific image's* own dimensions - raw originals vary a lot in
    size and aspect ratio across datasets (BHSig260 scans run ~279x950,
    for instance), so this can't be a fixed constant.
    """
    half_diag = math.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)
    rotation_shift = half_diag * math.sin(math.radians(config.rotation_deg))
    scale_shift = half_diag * (max(config.scale_range) - 1.0)
    shear_shift = half_diag * math.tan(math.radians(config.shear_deg))
    return int(math.ceil(rotation_shift + scale_shift + shear_shift)) + 8  # safety buffer


def _elastic_padding(config: AugmentConfig) -> int:
    """Border (px) needed so the elastic displacement field can't push ink off-canvas."""
    return int(math.ceil(config.elastic_alpha)) + 8


def cutout_strokes(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Unconditionally erase `cutout_count` small random rectangular
    patches to paper-white on the RAW original. `rng` is still used - to
    choose WHERE the patch(es) land - but this always applies at least one
    erasure; `random_cutout` (below) is the probability-gated wrapper that
    decides WHETHER to call this at all, same split as `dilate_strokes`
    (unconditional effect) vs. `random_morphology` (its gated wrapper).

    Only ever removes ink, never adds any - unlike the noise-before-crop
    bug this pipeline already fixed once (`add_boundary_noise`'s
    docstring), erasing a small patch can only shrink the eventual ink
    bounding box slightly, never blow it out, so this is safe to run
    pre-crop alongside the other `augment_view` steps.

    Placement is confined to the actual ink bounding box (via
    `otsu_binarize`, same convention as `preprocess.py`'s own bbox step),
    NOT sampled uniformly over the whole raw image - fixed a real bug found
    by regression-testing this exact function: raw scans have substantial
    blank paper margin around the signature, so a patch placed uniformly
    over the full image landed on already-blank background in ~80% of the
    trials where it "fired" (only 3/18 fired attempts actually touched ink).

    Bbox-confinement alone wasn't enough, though (re-tested: only improved
    to 6/18) - a signature is sparse even WITHIN its own tight bounding box
    (loops, gaps between letters/words; measured ~12% ink density there).
    So placement additionally uses rejection sampling: try up to 10 random
    positions within the bbox and keep the first one that actually
    overlaps ink, falling back to the last attempt in the rare case none
    do. This is what makes a "fired" cutout reliably occlude real stroke
    content instead of silently landing on blank space most of the time.
    """
    out = image.copy()
    h, w = out.shape

    ink_mask = otsu_binarize(image) > 0
    ys, xs = np.nonzero(ink_mask)
    if ys.size == 0:
        return out  # blank/corrupt scan - nothing to occlude

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    # Reference the bbox's LONGER extent, not the shorter one: `_square_pad`
    # pads the shorter side up to match the longer one, then resizes that
    # square down to 256 - so it's the longer extent that sets the eventual
    # downscale factor. Sizing against the shorter extent instead (an
    # earlier version of this function did) made cutout patches shrink to
    # just a handful of pixels after the final resize for elongated bboxes
    # like BHSig260's (short, wide signatures) - confirmed empirically via
    # the gallery: some patches ended up as small as 1-4 changed pixels in
    # the final 256x256 image, nowhere near the intended 5-15%.
    reference_extent = max(max(y_max - y_min, x_max - x_min), 1)

    for _ in range(config.cutout_count):
        size = max(1, int(rng.uniform(*config.cutout_size_range) * reference_extent))
        y_range = max(y_max - y_min - size, 1)
        x_range = max(x_max - x_min - size, 1)

        chosen_y0, chosen_x0 = None, None
        for _attempt in range(10):
            y0 = min(y_min + int(rng.integers(0, y_range)), h - size)
            x0 = min(x_min + int(rng.integers(0, x_range)), w - size)
            if chosen_y0 is None:
                chosen_y0, chosen_x0 = y0, x0  # fallback if no attempt overlaps ink
            if ink_mask[y0:y0 + size, x0:x0 + size].any():
                chosen_y0, chosen_x0 = y0, x0
                break

        out[chosen_y0:chosen_y0 + size, chosen_x0:chosen_x0 + size] = PAPER

    return out


def random_cutout(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """With probability `cutout_prob`, call `cutout_strokes`; otherwise
    leave the image unchanged. This is what `augment_view` calls."""
    if rng.random() >= config.cutout_prob:
        return image
    return cutout_strokes(image, rng, config)


def random_affine(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Random rotation + scale + shear about the image center.

    Pads the canvas first (see `_affine_padding`) and does NOT crop back
    down afterward - the output is intentionally larger than the input,
    so displaced strokes have somewhere to land instead of being clipped.
    The caller runs `preprocess_signature` on the final result, which
    crops tightly to wherever the ink actually ended up.
    """
    h, w = image.shape
    pad = _affine_padding(h, w, config)
    padded = _pad_canvas(image, pad)
    ph, pw = padded.shape

    angle = rng.uniform(-config.rotation_deg, config.rotation_deg)
    scale = rng.uniform(config.scale_range[0], config.scale_range[1])
    shear = np.tan(np.deg2rad(rng.uniform(-config.shear_deg, config.shear_deg)))

    def to_3x3(m: np.ndarray) -> np.ndarray:
        return np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float32)

    rot_scale = to_3x3(cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, scale))
    shear_mat = to_3x3(np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    affine_matrix = (rot_scale @ shear_mat)[:2, :]

    return cv2.warpAffine(
        padded, affine_matrix, (pw, ph),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PAPER,
    )


def elastic_warp(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Smooth random per-pixel displacement field (local stroke wobble).

    Same padded-canvas, no-crop-back treatment as `random_affine`, for
    the same reason.
    """
    h, w = image.shape
    pad = _elastic_padding(config)
    padded = _pad_canvas(image, pad)
    ph, pw = padded.shape

    dx = rng.uniform(-1.0, 1.0, size=(ph, pw)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, size=(ph, pw)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=config.elastic_sigma) * config.elastic_alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=config.elastic_sigma) * config.elastic_alpha

    x, y = np.meshgrid(np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32))
    map_x = x + dx
    map_y = y + dy

    return cv2.remap(
        padded, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PAPER,
    )


def _stroke_kernel(config: AugmentConfig) -> np.ndarray:
    # Elliptical (cross-like at small sizes), not a full square: a square
    # kernel requires all 8 neighbors (incl. diagonals) to agree, which
    # shatters thin strokes into disconnected speckle instead of just
    # thinning them. The elliptical kernel is the gentler standard choice
    # for thin curved structures.
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel, config.morph_kernel))


def dilate_strokes(image: np.ndarray, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Thicken (dark) ink strokes by one structuring-element radius.

    Ink is the LOW-intensity value here (raw grayscale: dark ink, light
    paper), so growing the dark region means taking the local MINIMUM,
    which is `cv2.erode` - not `cv2.dilate`. (Named for what it does to
    the ink, not for which OpenCV call implements it.)
    """
    return cv2.erode(image, _stroke_kernel(config), iterations=1)


def erode_strokes(image: np.ndarray, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Thin (dark) ink strokes by one structuring-element radius.

    Thinning the dark ink means growing the light background - the local
    MAXIMUM, i.e. `cv2.dilate` - for the same polarity reason as
    `dilate_strokes` above.
    """
    return cv2.dilate(image, _stroke_kernel(config), iterations=1)


def random_morphology(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """With probability `morph_prob`, thicken strokes; otherwise leave unchanged.

    Erosion (thinning) is deliberately excluded from this default policy.
    On strokes that are already only a few pixels wide post-preprocessing,
    even the gentle elliptical kernel used here disconnects them into
    disjoint fragments rather than just thinning them - confirmed
    visually in orchestration.py's "erode" demo gallery. Unlike rotation,
    shear, elastic wobble, or boundary noise, that fragmentation isn't a
    realistic stand-in for real intra-writer variation: a genuinely
    lighter or faster pen stroke stays thin but continuous, it doesn't
    drop out mid-stroke. Training the pretext to match a shattered stroke
    against its intact counterpart teaches a correspondence that has no
    analog in any real genuine/forgery comparison. `erode_strokes` is
    kept as a standalone function, and in the demo gallery, so this
    failure mode stays visible even though it isn't used here.
    """
    if rng.random() >= config.morph_prob:
        return image
    return dilate_strokes(image, config)


def augment_view(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Cutout + geometric + stroke-width jitter stage: random cutout ->
    affine -> elastic warp -> random morphology. Returns a grayscale
    image, generally larger than the input and NOT yet binarized - the
    caller runs `preprocess_signature` on the result to get the final
    clean 256x256 view (see `generate_view`, which does this for you).

    Cutout runs FIRST, before any geometric warp, so the erased patch
    itself gets naturally rotated/warped along with everything else
    rather than appearing as an axis-aligned rectangle stamped onto an
    already-warped image.

    Deliberately does NOT include noise (see `add_boundary_noise` for
    why) or re-threshold internally (see `add_boundary_noise`'s docstring
    for how repeated re-thresholding erased small features like a
    diacritic dot in an earlier version of this pipeline).
    """
    out = random_cutout(image, rng, config)
    out = random_affine(out, rng, config)
    out = elastic_warp(out, rng, config)
    out = random_morphology(out, rng, config)
    return out


def add_boundary_noise(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Jitter stroke boundaries on an already-preprocessed binary image
    (ink=255, background=0) - run this AFTER `preprocess_signature`, not
    as part of the pre-crop `augment_view` chain.

    Noise used to run pre-crop, across the large padded working canvas
    `augment_view` produces. That was actively dangerous: the crop-to-
    bbox step in `preprocess_signature` computes the ink bounding box as
    a plain min/max over thresholded pixel coordinates, which has zero
    outlier robustness. Adding noise across a canvas that's mostly blank
    padding (often several times larger than the final 256x256, since
    e.g. a BHSig260 scan pads out to ~650x1300) creates enough scattered
    false-positive "ink" pixels that the computed bounding box can blow
    out to cover almost the entire padded canvas - confirmed empirically:
    21.5% of a 649x1319 canvas got classified as ink, with a bounding box
    spanning the full frame. Doing noise here instead, on the final,
    already tightly-cropped and fixed-size 256x256 canvas, is safe: there
    is no further cropping downstream for stray pixels to corrupt, and
    the dense, mostly-signature content means Otsu's threshold stays well
    calibrated.
    """
    noise = rng.normal(0.0, config.noise_std, size=image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Plain THRESH_BINARY, not INV: `image` is already in the ink=255/
    # background=0 convention (preprocess_signature's output), the
    # opposite polarity from the raw-scan convention the rest of this
    # module works in.
    _, binary = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def generate_view(image: np.ndarray, rng: np.random.Generator, config: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """End-to-end: raw grayscale original -> one independently-augmented,
    clean 256x256 binary view (ink=255, background=0).

    This is the function View A / View B generation should call: it
    chains `augment_view` (geometric warps + stroke-width jitter on the
    raw original), `preprocess_signature` (the one tight crop and the
    one primary binarization), and `add_boundary_noise` (a second,
    deliberately-last, safe-to-repeat threshold pass for boundary
    jitter). Call this twice with the same `rng` (which advances between
    calls) to get View A and View B for the same source signature.
    """
    raw = augment_view(image, rng, config)
    clean = preprocess_signature(raw)
    return add_boundary_noise(clean, rng, config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python augment.py <path-to-signature-image>")
        raise SystemExit(1)

    original = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(sys.argv[1])

    rng = np.random.default_rng(0)
    view_a = generate_view(original, rng, DEFAULT_CONFIG)
    view_b = generate_view(original, rng, DEFAULT_CONFIG)

    print(f"Original shape: {original.shape}")
    print(f"View A: {view_a.shape}, ink px={int((view_a > 0).sum())}")
    print(f"View B: {view_b.shape}, ink px={int((view_b > 0).sum())}")
