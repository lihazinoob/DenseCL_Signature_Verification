"""Mutual-nearest-neighbor dense correspondence matching (DenseCL Figure 4/5 style).

Reproduces the matching rule described in the DenseCL paper (Section 3.4,
"Dense correspondence visualization"): for two dense feature grids from two
views of the same image, match each grid cell in view A to its most similar
(cosine similarity) cell in view B, and independently match each cell in
view B back to view A. A pair is kept only if it agrees in BOTH directions
(patch A's best match is patch B, AND patch B's best match is patch A) - a
one-directional argmax alone lets one generic-looking patch in B "steal" the
best-match slot for several different patches in A, which the mutual check
filters out.

One deliberate addition beyond the paper, specific to signatures: matching
is restricted to foreground (ink) grid cells only, via the same `mask_a`/
`mask_b` the DenseCL training loss itself already uses. The paper's natural
images are textured almost everywhere, so unrestricted matching is fine
there; a signature's 32x32 grid is ~88% blank paper, so without this
restriction the visualization would mostly show trivial
background-matches-background pairs.

Matching is computed on the BACKBONE's dense feature grid (`Encoder(x,
pool=False)`'s raw output), not the dense projection head's output - this
matches the paper's own choice, which their Table 6 ablation found gives
better correspondence quality than matching on the projected features (the
projection head is optimized to make the contrastive loss easy, which can
discard information useful for general-purpose patch identity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    """One mutually-agreed correspondence between a foreground cell in view
    A's grid and a foreground cell in view B's grid. `row`/`col` are grid
    coordinates (0..grid_size-1), not pixel coordinates - `visualize.py`
    converts these to pixel positions for drawing."""

    row_a: int
    col_a: int
    row_b: int
    col_b: int
    similarity: float


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return features / norms


def compute_dense_correspondence(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> list[Match]:
    """Find mutual nearest-neighbor matches between two dense feature grids,
    restricted to foreground cells in each view.

    Parameters
    ----------
    feat_a, feat_b:
        `(grid_size, grid_size, feature_dim)` dense feature grids (e.g. the
        encoder's `(32, 32, 256)` output for view A / view B of one image).
        Not required to be pre-normalized - cosine similarity is computed
        here.
    mask_a, mask_b:
        `(grid_size, grid_size)` boolean foreground masks for each view.

    Returns
    -------
    list[Match]
        One entry per mutually-agreed correspondence, in no particular
        order. Empty if either view has zero foreground cells.
    """
    if feat_a.shape != feat_b.shape:
        raise ValueError(f"feat_a/feat_b shape mismatch: {feat_a.shape} vs {feat_b.shape}")
    if mask_a.shape != mask_b.shape or mask_a.shape != feat_a.shape[:2]:
        raise ValueError(
            f"mask shape must match feature grid's spatial shape: "
            f"mask_a={mask_a.shape}, mask_b={mask_b.shape}, feat={feat_a.shape[:2]}"
        )

    grid_size = feat_a.shape[0]
    feature_dim = feat_a.shape[2]

    flat_feat_a = feat_a.reshape(grid_size * grid_size, feature_dim)
    flat_feat_b = feat_b.reshape(grid_size * grid_size, feature_dim)
    flat_mask_a = mask_a.reshape(grid_size * grid_size)
    flat_mask_b = mask_b.reshape(grid_size * grid_size)

    fg_indices_a = np.flatnonzero(flat_mask_a)
    fg_indices_b = np.flatnonzero(flat_mask_b)
    if fg_indices_a.size == 0 or fg_indices_b.size == 0:
        return []

    sub_a = _l2_normalize_rows(flat_feat_a[fg_indices_a])  # (num_fg_a, feature_dim)
    sub_b = _l2_normalize_rows(flat_feat_b[fg_indices_b])  # (num_fg_b, feature_dim)

    similarity = sub_a @ sub_b.T  # (num_fg_a, num_fg_b) cosine similarity

    best_b_for_a = similarity.argmax(axis=1)  # for each row in sub_a, its best column in sub_b
    best_a_for_b = similarity.argmax(axis=0)  # for each column in sub_b, its best row in sub_a

    matches: list[Match] = []
    for a_local, b_local in enumerate(best_b_for_a):
        if best_a_for_b[b_local] != a_local:
            continue  # not a mutual match - one-directional only, discard

        grid_index_a = int(fg_indices_a[a_local])
        grid_index_b = int(fg_indices_b[b_local])
        row_a, col_a = divmod(grid_index_a, grid_size)
        row_b, col_b = divmod(grid_index_b, grid_size)

        matches.append(Match(
            row_a=row_a, col_a=col_a,
            row_b=row_b, col_b=col_b,
            similarity=float(similarity[a_local, b_local]),
        ))

    return matches


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    grid_size, feature_dim = 8, 16

    # Sanity check: identical features everywhere in the foreground should
    # produce a match for every foreground cell (each cell matches itself
    # trivially, since all cells are identical - a stronger, more realistic
    # check would use randomized-but-correlated features, but this at least
    # confirms the matching/masking plumbing runs end to end).
    shared = rng.normal(size=(grid_size, grid_size, feature_dim)).astype(np.float32)
    mask = rng.random((grid_size, grid_size)) > 0.5

    matches = compute_dense_correspondence(shared, shared, mask, mask)
    print(f"Foreground cells: {mask.sum()} / {mask.size}")
    print(f"Mutual matches found: {len(matches)}")
    if matches:
        print(f"Example match: {matches[0]}")
