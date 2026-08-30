"""Draw DenseCL Figure-4/5-style correspondence galleries.

Two 256x256 views are shown side by side; a line connects each mutually-
matched grid-cell pair (`correspondence.Match`), color-coded by cosine
similarity (red=low, green=high) so match confidence is visible at a
glance instead of needing a hard similarity cutoff. Deliberately NO
thresholding by default - unlike the paper's own Figure 4 (which only
shows similarity >= 0.9 matches for a fully-converged model), a
still-early-training checkpoint's honest match quality is exactly what
this analysis needs to see, matching the paper's own no-threshold choice
for Figure 5's random-init/partially-trained comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

from correspondence import Match

IMAGE_SIZE = 256


@dataclass
class GalleryRow:
    """One row of a correspondence gallery: two views of one signature plus
    their mutual matches. `label` is shown above View A (e.g. writer/file
    identity)."""

    view_a: np.ndarray
    view_b: np.ndarray
    matches: list[Match]
    label: str


def _grid_to_pixel_center(row: int, col: int, grid_size: int, image_size: int = IMAGE_SIZE) -> tuple[float, float]:
    """Grid cell (row, col) -> the pixel coordinates of its center, in the
    (x, y) convention matplotlib's `imshow` uses (x=column axis, y=row axis)."""
    cell = image_size / grid_size
    x = col * cell + cell / 2.0
    y = row * cell + cell / 2.0
    return x, y


def _new_gallery_figure(num_rows: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(num_rows, 2, figsize=(6, 3 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, 2)
    return fig, axes


def plot_correspondence_gallery(
    rows: list[GalleryRow],
    out_path: Path,
    suptitle: str,
    grid_size: int = 32,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Save one gallery figure: one row per `GalleryRow`, View A next to
    View B with mutual-match lines drawn between them, color-coded by
    similarity (colormap 'RdYlGn', normalized over cosine similarity's
    [0, 1] positive range - matches below 0 similarity are rare in
    practice for L2-normalized features and are just clipped to the
    coldest color rather than given their own scale)."""
    fig, axes = _new_gallery_figure(len(rows))
    cmap = plt.get_cmap("RdYlGn")

    for row_index, row in enumerate(rows):
        ax_a, ax_b = axes[row_index, 0], axes[row_index, 1]
        ax_a.imshow(row.view_a, cmap="gray", vmin=0, vmax=255)
        ax_b.imshow(row.view_b, cmap="gray", vmin=0, vmax=255)
        ax_a.set_xlim(0, image_size)
        ax_a.set_ylim(image_size, 0)
        ax_b.set_xlim(0, image_size)
        ax_b.set_ylim(image_size, 0)
        ax_a.axis("off")
        ax_b.axis("off")

        mean_similarity = sum(m.similarity for m in row.matches) / len(row.matches) if row.matches else 0.0
        ax_a.set_title(f"View A\n{row.label}", fontsize=8)
        ax_b.set_title(f"View B\n{len(row.matches)} matches, mean sim={mean_similarity:.2f}", fontsize=8)

        for match in row.matches:
            xa, ya = _grid_to_pixel_center(match.row_a, match.col_a, grid_size, image_size)
            xb, yb = _grid_to_pixel_center(match.row_b, match.col_b, grid_size, image_size)
            color = cmap(max(0.0, min(1.0, match.similarity)))

            connection = ConnectionPatch(
                xyA=(xa, ya), coordsA="data", axesA=ax_a,
                xyB=(xb, yb), coordsB="data", axesB=ax_b,
                color=color, alpha=0.6, linewidth=0.8,
            )
            fig.add_artist(connection)
            ax_a.scatter([xa], [ya], s=6, color=color, zorder=3)
            ax_b.scatter([xb], [yb], s=6, color=color, zorder=3)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
