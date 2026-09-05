"""4-tuple dataset for downstream dual-triplet supervised verification training.

Adapted from the SURDS-era `DynamicDualTripletDataset`
(`Thesis_Final/downstream_verification/datasets/DynamicDualTripletDataset.py`) -
same sampling logic (anchor / positive / negative_intra / negative_inter,
re-sampled fresh every epoch via `set_epoch()` so a small, fixed pool of
images is spread over combinatorially many distinct training tuples instead
of the model memorizing one fixed set of quadruples), but with two
deliberate changes:

  - Takes `writer_ids` + `dataset_dir` directly instead of a pre-built
    inventory DataFrame - the genuine/forged filename-parsing ("inventory")
    logic is folded straight into this module instead of kept as a
    separate step (see
    `docs/claude_response/downstream_supervised_learning_approach.md`).
  - Uses THIS project's preprocessing (`preprocess_signature`) and tensor
    convention (binary {0,1} float32) instead of the old pipeline's
    `ToTensor()` + `Normalize(mean=0.5, std=0.5)` (which produces a
    {-1,+1}-valued tensor). This is not a stylistic choice: the
    SSL-pretrained encoder was trained on {0,1} binary tensors
    (`self_supervised_approach/utils/dataset.py`, `(binary_view > 0).astype(np.float32)`).
    Feeding it {-1,+1} inputs now would silently put every signature
    out-of-distribution for the pretrained weights.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = DATASETS_DIR.parent  # DenseCL_approach/supervised_verification_approach
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent  # DenseCL_approach
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))

from preprocess import preprocess_signature  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABELED_DATASETS = ("CEDAR", "BHSig260_Bengali", "BHSig260_Hindi")  # Mendeley has no genuine/forged labels


def list_genuine_and_forged_images(writer_dir: Path, dataset_name: str) -> tuple[list[Path], list[Path]]:
    """Genuine and forged signature paths for one writer. Naming conventions
    (confirmed on disk; genuine-only half of this already exists in
    `tools/writer_separation_analysis/embedding_extraction.py`'s
    `list_genuine_images` - duplicated here in miniature rather than
    imported, since that module lives under `tools/`, a leaf analysis app,
    not a shared library):
      - CEDAR: `original_<writer>_<n>.png` (genuine) vs `forgeries_...` (forged)
      - BHSig260 (Bengali/Hindi): `*-G-*.tif` (genuine) vs `*-F-*.tif` (forged)
    """
    if dataset_name not in LABELED_DATASETS:
        raise ValueError(f"Unsupported dataset for genuine/forged labeling: {dataset_name}")

    all_images = sorted(p for p in writer_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if dataset_name == "CEDAR":
        genuine = [p for p in all_images if p.name.lower().startswith("original_")]
        forged = [p for p in all_images if p.name.lower().startswith("forgeries_")]
    else:
        genuine = [p for p in all_images if "-g-" in p.name.lower()]
        forged = [p for p in all_images if "-f-" in p.name.lower()]
    return genuine, forged


def load_signature_tensor(image_path: Path) -> torch.Tensor:
    """Signature file -> preprocessed 256x256 view -> binary {0,1} float32
    tensor, shape (1, 256, 256) - the exact input convention the
    SSL-pretrained encoder was trained on."""
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    view = preprocess_signature(raw)
    binary = (view > 0).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


class DualTripletDataset(Dataset):
    """Emits 4-tuples: (anchor_genuine, positive_genuine, negative_intra_forgery, negative_inter_genuine).

    - anchor and positive are both genuine signatures from the same writer.
    - negative_intra is a skilled forgery of the same writer (intra-writer, wrong type).
    - negative_inter is a genuine signature from a *different* writer (inter-writer, right type).

    One anchor record exists per genuine signature in the writer pool, so
    `len(dataset)` = total genuine count and one epoch uses every genuine
    once as an anchor. The other three tuple members are resampled at
    random on every `__getitem__` call; call `set_epoch()` before each
    epoch so the random draw changes epoch to epoch instead of the model
    seeing the exact same quadruples every time.
    """

    def __init__(
        self,
        writer_ids: list[str],
        dataset_dir: Path,
        dataset_name: str,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.current_epoch = 0

        self.writer_to_genuine_paths: dict[str, list[Path]] = {}
        self.writer_to_forgery_paths: dict[str, list[Path]] = {}
        for writer_id in writer_ids:
            writer_dir = dataset_dir / writer_id
            if not writer_dir.is_dir():
                raise FileNotFoundError(f"Writer directory not found: {writer_dir}")

            genuine_paths, forgery_paths = list_genuine_and_forged_images(writer_dir, dataset_name)
            if len(genuine_paths) < 2:
                raise ValueError(
                    f"Writer {writer_id} has fewer than two genuine signatures "
                    f"(found {len(genuine_paths)}). Cannot form anchor-positive pairs."
                )
            if len(forgery_paths) < 1:
                raise ValueError(f"Writer {writer_id} has no forgery signatures.")

            self.writer_to_genuine_paths[writer_id] = genuine_paths
            self.writer_to_forgery_paths[writer_id] = forgery_paths

        self.writer_ids: list[str] = sorted(self.writer_to_genuine_paths.keys(), key=int)
        if len(self.writer_ids) < 2:
            raise ValueError(
                f"Need at least two writers for inter-negative sampling (found {len(self.writer_ids)})."
            )

        self.anchor_records: list[tuple[str, Path]] = [
            (writer_id, path)
            for writer_id in self.writer_ids
            for path in self.writer_to_genuine_paths[writer_id]
        ]

    def __len__(self) -> int:
        return len(self.anchor_records)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch's DataLoader iteration to refresh tuple sampling."""
        self.current_epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        anchor_writer_id, anchor_path = self.anchor_records[index]
        rng = random.Random(self.seed + (self.current_epoch * 100003) + index)

        positive_candidates = [p for p in self.writer_to_genuine_paths[anchor_writer_id] if p != anchor_path]
        positive_path = rng.choice(positive_candidates)

        negative_intra_path = rng.choice(self.writer_to_forgery_paths[anchor_writer_id])

        inter_writer_id = rng.choice([wid for wid in self.writer_ids if wid != anchor_writer_id])
        negative_inter_path = rng.choice(self.writer_to_genuine_paths[inter_writer_id])

        return {
            "anchor": load_signature_tensor(anchor_path),
            "positive": load_signature_tensor(positive_path),
            "negative_intra": load_signature_tensor(negative_intra_path),
            "negative_inter": load_signature_tensor(negative_inter_path),
            "anchor_writer_id": anchor_writer_id,
            "inter_writer_id": inter_writer_id,
        }
