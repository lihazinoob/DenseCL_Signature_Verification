"""PyTorch Dataset producing DenseCL's two-view training tuples.

Given a signature image path, `SignatureSSLDataset.__getitem__` returns two
independently, stochastically augmented views of the same signature (View A,
View B) plus each view's foreground grid mask - exactly the four tensors the
DenseCL pretraining loop needs (global + dense losses, foreground-masked).

No labels are used anywhere here: writer identity is not read for training
purposes, only checked against the held-out test split (see below) so
those writers can be excluded. This is a pretraining-stage dataset, not the
downstream verification dataset.

Writer identity IS used for one thing: excluding the held-out test writers
created by `test_set_creation.py`. A writer-independent verifier's test set
must never be seen by EITHER pipeline stage - not just the downstream
supervised stage, the SSL pretraining stage too - or "held out" stops
meaning anything. See `list_all_signature_paths`'s `exclude_test_writers`
parameter (on by default).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augment import DEFAULT_CONFIG, AugmentConfig, generate_view
from mask import compute_foreground_mask
from test_set_creation import OUTPUT_DIR as TEST_SPLIT_ROOT
from test_set_creation import load_test_writer_ids

# Silences OpenCV's own internal WARNING-level logging (e.g. grfmt_tiff.cpp's
# "TIFFFetchNormalTag: ... Software tag contains null byte" spam - harmless
# metadata truncation on every BHSig260 .tif read, not a real problem) while
# still surfacing actual errors. Set once at import time since this is a
# process-wide OpenCV setting, not per-call.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_all_signature_paths(
    data_root: Path,
    exclude_test_writers: bool = True,
    extra_exclude_writer_ids: dict[str, set[str]] | None = None,
    test_split_dir: Path = TEST_SPLIT_ROOT,
    dataset_names: set[str] | None = None,
) -> list[Path]:
    """Every signature image path under `data_root/<dataset>/<writer>/<file>`.

    `exclude_test_writers=True` (the default, and the only correct setting
    for an actual training run) skips any writer listed in that dataset's
    held-out test split (`test_set_creation.py` /
    `data/test_set_writer_split/<dataset>_test_writers.json`, or
    `<test_split_dir>/<dataset>_test_writers.json` if `test_split_dir` is
    overridden - e.g. `data/test_set_writer_split/fold_1/` for a K-fold CV
    fold, see `create_cv_fold_split.py`). Datasets with no split file under
    `test_split_dir` (Mendeley - see `test_set_creation.py`'s docstring for
    why) have nothing excluded, via `load_test_writer_ids`'s empty-set
    fallback. Only pass `exclude_test_writers=False` for debugging/
    inspection - never for training.

    `extra_exclude_writer_ids`, if given, additionally skips specific writer
    IDs per dataset (e.g. `{"CEDAR": {"3", "7"}}`) - used to also strip the
    SSL validation writers (`validation_set_creation.py`) out of the
    training pool, on top of the test writers.

    `dataset_names`, if given, restricts the walk to only these dataset
    subdirectory names (e.g. `{"BHSig260_Hindi"}`) instead of pooling every
    dataset under `data_root` - the single-dataset SSL pretraining case
    (matched-domain pretrain+finetune control), as opposed to the default
    `None` (pool everything), which is what every prior SSL run used.
    """
    data_root = Path(data_root)
    paths = []
    excluded_writer_counts: dict[str, int] = {}

    for dataset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if dataset_names is not None and dataset_dir.name not in dataset_names:
            continue
        test_writer_ids = load_test_writer_ids(dataset_dir.name, split_dir=test_split_dir) if exclude_test_writers else set()
        extra_ids = (extra_exclude_writer_ids or {}).get(dataset_dir.name, set())
        skip_ids = test_writer_ids | extra_ids
        excluded_writer_counts[dataset_dir.name] = 0

        for writer_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if writer_dir.name in skip_ids:
                excluded_writer_counts[dataset_dir.name] += 1
                continue
            for image_path in sorted(writer_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(image_path)

    for dataset_name, count in excluded_writer_counts.items():
        if count > 0:
            print(f"[SignatureSSLDataset] {dataset_name}: excluded {count} writer(s) (test and/or validation)")

    return paths


def list_specific_writer_signature_paths(data_root: Path, writer_ids: dict[str, set[str]]) -> list[Path]:
    """Every signature image path, but restricted to ONLY the given writer
    IDs per dataset (e.g. `{"CEDAR": {"3", "7"}}`) - the inverse query shape
    from `list_all_signature_paths` ("only these" instead of "everything
    except these"). Used to build the small, fixed SSL validation pool from
    `validation_set_creation.py`'s output."""
    data_root = Path(data_root)
    paths = []
    for dataset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        wanted = writer_ids.get(dataset_dir.name, set())
        if not wanted:
            continue
        for writer_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if writer_dir.name not in wanted:
                continue
            for image_path in sorted(writer_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(image_path)
    return paths


class SignatureSSLDataset(Dataset):
    """Yields (view_a, view_b, mask_a, mask_b) for DenseCL pretraining.

    `view_a`/`view_b`: float32 tensors, shape (1, 256, 256), values in
    [0, 1] (ink=1.0, background=0.0) - the channel dim matches the
    existing thesis encoder's 1-channel input convention.
    `mask_a`/`mask_b`: bool tensors, shape (32, 32), True where the grid
    cell counts as foreground (see `mask.py`'s `min_coverage` default).

    Each `__getitem__` call draws fresh randomness (a `np.random.default_rng`
    seeded from the item index and the epoch-varying `torch` seed state), so
    the same index yields a different augmented pair on every access - the
    normal SSL behavior of re-augmenting every epoch rather than caching one
    fixed pair per image.
    """

    def __init__(
        self,
        data_root: str | Path,
        config: AugmentConfig = DEFAULT_CONFIG,
        grid_size: int = 32,
        min_coverage: float = 0.0,
        exclude_test_writers: bool = True,
        extra_exclude_writer_ids: dict[str, set[str]] | None = None,
        image_paths_override: list[Path] | None = None,
        test_split_dir: Path = TEST_SPLIT_ROOT,
        dataset_names: set[str] | None = None,
    ) -> None:
        """`image_paths_override`, if given, bypasses the normal directory
        walk entirely and uses exactly this list of paths - the mechanism
        `driver/train.py` uses to build the validation dataset from
        `list_specific_writer_signature_paths`'s output (an "only these
        writers" pool, not an "everything except" one). `test_split_dir`
        is forwarded to `list_all_signature_paths` unchanged - see there
        for the K-fold CV use case. `dataset_names` is also forwarded
        unchanged - see `list_all_signature_paths` for the single-dataset
        SSL pretraining use case; ignored when `image_paths_override` is
        given, since that path already specifies its own exact pool."""
        if image_paths_override is not None:
            self.image_paths = image_paths_override
        else:
            self.image_paths = list_all_signature_paths(
                data_root,
                exclude_test_writers=exclude_test_writers,
                extra_exclude_writer_ids=extra_exclude_writer_ids,
                test_split_dir=test_split_dir,
                dataset_names=dataset_names,
            )
        if not self.image_paths:
            raise FileNotFoundError(f"No signature images found under {data_root}")
        self.config = config
        self.grid_size = grid_size
        self.min_coverage = min_coverage

    def __len__(self) -> int:
        return len(self.image_paths)

    def _to_tensor(self, binary_view: np.ndarray) -> torch.Tensor:
        normalized = (binary_view > 0).astype(np.float32)
        return torch.from_numpy(normalized).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[index]
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image at {path}")

        # A fresh, unseeded generator per call: two calls to generate_view
        # below must draw different randomness (else View A == View B), and
        # every subsequent epoch's pass over the same index must also differ.
        rng = np.random.default_rng()

        view_a = generate_view(original, rng, self.config)
        view_b = generate_view(original, rng, self.config)

        mask_a, _ = compute_foreground_mask(view_a, self.grid_size, self.min_coverage)
        mask_b, _ = compute_foreground_mask(view_b, self.grid_size, self.min_coverage)

        return {
            "view_a": self._to_tensor(view_a),
            "view_b": self._to_tensor(view_b),
            "mask_a": torch.from_numpy(mask_a),
            "mask_b": torch.from_numpy(mask_b),
        }


if __name__ == "__main__":
    from pathlib import Path as _Path

    SCRIPT_DIR = _Path(__file__).resolve().parent
    DATA_ROOT = SCRIPT_DIR.parent.parent / "data" / "all"

    dataset = SignatureSSLDataset(DATA_ROOT)
    print(f"Dataset size: {len(dataset)} signature images")

    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    fg_frac_a = sample["mask_a"].float().mean().item()
    fg_frac_b = sample["mask_b"].float().mean().item()
    print(f"View A foreground fraction: {fg_frac_a:.3f}")
    print(f"View B foreground fraction: {fg_frac_b:.3f}")
