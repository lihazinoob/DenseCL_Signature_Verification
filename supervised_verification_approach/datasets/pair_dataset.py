"""Pair dataset for downstream double-margin (Step 2) supervised training.

Wraps `DualTripletDataset` rather than reimplementing writer-pool /
per-epoch resampling logic - one already-sampled 4-tuple (anchor,
positive, negative_intra, negative_inter) decomposes into exactly the
three labeled pairs a double-margin contrastive loss needs, per
`docs/claude_response/downstream_supervised_learning_approach.md` SS6:

    (anchor, positive,       label=1)  # same writer, both genuine
    (anchor, negative_intra, label=0)  # same writer, forged
    (anchor, negative_inter, label=0)  # different writer, genuine

`len(dataset)` = 3 * len(quad_dataset), so one epoch still sees every
genuine signature once as an anchor, now expressed as three pair records
instead of one quadruple record. `set_epoch()` forwards to the wrapped
dataset, so the same per-epoch resampling discipline applies unchanged -
and since the wrapped dataset seeds its per-item RNG from
`(epoch, quad_index)`, the three pair records sharing one quad_index
always see the same positive/negative_intra/negative_inter draw for that
epoch, regardless of DataLoader shuffling.

Works with any object exposing `__len__`/`__getitem__` in
`DualTripletDataset`'s 4-tuple-dict shape - in practice either that class
(training) or `FixedDualTripletDataset` (validation). `set_epoch()` is
only ever called on the training wrapper by `train_and_validate_model`
(see `train_validation.py`), so it is not implemented defensively here
for the fixed/validation case.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import Dataset

DATASETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASETS_DIR))

from dual_triplet_dataset import DualTripletDataset  # noqa: E402

_PAIR_OTHER_KEYS = ("positive", "negative_intra", "negative_inter")
_PAIR_LABELS = (1.0, 0.0, 0.0)


class PairDataset(Dataset):
    def __init__(self, quad_dataset: DualTripletDataset) -> None:
        super().__init__()
        self.quad_dataset = quad_dataset

    def __len__(self) -> int:
        return len(self.quad_dataset) * 3

    def set_epoch(self, epoch: int) -> None:
        """Forwards to the wrapped dataset - only valid (and only ever
        called) when wrapping a `DualTripletDataset`."""
        self.quad_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        quad_index, pair_slot = divmod(index, 3)
        quad = self.quad_dataset[quad_index]
        other_key = _PAIR_OTHER_KEYS[pair_slot]
        return {
            "image_a": quad["anchor"],
            "image_b": quad[other_key],
            "label": _PAIR_LABELS[pair_slot],
            "anchor_writer_id": quad["anchor_writer_id"],
        }
