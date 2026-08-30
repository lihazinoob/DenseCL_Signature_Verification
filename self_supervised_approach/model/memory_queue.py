"""FIFO memory queue of global feature vectors, MoCo-style.

The global InfoNCE loss (see `losses.py`) needs a large, diverse pool of
"different image" negatives to contrast each positive pair against, without
having to re-encode thousands of images every training step. This queue
holds a running buffer of recent momentum-encoder global vectors (`z_b`
from `heads.GlobalHead`, already L2-normalized) and reuses them as
negatives for many steps after they were computed.

Usage order matters: call `get()` to fetch the current negatives and
compute the loss BEFORE calling `enqueue()` with the current batch's
momentum vectors. Enqueueing first would let the current batch's own
positive keys leak into its own negative pool, defeating the point of using
*past* batches for diversity.

NOTE ON THE FILE NAME: deliberately not `queue.py` - that would shadow
Python's own standard-library `queue` module, which `torch`'s multi-worker
DataLoader relies on internally (`num_workers > 0` spawns worker processes
that import it). A flat-import module named `queue.py` sitting on `sys.path`
ahead of the standard library would silently break that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import PROJECTION_DIM

DEFAULT_QUEUE_SIZE = 4096  # ~18% of the ~22,680-image pretraining set - tunable, not load-bearing


class MemoryQueue(nn.Module):
    """Fixed-size FIFO buffer of L2-normalized global vectors."""

    def __init__(self, dim: int = PROJECTION_DIM, size: int = DEFAULT_QUEUE_SIZE) -> None:
        super().__init__()
        self.dim = dim
        self.size = size
        # Random-but-normalized init: never queried before the first enqueue()
        # in practice (the training loop should warm up briefly), but a
        # normalized init keeps early cosine similarities well-behaved if it
        # ever is.
        self.register_buffer("vectors", F.normalize(torch.randn(size, dim), dim=1))
        self.register_buffer("pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """Push a batch of L2-normalized momentum-encoder global vectors in,
        overwriting the oldest entries (circular buffer)."""
        batch_size = keys.shape[0]
        if batch_size > self.size:
            keys = keys[-self.size:]
            batch_size = keys.shape[0]

        ptr = int(self.pointer.item())
        end = ptr + batch_size
        if end <= self.size:
            self.vectors[ptr:end] = keys
        else:
            first_chunk = self.size - ptr
            self.vectors[ptr:] = keys[:first_chunk]
            self.vectors[: end - self.size] = keys[first_chunk:]
        self.pointer[0] = end % self.size

    def get(self) -> torch.Tensor:
        """Current queue contents, `(size, dim)`. Cloned defensively so a
        later `enqueue()` can't mutate a tensor a caller is still using."""
        return self.vectors.clone()


if __name__ == "__main__":
    queue = MemoryQueue(dim=8, size=10)
    print(f"Initial queue shape: {tuple(queue.get().shape)}")

    batch1 = F.normalize(torch.ones(4, 8), dim=1)
    queue.enqueue(batch1)
    print(f"After enqueueing 4: pointer={int(queue.pointer.item())}")
    print(f"Rows 0-3 match batch1: {torch.allclose(queue.get()[:4], batch1)}")

    batch2 = F.normalize(torch.full((8, 8), 2.0), dim=1)
    queue.enqueue(batch2)  # 4 + 8 = 12 > size 10 -> must wrap around
    print(f"After enqueueing 8 more (size=10, wraps around): pointer={int(queue.pointer.item())}")
    contents = queue.get()
    print(f"Rows 4-9 hold the first 6 of batch2: {torch.allclose(contents[4:10], batch2[:6])}")
    print(f"Rows 0-1 hold the wrapped remainder of batch2: {torch.allclose(contents[0:2], batch2[6:8])}")
