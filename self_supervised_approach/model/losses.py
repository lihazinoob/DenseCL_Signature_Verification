"""Contrastive losses for DenseCL pretraining.

Two losses, combined: global InfoNCE (whole-image instance discrimination)
and dense foreground-masked correspondence (patch-level instance
discrimination). See `progress_so_far.md` Section 2 for how these plug into
the rest of the architecture, and the earlier chat discussion of "what is
the loss function" / "what is the patch matching technique" for the
reasoning behind the design choices below - this module implements exactly
that discussion, not a different design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    """Tunable loss hyperparameters. Defaults are standard MoCo/DenseCL
    starting points, not final values - meant to be swept against
    downstream validation AUC later (lambda_dense especially - see
    `progress_so_far.md` Section 5, "lambda_dense sweep"), the same way
    `AugmentConfig` and `min_coverage` are treated."""

    temperature_global: float = 0.2
    temperature_dense: float = 0.2
    lambda_global: float = 1.0
    lambda_dense: float = 1.0


DEFAULT_LOSS_CONFIG = LossConfig()


class GlobalInfoNCELoss(nn.Module):
    """Whole-image instance discrimination: pull View A's and View B's
    global vectors together, push apart from every vector in the memory
    queue (see `memory_queue.py`).

    Standard MoCo formulation: a `(1 + queue_size)`-way classification
    problem where the true positive pair is always placed at index 0, and
    cross-entropy is used to score how confidently the model picks it out
    against every queue negative.
    """

    def __init__(self, temperature: float = DEFAULT_LOSS_CONFIG.temperature_global) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """z_a: (batch, dim) online View A vectors (query, gradients flow).
        z_b: (batch, dim) momentum View B vectors (positive key, no grad).
        queue: (queue_size, dim) momentum vectors from past batches
        (negative keys, no grad). All rows are assumed L2-normalized
        already (both `GlobalHead` and `MemoryQueue` guarantee this), so
        dot products here are cosine similarities."""
        pos_logits = (z_a * z_b).sum(dim=1, keepdim=True) / self.temperature       # (batch, 1)
        neg_logits = z_a @ queue.T / self.temperature                               # (batch, queue_size)
        logits = torch.cat([pos_logits, neg_logits], dim=1)                         # (batch, 1+queue_size)
        labels = torch.zeros(z_a.shape[0], dtype=torch.long, device=z_a.device)     # positive is always index 0
        return F.cross_entropy(logits, labels)


class DenseCorrespondenceLoss(nn.Module):
    """Patch-level instance discrimination, restricted to ink-bearing
    (foreground) grid cells only.

    For each foreground patch in View A, find its nearest neighbor (highest
    cosine similarity) among View B's foreground patches OF THE SAME
    signature - that becomes the positive pair (matching never crosses
    images: two different signatures' patches have no correspondence to
    find in the first place). Negatives are every foreground patch in View
    B across the WHOLE batch.

    Deliberate simplification vs. the original DenseCL paper: negatives
    come from an in-batch pool, not a separate dense memory queue. A dense
    queue would need one slot per ink-bearing grid cell rather than one per
    image - tens of thousands of slots instead of a few thousand - which is
    real added complexity for a first working version. Upgradeable later if
    the in-batch negative pool proves too small in practice.

    Matching runs per-sample in a plain Python loop over the batch rather
    than a single fully-vectorized op - clearer to read and get right, and
    batch sizes here are small enough (tens, not thousands) that this is
    not a bottleneck.
    """

    def __init__(self, temperature: float = DEFAULT_LOSS_CONFIG.temperature_dense) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """z_a/z_b: (batch, dim, H, W) online/momentum dense projected
        features (L2-normalized per cell, from `DenseHead`). mask_a/mask_b:
        (batch, H, W) bool foreground masks at the same H, W (from
        `mask.py`'s `compute_foreground_mask`, computed per-view - see the
        earlier chat explanation of why masking must be recomputed per
        augmented view rather than reused from the original image)."""
        batch_size, dim, h, w = z_a.shape

        feat_a = z_a.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)
        feat_b = z_b.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)
        flat_mask_a = mask_a.reshape(batch_size, h * w)
        flat_mask_b = mask_b.reshape(batch_size, h * w)

        # Negative pool: every foreground patch in View B, across the whole batch.
        all_keys = feat_b[flat_mask_b]  # (total_fg_in_batch, dim)
        if all_keys.shape[0] == 0:
            return z_a.new_zeros(())  # degenerate: no ink anywhere in the batch's View B

        per_sample_losses = []
        for n in range(batch_size):
            query_idx = flat_mask_a[n].nonzero(as_tuple=True)[0]
            key_idx = flat_mask_b[n].nonzero(as_tuple=True)[0]
            if query_idx.numel() == 0 or key_idx.numel() == 0:
                continue  # this view has no ink (shouldn't normally happen post-preprocessing)

            queries = feat_a[n, query_idx]           # (Nq, dim)
            keys_same_sample = feat_b[n, key_idx]    # (Nk, dim)

            # Nearest-neighbor matching is a hard index selection with no
            # gradient of its own - compute it under no_grad so we don't
            # build an autograd graph for a similarity matrix we only ever
            # call argmax() on.
            with torch.no_grad():
                match_similarity = queries @ keys_same_sample.T   # (Nq, Nk)
                best_match = match_similarity.argmax(dim=1)        # (Nq,)
            positives = keys_same_sample[best_match]                # (Nq, dim)

            pos_logits = (queries * positives).sum(dim=1, keepdim=True) / self.temperature
            neg_logits = queries @ all_keys.T / self.temperature
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
            per_sample_losses.append(F.cross_entropy(logits, labels))

        if not per_sample_losses:
            return z_a.new_zeros(())
        return torch.stack(per_sample_losses).mean()


class DenseCLLoss(nn.Module):
    """Combines both losses: total = lambda_global * L_global + lambda_dense * L_dense."""

    def __init__(self, config: LossConfig = DEFAULT_LOSS_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.global_loss = GlobalInfoNCELoss(temperature=config.temperature_global)
        self.dense_loss = DenseCorrespondenceLoss(temperature=config.temperature_dense)

    def forward(
        self,
        z_a_global: torch.Tensor,
        z_b_global: torch.Tensor,
        queue: torch.Tensor,
        z_a_dense: torch.Tensor,
        z_b_dense: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_global = self.global_loss(z_a_global, z_b_global, queue)
        loss_dense = self.dense_loss(z_a_dense, z_b_dense, mask_a, mask_b)
        total = self.config.lambda_global * loss_global + self.config.lambda_dense * loss_dense
        return {"total": total, "global": loss_global, "dense": loss_dense}


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Global loss: behavioral sanity check (not just shapes) ----
    # A loss that "works" should score a true positive pair as clearly more
    # likely than an unrelated pair, given the same random negatives.
    batch_size, dim, queue_size = 4, 128, 100
    global_loss_fn = GlobalInfoNCELoss(temperature=0.2)
    queue_vectors = F.normalize(torch.randn(queue_size, dim), dim=1)

    z_a = F.normalize(torch.randn(batch_size, dim), dim=1)
    z_b_matched = z_a.clone()
    z_b_random = F.normalize(torch.randn(batch_size, dim), dim=1)

    loss_matched = global_loss_fn(z_a, z_b_matched, queue_vectors)
    loss_random = global_loss_fn(z_a, z_b_random, queue_vectors)
    print(f"Global loss (true positive pair):  {loss_matched.item():.4f} (expect low)")
    print(f"Global loss (unrelated pair):       {loss_random.item():.4f} (expect higher)")
    assert loss_matched.item() < loss_random.item(), "Global loss must reward true positives over random pairs"

    # ---- Dense loss: same kind of behavioral check ----
    grid = 32
    dense_loss_fn = DenseCorrespondenceLoss(temperature=0.2)
    z_a_dense = F.normalize(torch.randn(batch_size, dim, grid, grid), dim=1)
    z_b_dense_matched = z_a_dense.clone()
    z_b_dense_random = F.normalize(torch.randn(batch_size, dim, grid, grid), dim=1)

    mask_a = torch.rand(batch_size, grid, grid) < 0.15  # ~matches the ~12% ink density measured on real data
    mask_b = torch.rand(batch_size, grid, grid) < 0.15
    mask_a[:, 0, 0] = True  # guarantee non-degenerate (at least one fg cell) in this synthetic test
    mask_b[:, 0, 0] = True

    loss_dense_matched = dense_loss_fn(z_a_dense, z_b_dense_matched, mask_a, mask_a)
    loss_dense_random = dense_loss_fn(z_a_dense, z_b_dense_random, mask_a, mask_b)
    print(f"Dense loss (identical view, same mask): {loss_dense_matched.item():.4f} (expect low)")
    print(f"Dense loss (unrelated views/masks):     {loss_dense_random.item():.4f} (expect higher)")
    assert loss_dense_matched.item() < loss_dense_random.item(), "Dense loss must reward true correspondences"

    # ---- Full pipeline, real data, real backward pass ----
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
    from torch.utils.data import DataLoader

    from dataset import SignatureSSLDataset
    from encoder import Encoder
    from heads import DenseHead, GlobalHead
    from memory_queue import MemoryQueue
    from momentum import EMAModule, MomentumEncoder

    DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "all"
    ds = SignatureSSLDataset(DATA_ROOT)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    real_batch = next(iter(loader))

    online_encoder = Encoder()
    momentum_encoder = MomentumEncoder(online_encoder)
    global_head = GlobalHead()
    dense_head = DenseHead()
    momentum_global_head = EMAModule(global_head)
    momentum_dense_head = EMAModule(dense_head)
    memory_queue = MemoryQueue()
    dense_cl_loss = DenseCLLoss()

    z_a_global = global_head(online_encoder(real_batch["view_a"], pool=True))
    z_a_dense = dense_head(online_encoder(real_batch["view_a"], pool=False))
    z_b_global = momentum_global_head(momentum_encoder(real_batch["view_b"], pool=True))
    z_b_dense = momentum_dense_head(momentum_encoder(real_batch["view_b"], pool=False))

    losses = dense_cl_loss(
        z_a_global, z_b_global, memory_queue.get(),
        z_a_dense, z_b_dense, real_batch["mask_a"], real_batch["mask_b"],
    )
    print(f"\nReal-data losses: total={losses['total'].item():.4f}, "
          f"global={losses['global'].item():.4f}, dense={losses['dense'].item():.4f}")

    losses["total"].backward()
    grad = online_encoder.stem.layers[0].block[0].weight.grad
    print(f"backward() ok, gradient reached the encoder's first layer: {grad is not None and grad.abs().sum().item() > 0}")

    # Only AFTER computing the loss: enqueue this batch's momentum keys for future steps.
    memory_queue.enqueue(z_b_global.detach())
    print(f"Queue pointer after one enqueue: {int(memory_queue.pointer.item())}")
