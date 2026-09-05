"""Step 4: the combined distance - global (Method A) + structural (Method
B, now with a trainable local projection), per DetailSemNet's Eq. 2:

    dis = lambda_0 * dis_global + dis_struct

A raw, unnormalized sum (lambda_0 = 1.0 default) - NOT the scale-normalized
alpha-blend `blended_distance.py` implements. That module belongs to the
retired old design (`downstream_supervised_learning_approach.md` SS5) and
is not reused here; SS6/SS10 of that doc are why the raw sum is the
correct formula to replicate.

Batched: computes the combined distance for a whole batch of pairs in one
call, built on `dense_matching.py`'s batched Sinkhorn primitives - the
grid stays at its NATIVE resolution (no coarsening - see the roadmap
doc's SS9 for why that was deliberately rejected as a preemptive default).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

MATCHING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = MATCHING_DIR.parent
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))
sys.path.insert(0, str(MATCHING_DIR))

from mask import compute_foreground_mask  # noqa: E402
from dense_matching import batched_dense_matching_distance  # noqa: E402


def compute_masks_from_tensor_batch(image_batch: torch.Tensor) -> torch.Tensor:
    """`(B, 1, H, W)` binary `{0,1}` image batch -> `(B, grid, grid)` bool
    foreground mask batch. Mask computation is lightweight per-image numpy
    work (not a GPU bottleneck, unlike Sinkhorn), so a plain Python loop
    is fine here - only `batched_dense_matching_distance` needs the real
    batching."""
    masks = []
    for image in image_batch:
        image_np = image.squeeze().detach().cpu().numpy()
        mask_np, _ = compute_foreground_mask(image_np)
        masks.append(torch.from_numpy(mask_np))
    return torch.stack(masks).to(image_batch.device)


def combined_distance_batched(
    model: nn.Module,
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> torch.Tensor:
    """`(B, 1, H, W)` x2 -> `(B,)` combined distances, fully
    differentiable through the global projector, `model.local_projection`,
    and the encoder (whichever stages are unfrozen) - one shared encoder
    forward pass per image feeds both branches, the same "one computation,
    two read-outs" relationship `forward_global`/`forward_dense` have
    always had.

    `model` must expose `forward_global`, `forward_dense`, and a non-None
    `local_projection` submodule (i.e. built with `local_embedding_dim`
    set - see `embedding_model.py`).
    """
    if model.local_projection is None:
        raise ValueError(
            "model.local_projection is None - build the model with "
            "local_embedding_dim set to use the combined distance."
        )

    global_a = model.forward_global(images_a)
    global_b = model.forward_global(images_b)
    dis_global = torch.norm(global_a - global_b, p=2, dim=1)

    dense_a = model.forward_dense(images_a)
    dense_b = model.forward_dense(images_b)
    masks_a = compute_masks_from_tensor_batch(images_a)
    masks_b = compute_masks_from_tensor_batch(images_b)

    dis_struct = batched_dense_matching_distance(
        dense_a, dense_b, masks_a, masks_b, model.local_projection,
        epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations,
    )

    return lambda_0 * dis_global + dis_struct
