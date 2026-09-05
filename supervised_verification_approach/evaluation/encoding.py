"""Step 5, part 1: cache every signature's representation once.

Per `docs/claude_response/downstream_supervised_learning_approach.md`
("Inference and validation protocol", step 1): run every evaluation
signature through the frozen encoder once and cache the result, keyed by
image path - reused across every reference draw and every query. This
matters more for Method B than Method A: within one reference draw, the
same K references get compared against every one of a writer's ~40+
queries, so recomputing the dense feature grid + foreground mask (and
running a fresh Sinkhorn solve) for the same reference image over and
over would be pure waste.

`EncodedSignature` deliberately caches BOTH representations (global
embedding AND dense grid + mask) regardless of which method(s) will
actually be scored - computing the unused half is cheap (the model has
already done the shared backbone forward pass by the time `forward_dense`
is called; `forward_global` just adds one pooling + linear layer on top),
and it keeps one cache usable for Method A, Method B, or the blend without
re-encoding.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

EVALUATION_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = EVALUATION_DIR.parent  # DenseCL_approach/supervised_verification_approach
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent  # DenseCL_approach
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))
sys.path.insert(0, str(SUPERVISED_DIR / "models"))
sys.path.insert(0, str(SUPERVISED_DIR / "matching"))

from preprocess import preprocess_signature  # noqa: E402
from mask import compute_foreground_mask  # noqa: E402
from embedding_model import DownstreamVerificationModel  # noqa: E402
from dense_matching import (  # noqa: E402
    batched_dense_matching_distance,
    dense_matching_distance,
    dense_matching_distance_projected,
)


@dataclass(frozen=True)
class EncodedSignature:
    global_embedding: torch.Tensor  # (embedding_dim,) - Method A
    dense_features: torch.Tensor    # (grid, grid, feature_dim) - Method B
    foreground_mask: torch.Tensor   # (grid, grid) bool - Method B


@torch.no_grad()
def encode_signature(model: DownstreamVerificationModel, image_path: Path, device: torch.device) -> EncodedSignature:
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    view = preprocess_signature(raw)
    mask_np, _ = compute_foreground_mask(view)

    tensor = torch.from_numpy((view > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    global_embedding = model.forward_global(tensor).squeeze(0)
    dense_features = model.forward_dense(tensor).squeeze(0)

    return EncodedSignature(
        global_embedding=global_embedding,
        dense_features=dense_features,
        foreground_mask=torch.from_numpy(mask_np).to(device),
    )


def encode_all_signatures(
    model: DownstreamVerificationModel,
    image_paths: list[Path],
    device: torch.device,
    show_progress: bool = True,
) -> dict[Path, EncodedSignature]:
    """De-duplicates `image_paths` before encoding (a reference signature
    in one writer's set never needs encoding twice, and callers building
    the path list from multiple reference draws may naturally repeat
    paths)."""
    model.eval()
    unique_paths = sorted(set(image_paths))
    iterator = tqdm(unique_paths, desc="Encoding signatures", ncols=100) if show_progress else unique_paths
    return {path: encode_signature(model, path, device) for path in iterator}


def method_a_distance_cached(encoded_a: EncodedSignature, encoded_b: EncodedSignature) -> float:
    """Pure Method A distance between two already-encoded signatures."""
    return torch.norm(encoded_a.global_embedding - encoded_b.global_embedding, p=2).item()


def method_b_distance_cached(
    encoded_a: EncodedSignature,
    encoded_b: EncodedSignature,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> float:
    """Pure Method B distance between two already-encoded signatures -
    reuses `dense_matching.dense_matching_distance` directly on the cached
    dense grids/masks, no re-encoding."""
    return dense_matching_distance(
        encoded_a.dense_features, encoded_b.dense_features,
        encoded_a.foreground_mask, encoded_b.foreground_mask,
        epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations,
    ).item()


def blended_distance_cached(
    encoded_a: EncodedSignature,
    encoded_b: EncodedSignature,
    alpha: float,
    method_a_scale: float,
    method_b_scale: float,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> float:
    """Step 3's blend, on already-encoded signatures. `alpha=0`/`alpha=1`
    skip the unused method's (expensive, for Method B) computation, same
    as `blended_distance` during training."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    component_a = 0.0
    component_b = 0.0
    if alpha < 1.0:
        component_a = (1.0 - alpha) * (method_a_distance_cached(encoded_a, encoded_b) / method_a_scale)
    if alpha > 0.0:
        component_b = alpha * (
            method_b_distance_cached(encoded_a, encoded_b, sinkhorn_epsilon, sinkhorn_iterations) / method_b_scale
        )
    return component_a + component_b


def combined_distance_cached(
    encoded_a: EncodedSignature,
    encoded_b: EncodedSignature,
    local_projection: torch.nn.Module,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> float:
    """Step 4's combined distance (`lambda_0 * dis_global + dis_struct`,
    DetailSemNet Eq. 2 - see `matching/combined_distance.py`) on two
    already-encoded signatures. `local_projection` must be passed in
    explicitly (`model.local_projection`) since `EncodedSignature` only
    caches raw, unprojected dense features - the projection's weights
    change every epoch, so they can never be baked into the cache, only
    the raw features that feed them."""
    dis_global = method_a_distance_cached(encoded_a, encoded_b)
    dis_struct = dense_matching_distance_projected(
        encoded_a.dense_features, encoded_b.dense_features,
        encoded_a.foreground_mask, encoded_b.foreground_mask,
        local_projection, epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations,
    ).item()
    return lambda_0 * dis_global + dis_struct


def combined_distance_batch_cached(
    pairs: list[tuple[EncodedSignature, EncodedSignature]],
    local_projection: torch.nn.Module,
    lambda_0: float = 1.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> list[float]:
    """Batched sibling of `combined_distance_cached` over MANY (a, b) pairs
    at once - the fix for the verification protocol's dominant per-epoch
    cost. `score_writer`/`run_verification_draw` used to call the
    single-pair wrapper (`dense_matching_distance_projected`, batch size 1)
    once per (query, reference) comparison - thousands of times per draw -
    paying full GPU kernel-launch/sync overhead per pair instead of
    amortizing it across a batch the way training's `run_one_epoch_combined`
    already does. This reuses the exact same trusted batched primitive
    (`batched_dense_matching_distance`) `dense_matching_distance_projected`
    wraps, just called once with `len(pairs)` items instead of `len(pairs)`
    times with 1 item - identical math, identical numbers, far fewer kernel
    launches. `torch.no_grad()`: this is an evaluation-only path (no
    backward ever runs on these distances), so building an autograd graph
    through `local_projection` here would be pure waste."""
    if not pairs:
        return []

    with torch.no_grad():
        embeddings_a = torch.stack([a.global_embedding for a, _ in pairs])
        embeddings_b = torch.stack([b.global_embedding for _, b in pairs])
        dis_global = torch.norm(embeddings_a - embeddings_b, p=2, dim=1)

        dense_a = torch.stack([a.dense_features for a, _ in pairs])
        dense_b = torch.stack([b.dense_features for _, b in pairs])
        masks_a = torch.stack([a.foreground_mask for a, _ in pairs])
        masks_b = torch.stack([b.foreground_mask for _, b in pairs])

        dis_struct = batched_dense_matching_distance(
            dense_a, dense_b, masks_a, masks_b, local_projection,
            epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations,
        )
        dis_combined = lambda_0 * dis_global + dis_struct

    return dis_combined.tolist()
