"""Step 3: blend Method A (global) and Method B (dense matching) into one
overall distance, per `docs/claude_response/downstream_supervised_learning_approach.md`:

    overall_distance = (1 - alpha) * Method A + alpha * Method B

`alpha=0` reproduces the existing pipeline exactly (pure Method A);
`alpha=1` is pure Method B; values in between blend the two.

One subtlety that isn't just plumbing: Method A's raw distance (Euclidean
distance between two L2-normalized global embeddings) and Method B's raw
distance (Sinkhorn-matched cosine cost) do not naturally live on the same
scale - their typical magnitudes differ, and if the raw numbers were
blended directly, `alpha` would not mean "how much weight on each method"
in any consistent sense (e.g. alpha=0.5 would not mean "equal
contribution" if one method's raw numbers are, say, three times the size
of the other's on average). This module fixes that by dividing each raw
distance by a fixed scale constant (each method's own mean raw distance,
estimated once over a representative sample of TRAINING-writer pairs)
before blending, so `alpha` has a stable, comparable meaning across its
whole sweep (Step 5's whole point depends on this).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MATCHING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = MATCHING_DIR.parent
sys.path.insert(0, str(MATCHING_DIR))
sys.path.insert(0, str(SUPERVISED_DIR / "models"))
sys.path.insert(0, str(SUPERVISED_DIR / "datasets"))

from dense_matching import compute_mask_from_tensor, dense_matching_distance  # noqa: E402
from dual_triplet_dataset import DualTripletDataset  # noqa: E402
from embedding_model import DownstreamVerificationModel  # noqa: E402


@dataclass(frozen=True)
class DistanceScales:
    method_a_scale: float
    method_b_scale: float


def _ensure_batched(image_tensor: torch.Tensor) -> torch.Tensor:
    return image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor


def method_a_distance(
    model: DownstreamVerificationModel,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> torch.Tensor:
    """Pure Method A: Euclidean distance between two L2-normalized global
    embeddings. No scale normalization applied - this is the raw distance;
    `blended_distance` is what normalizes and weights it. Used standalone
    for Step 5's Method-A-only evaluation cell and for scale estimation."""
    image_a = _ensure_batched(image_a)
    image_b = _ensure_batched(image_b)
    global_a = model.forward_global(image_a).squeeze(0)
    global_b = model.forward_global(image_b).squeeze(0)
    return torch.norm(global_a - global_b, p=2)


def method_b_distance(
    model: DownstreamVerificationModel,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> torch.Tensor:
    """Pure Method B: the dense piece-by-piece matching distance. No scale
    normalization applied - see `method_a_distance`. Used standalone for
    Step 5's Method-B-only evaluation cell and for scale estimation."""
    image_a = _ensure_batched(image_a)
    image_b = _ensure_batched(image_b)
    dense_a = model.forward_dense(image_a).squeeze(0)
    dense_b = model.forward_dense(image_b).squeeze(0)
    mask_a = compute_mask_from_tensor(image_a)
    mask_b = compute_mask_from_tensor(image_b)
    return dense_matching_distance(
        dense_a, dense_b, mask_a, mask_b,
        epsilon=sinkhorn_epsilon, num_iterations=sinkhorn_iterations,
    )


def compute_raw_distances(
    model: DownstreamVerificationModel,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both methods' raw (unnormalized) distance for one signature pair.
    Always computes both, regardless of any `alpha` - used for scale
    estimation, where every pair's contribution to both methods' typical
    scale is needed no matter what `alpha` will later be swept over."""
    raw_a = method_a_distance(model, image_a, image_b)
    raw_b = method_b_distance(model, image_a, image_b, sinkhorn_epsilon, sinkhorn_iterations)
    return raw_a, raw_b


def estimate_distance_scales(
    model: DownstreamVerificationModel,
    training_dataset: DualTripletDataset,
    device: torch.device,
    num_quadruples: int = 200,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
    seed: int = 42,
) -> DistanceScales:
    """Estimate each method's fixed scale constant from a sample of
    TRAINING-writer quadruples (`training_dataset` must be built on
    training writers - see caveat below). For every pair inside each
    sampled quadruple (anchor-positive, anchor-negative_intra,
    anchor-negative_inter - the full mix of pair types the blended
    distance will actually be asked to score during training), computes
    both methods' raw distance. The scale is each method's mean raw
    distance across all those pairs.

    Caveat: unlike every other hyperparameter in this pipeline (alpha
    itself, the Sinkhorn epsilon, checkpoint selection, the tau
    threshold - all tuned on validation writers only), these scale
    constants are computed from TRAINING writers. This is a deliberate,
    different choice: they are a data-characterization statistic (how big
    are this encoder's distances, typically), analogous to computing a
    dataset's pixel mean/std for input normalization, not a
    model-selection decision - they never touch a writer label or a
    genuine/forgery distinction in a way that could leak. Using training
    data for this is standard practice, the same way ImageNet's own
    per-channel mean/std are computed from its training split.
    """
    training_dataset.set_epoch(0)
    rng = random.Random(seed)
    indices = rng.sample(range(len(training_dataset)), min(num_quadruples, len(training_dataset)))

    method_a_distances: list[float] = []
    method_b_distances: list[float] = []

    model.eval()
    with torch.no_grad():
        for index in indices:
            item = training_dataset[index]
            anchor = item["anchor"].to(device)
            others = (
                item["positive"].to(device),
                item["negative_intra"].to(device),
                item["negative_inter"].to(device),
            )
            for other in others:
                raw_a, raw_b = compute_raw_distances(
                    model, anchor, other,
                    sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iterations=sinkhorn_iterations,
                )
                method_a_distances.append(raw_a.item())
                method_b_distances.append(raw_b.item())

    return DistanceScales(
        method_a_scale=float(np.mean(method_a_distances)),
        method_b_scale=float(np.mean(method_b_distances)),
    )


def blended_distance(
    model: DownstreamVerificationModel,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    alpha: float,
    scales: DistanceScales,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 50,
) -> torch.Tensor:
    """The Step 3 blend: `(1 - alpha) * normalized Method A + alpha *
    normalized Method B`. `alpha=0` skips Method B's matching entirely
    (Method B is ~100-1000x more expensive per pair than Method A - see
    the roadmap doc - so this matters in practice, not just in theory,
    for Step 5's 4-way comparison, whose Method-A-only cells should not
    pay Method B's cost); `alpha=1` skips Method A likewise. Fully
    differentiable at any `0 <= alpha <= 1` (gradients flow through
    whichever method(s) are actually computed)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    method_a_component: torch.Tensor | None = None
    method_b_component: torch.Tensor | None = None

    if alpha < 1.0:
        raw_a = method_a_distance(model, image_a, image_b)
        method_a_component = (1.0 - alpha) * (raw_a / scales.method_a_scale)

    if alpha > 0.0:
        raw_b = method_b_distance(model, image_a, image_b, sinkhorn_epsilon, sinkhorn_iterations)
        method_b_component = alpha * (raw_b / scales.method_b_scale)

    if method_a_component is None:
        return method_b_component
    if method_b_component is None:
        return method_a_component
    return method_a_component + method_b_component
