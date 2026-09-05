"""Method B: piece-by-piece dense matching distance between two signatures.

See `docs/claude_response/downstream_supervised_learning_approach.md`
("Step 2", and the plain-language walkthrough of why prototypes don't
exist for Method B) for the full reasoning. In short: `embedding_model.
forward_dense()` turns a signature into a pile of small feature vectors,
one per ink-containing grid cell - not one vector, and the pile size
varies signature to signature. Ordinary Euclidean/cosine distance needs
two same-length vectors, so it cannot compare two piles directly. This
module implements the alternative: build a piece-to-piece cost table
between the two piles, find the cheapest overall way to pair every piece
up (entropic-regularized optimal transport, solved via the Sinkhorn
algorithm), and sum the matched costs into one scalar distance - the same
mechanism DetailSemNet (the current supervised state-of-the-art for this
task) uses on its own patch tokens.

Shape/masking conventions match `correspondence.py`'s mutual-NN matcher
(the existing qualitative tool in `self_supervised_approach/analyzer/`):
`(grid_size, grid_size, feature_dim)` dense feature grids and
`(grid_size, grid_size)` boolean foreground masks. Unlike that module
(numpy, non-differentiable argmax matching, built for visualization), this
one is pure PyTorch and fully differentiable - every op is
logsumexp/exp/subtraction, so gradients can flow back into the encoder if
it's ever fine-tuned (Step 7), the same way Method A's Euclidean distance
already does. It has no trainable parameters of its own - `epsilon` and
`num_iterations` are fixed hyperparameters, swept on validation writers
only (see the roadmap doc's open decisions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MATCHING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = MATCHING_DIR.parent  # DenseCL_approach/supervised_verification_approach
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent  # DenseCL_approach
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "utils"))

from mask import compute_foreground_mask  # noqa: E402


def compute_mask_from_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """`(1, H, W)` or `(H, W)` binary `{0,1}` image tensor -> `(grid, grid)`
    bool foreground mask, on the same device. Reuses `mask.py`'s
    `compute_foreground_mask` directly - it only ever tests `ink > 0`, so
    it works identically whether the binary image is scaled `{0,1}` (this
    project's tensor convention) or `{0,255}` (the raw preprocessed
    numpy array `mask.py` was originally written against)."""
    image_np = image_tensor.squeeze().detach().cpu().numpy()
    mask_np, _ = compute_foreground_mask(image_np)
    return torch.from_numpy(mask_np).to(image_tensor.device)


def extract_foreground_pieces(dense_features: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """`(grid, grid, dim)` dense feature grid + `(grid, grid)` bool
    foreground mask -> `(num_ink_pieces, dim)`. Discards background cells
    before any comparison happens - a signature's grid is ~88% blank
    paper, so unrestricted matching would mostly compare blank-to-blank."""
    if dense_features.shape[:2] != foreground_mask.shape:
        raise ValueError(
            f"mask shape must match feature grid's spatial shape: "
            f"features={tuple(dense_features.shape[:2])}, mask={tuple(foreground_mask.shape)}"
        )
    return dense_features[foreground_mask]


def cosine_cost_matrix(pieces_a: torch.Tensor, pieces_b: torch.Tensor) -> torch.Tensor:
    """`(n, dim)` x `(m, dim)` -> `(n, m)` cosine DISTANCE (1 - cosine
    similarity) ground cost - the same ground distance DetailSemNet uses
    for its own patch-token matching."""
    a_norm = F.normalize(pieces_a, p=2, dim=1)
    b_norm = F.normalize(pieces_b, p=2, dim=1)
    similarity = a_norm @ b_norm.T
    return 1.0 - similarity


def sinkhorn_ot_distance(cost: torch.Tensor, epsilon: float = 0.05, num_iterations: int = 50) -> torch.Tensor:
    """Entropic-regularized optimal transport distance between two piles of
    pieces, given their `(n, m)` pairwise ground cost (`cost[i, j]` = how
    different query-piece i is from reference-piece j). Uniform marginals
    over both piles (every piece counted equally - no piece is considered
    more or less important a priori). Log-domain, numerically stable
    Sinkhorn iteration (Cuturi 2013 / Peyre & Cuturi's stabilized
    formulation): alternately solve for each pile's dual potential (`f`/
    `g`) holding the other fixed, then read the transport plan and its
    total cost off the converged potentials.
    """
    n, m = cost.shape
    log_a = -torch.log(torch.tensor(float(n), device=cost.device, dtype=cost.dtype))
    log_b = -torch.log(torch.tensor(float(m), device=cost.device, dtype=cost.dtype))

    f = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(m, device=cost.device, dtype=cost.dtype)

    for _ in range(num_iterations):
        f = epsilon * log_a - epsilon * torch.logsumexp((g.unsqueeze(0) - cost) / epsilon, dim=1)
        g = epsilon * log_b - epsilon * torch.logsumexp((f.unsqueeze(1) - cost) / epsilon, dim=0)

    transport_plan = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon)
    return (transport_plan * cost).sum()


def dense_matching_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    epsilon: float = 0.05,
    num_iterations: int = 50,
) -> torch.Tensor:
    """The full Method B distance between two signatures.

    `(grid, grid, dim)` dense feature grids + `(grid, grid)` boolean
    foreground masks -> one scalar distance. Steps: mask each grid down to
    its ink-only pieces (`extract_foreground_pieces`), build the
    piece-to-piece cost table (`cosine_cost_matrix`), find the cheapest
    overall pairing and sum it (`sinkhorn_ot_distance`). The two piles do
    not need to be the same size - that's the whole point of routing
    through a matching step instead of a direct vector distance.

    Kept unchanged and untouched by Step 4's batched/projected additions
    below - this is still Method B's original, un-projected, one-pair-at-
    a-time path, used by existing regression/diagnostic call sites.
    """
    pieces_a = extract_foreground_pieces(features_a, mask_a)
    pieces_b = extract_foreground_pieces(features_b, mask_b)

    if pieces_a.shape[0] == 0 or pieces_b.shape[0] == 0:
        raise ValueError(
            f"Cannot compute a matching distance with zero foreground pieces "
            f"(found {pieces_a.shape[0]} in A, {pieces_b.shape[0]} in B)."
        )

    cost = cosine_cost_matrix(pieces_a, pieces_b)
    return sinkhorn_ot_distance(cost, epsilon=epsilon, num_iterations=num_iterations)


# ---------------------------------------------------------------------------
# Step 4: batched Sinkhorn + trainable local projection.
#
# Everything above this line solves ONE pair's transport problem per call,
# in a Python-level loop when driven over a batch (see
# `run_one_epoch_blended.py`'s docstring, which already flags this as the
# dominant training-time cost). The functions below solve a whole BATCH of
# pairs' transport problems in one vectorized set of tensor ops instead -
# required for Step 4 training to be feasible (roadmap doc SS9).
#
# Ragged foreground piles (a different number of ink cells per signature)
# are handled by padding every pile in the batch up to the batch's own
# max length and tracking which entries are real with a boolean `valid`
# mask, then giving padded (invalid) rows/columns -inf marginal log-mass
# in the Sinkhorn iteration - the same masking technique transformer
# attention uses for padded tokens. This is provably safe here (no NaN):
# every reduction always has at least one valid entry (`_extract_and_pad`
# raises if any image has zero foreground pieces), so no logsumexp ever
# reduces over an all -inf axis, and -inf only ever appears as an ADDEND
# inside an `exp(...)`, never combined with another -inf or with +inf -
# `exp(-inf) = 0` cleanly, both in the forward pass and its local
# gradient, so it never produces the `0 * inf = NaN` pattern that literal
# -inf masking can hit in less careful implementations.
# ---------------------------------------------------------------------------


def _extract_and_pad(dense_features_batch: torch.Tensor, masks_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, grid, grid, dim)` dense feature grids + `(B, grid, grid)` bool
    masks -> padded pieces `(B, N_max, dim)` + `(B, N_max)` bool valid
    mask, where `N_max` is the largest ink-cell count in this batch. The
    ragged extraction itself (`dense_features_batch[i][masks_batch[i]]`)
    is inherently per-image (boolean-mask indexing can't be vectorized
    across a batch) but is cheap - only the Sinkhorn iteration below is
    the expensive part, and that IS fully batched."""
    batch_size, _, _, dim = dense_features_batch.shape
    pieces_list = [dense_features_batch[i][masks_batch[i]] for i in range(batch_size)]
    counts = [p.shape[0] for p in pieces_list]
    if min(counts) == 0:
        raise ValueError(f"Every item needs at least one foreground piece (counts={counts}).")

    max_count = max(counts)
    padded = dense_features_batch.new_zeros(batch_size, max_count, dim)
    valid = torch.zeros(batch_size, max_count, dtype=torch.bool, device=dense_features_batch.device)
    for i, pieces in enumerate(pieces_list):
        n = pieces.shape[0]
        padded[i, :n] = pieces
        valid[i, :n] = True
    return padded, valid


def batched_cosine_cost_matrix(pieces_a: torch.Tensor, pieces_b: torch.Tensor) -> torch.Tensor:
    """`(B, N, dim)` x `(B, M, dim)` -> `(B, N, M)` cosine distance,
    batched via `bmm`. Padded rows/columns get some finite (harmless)
    cost value here - they are excluded not by their cost, but by the
    -inf marginal mass `batched_sinkhorn_ot_distance` gives them."""
    a_norm = F.normalize(pieces_a, p=2, dim=-1)
    b_norm = F.normalize(pieces_b, p=2, dim=-1)
    similarity = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return 1.0 - similarity


def batched_sinkhorn_ot_distance(
    cost: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    epsilon: float = 0.05,
    num_iterations: int = 50,
) -> torch.Tensor:
    """`(B, N, M)` cost + `(B, N)`/`(B, M)` valid masks -> `(B,)`
    distances. Same log-domain Sinkhorn iteration as `sinkhorn_ot_distance`,
    vectorized across the batch dimension; uniform marginals over each
    pair's OWN valid piece count (not the padded `N`/`M`), everything
    else (padding) held at -inf log-mass throughout - see the module-level
    note above for why this is numerically safe."""
    batch_size, n, m = cost.shape

    count_a = valid_a.sum(dim=1, keepdim=True).float()  # (B, 1)
    count_b = valid_b.sum(dim=1, keepdim=True).float()
    if (count_a < 1).any() or (count_b < 1).any():
        raise ValueError("Every item in the batch must have at least one foreground piece.")

    neg_inf = torch.tensor(float("-inf"), device=cost.device, dtype=cost.dtype)
    log_a = torch.where(valid_a, -torch.log(count_a), neg_inf)  # (B, N)
    log_b = torch.where(valid_b, -torch.log(count_b), neg_inf)  # (B, M)

    f = torch.where(valid_a, torch.zeros_like(log_a), neg_inf)
    g = torch.where(valid_b, torch.zeros_like(log_b), neg_inf)

    for _ in range(num_iterations):
        f = epsilon * log_a - epsilon * torch.logsumexp((g.unsqueeze(1) - cost) / epsilon, dim=2)
        g = epsilon * log_b - epsilon * torch.logsumexp((f.unsqueeze(2) - cost) / epsilon, dim=1)

    transport_plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / epsilon)
    return (transport_plan * cost).sum(dim=(1, 2))


def batched_dense_matching_distance(
    dense_features_a: torch.Tensor,
    dense_features_b: torch.Tensor,
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    local_projection: torch.nn.Module,
    epsilon: float = 0.05,
    num_iterations: int = 50,
) -> torch.Tensor:
    """The Step 4 structural distance for a whole batch of pairs at once.

    `(B, grid, grid, dim)` raw dense grids x2 + `(B, grid, grid)` bool
    masks x2 + the trainable `local_projection` (applied to the PADDED
    raw pieces in one batched matmul, before the cost matrix - padded
    rows become an arbitrary but finite bias-vector output, harmless
    since they carry zero transport-plan mass regardless) -> `(B,)`
    distances, fully differentiable back through `local_projection` (and
    through the encoder, for whichever pieces of it are unfrozen)."""
    padded_a, valid_a = _extract_and_pad(dense_features_a, masks_a)
    padded_b, valid_b = _extract_and_pad(dense_features_b, masks_b)

    projected_a = local_projection(padded_a)
    projected_b = local_projection(padded_b)

    cost = batched_cosine_cost_matrix(projected_a, projected_b)
    return batched_sinkhorn_ot_distance(cost, valid_a, valid_b, epsilon=epsilon, num_iterations=num_iterations)


def dense_matching_distance_projected(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    local_projection: torch.nn.Module,
    epsilon: float = 0.05,
    num_iterations: int = 50,
) -> torch.Tensor:
    """Single-pair convenience wrapper around the batched primitives above
    (batch size 1) - used by the per-epoch verification callback and other
    single-pair evaluation call sites, so there is exactly ONE Sinkhorn
    implementation to trust for Step 4, not two. Unlike
    `dense_matching_distance`, applies `local_projection`'s trainable
    weights to the pieces before matching."""
    distances = batched_dense_matching_distance(
        features_a.unsqueeze(0), features_b.unsqueeze(0),
        mask_a.unsqueeze(0), mask_b.unsqueeze(0),
        local_projection, epsilon=epsilon, num_iterations=num_iterations,
    )
    return distances.squeeze(0)
