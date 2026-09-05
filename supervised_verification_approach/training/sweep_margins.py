"""One-time margin sweep for Step 2's double-margin loss (`loss/double_margin_loss.py`).

`m`/`n` aren't published anywhere DetailSemNet reports (see that module's
docstring) - this script picks them empirically instead of guessing, by
looking at where genuine-pair and negative-pair distances actually land
under Step 1's already-trained checkpoint (same frozen encoder + trained
projector Step 2 starts from), measured on VALIDATION writers only.

Not a training loop: no gradients, one forward pass per validation image
(cached by path), one Euclidean distance per pair - only the trained
embedding space is being *read*, not changed.

For each candidate (m, n) pair, prints the resulting "active fraction" -
what share of genuine pairs still sit farther than m (i.e. would still
produce gradient under m) and what share of negative pairs still sit
closer than n. A margin so loose that ~0% of pairs are active reproduces
Step 1's dead-triplet problem before training even starts; a margin so
tight that ~100% are active gives no signal about which pairs are already
fine. Aim for something in between, then freeze the chosen (m, n) into
`trainer.py`'s `MARGIN_M`/`MARGIN_N`.

Usage:
    python sweep_margins.py [--dataset BHSig260_Hindi] [--run_tag step1_baseline_dualtriplet_frozen]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TRAINING_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = TRAINING_DIR.parent
for sub in ("utils", "models", "datasets", "evaluation"):
    sys.path.insert(0, str(SUPERVISED_DIR / sub))

from writer_splits import DATA_ROOT, get_writer_split  # noqa: E402
from embedding_model import load_downstream_model  # noqa: E402
from fixed_dual_triplet_dataset import build_fixed_triplet_records  # noqa: E402
from dual_triplet_dataset import load_signature_tensor  # noqa: E402

SSL_RUN_NAME = "densecl_pretrain_v1"
SSL_CHECKPOINT_EPOCH = 50
PERCENTILES = (10, 25, 50, 75, 90)
CANDIDATE_PERCENTILES = (30, 50, 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BHSig260_Hindi")
    parser.add_argument("--run_tag", default="step1_baseline_dualtriplet_frozen")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--tuples_per_anchor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = get_writer_split(args.dataset)
    dataset_dir = DATA_ROOT / args.dataset

    run_dir = SUPERVISED_DIR / "results" / "all_data_ssl" / args.dataset / args.run_tag
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    model = load_downstream_model(SSL_RUN_NAME, SSL_CHECKPOINT_EPOCH, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.run_tag}/{args.checkpoint} (epoch {checkpoint['epoch']})")

    records = build_fixed_triplet_records(
        split.validation_writer_ids, dataset_dir, args.dataset,
        tuples_per_anchor=args.tuples_per_anchor, seed=args.seed,
    )
    print(f"Validation writers: {len(split.validation_writer_ids)} | pair records: {len(records) * 3}")

    # Cache embeddings by path - the same anchor/positive image appears
    # across many records, no reason to re-encode it every time.
    embedding_cache: dict[Path, torch.Tensor] = {}

    @torch.no_grad()
    def embed(path: Path) -> torch.Tensor:
        if path not in embedding_cache:
            tensor = load_signature_tensor(path).unsqueeze(0).to(device)
            embedding_cache[path] = model(tensor).squeeze(0).cpu()
        return embedding_cache[path]

    positive_distances: list[float] = []
    negative_distances: list[float] = []
    with torch.no_grad():
        for record in records:
            anchor_embedding = embed(record.anchor_path).unsqueeze(0)
            positive_distances.append(
                float(F.pairwise_distance(anchor_embedding, embed(record.positive_path).unsqueeze(0)))
            )
            negative_distances.append(
                float(F.pairwise_distance(anchor_embedding, embed(record.negative_intra_path).unsqueeze(0)))
            )
            negative_distances.append(
                float(F.pairwise_distance(anchor_embedding, embed(record.negative_inter_path).unsqueeze(0)))
            )

    positive_distances_arr = np.array(positive_distances)
    negative_distances_arr = np.array(negative_distances)

    print(f"\nGenuine-genuine pairs (n={len(positive_distances_arr)}): mean={positive_distances_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(positive_distances_arr, p):.4f}")

    print(f"\nNegative pairs (n={len(negative_distances_arr)}): mean={negative_distances_arr.mean():.4f}")
    for p in PERCENTILES:
        print(f"  P{p}: {np.percentile(negative_distances_arr, p):.4f}")

    print(
        "\nCandidate (m, n) margins, from percentiles of the observed distributions, "
        "with the resulting active fraction (share of pairs that would still produce "
        "a nonzero gradient under that margin):"
    )
    print(f"{'m %ile':>8} {'n %ile':>8} {'m':>8} {'n':>8} {'pos_active':>11} {'neg_active':>11}")
    for m_pct in CANDIDATE_PERCENTILES:
        for n_pct in CANDIDATE_PERCENTILES:
            m = float(np.percentile(positive_distances_arr, m_pct))
            n = float(np.percentile(negative_distances_arr, n_pct))
            if m >= n:
                continue
            pos_active = float((positive_distances_arr > m).mean())
            neg_active = float((negative_distances_arr < n).mean())
            print(f"{m_pct:>7}% {n_pct:>7}% {m:>8.4f} {n:>8.4f} {pos_active:>10.1%} {neg_active:>10.1%}")


if __name__ == "__main__":
    main()
