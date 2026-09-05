"""Downstream verification embedding model: the SSL-pretrained encoder plus
a small trainable projector, exposing both comparison methods off one
shared encoder (see `docs/claude_response/downstream_supervised_learning_approach.md`).

Adapted from the SURDS-era `DownstreamSignatureEmbeddingModel`
(`Thesis_Final/downstream_verification/models/Embedding_Model.py`), with
one structural addition: that model only ever exposed Method A (pooled
vector -> projector). This version exposes Method A unchanged AND Method
B (the raw dense feature grid, no projector) off the same forward pass
through the encoder, since both comparison methods need to be tested
against each other on identical underlying features (Step 5's 4-way
comparison).

Step 4 adds a second structural addition: `local_projection`, a small
trainable linear layer for the local/structural branch (DetailSemNet's
Eq. 3/4 - their local branch has learnable weights of its own; ours did
not until now). It is deliberately NOT applied inside `forward_dense` -
that method still returns the raw, unprojected grid (still needed for
Method B's original raw-feature comparisons and diagnostics); the
projection is applied downstream, only where the Step 4 combined distance
is actually being computed (`matching/dense_matching.py`,
`matching/combined_distance.py`), on the masked ink pieces only. See
`downstream_supervised_learning_approach.md` SS10 for the full design.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parent
SUPERVISED_DIR = MODELS_DIR.parent  # DenseCL_approach/supervised_verification_approach
DENSECL_APPROACH_DIR = SUPERVISED_DIR.parent  # DenseCL_approach
SELF_SUPERVISED_DIR = DENSECL_APPROACH_DIR / "self_supervised_approach"

sys.path.insert(0, str(SELF_SUPERVISED_DIR / "model"))
sys.path.insert(0, str(SELF_SUPERVISED_DIR / "analyzer"))

from encoder import Encoder  # noqa: E402
from similarity_analyzer import RESULTS_TRAINING_DIR, find_encoder_checkpoint  # noqa: E402


class DownstreamVerificationModel(nn.Module):
    """Wraps the SSL-pretrained encoder for the downstream verification
    phase. Exposes two forward paths off the SAME shared encoder:

      - `forward_global(x)`: Method A - pooled 256-d backbone feature ->
        trainable projector -> L2-normalized embedding. This is the only
        path with extra trainable weights (the projector); the encoder
        itself is frozen by default (`trainable_encoder_stages=()`),
        matching the published v4 baseline's setting.
      - `forward_dense(x)`: Method B - raw dense backbone feature grid
        (`pool=False`), NO projector applied. Per DenseCL's own ablation
        (Table 6), raw backbone features give better correspondence-
        matching quality than any extra projection head's output, so none
        is added here - foreground-masking and the actual piece-by-piece
        matching (Step 2) happen downstream of this, on these raw features.

    Passing `pretrained_ssl_checkpoint_path=None` leaves the encoder at its
    random initialization - the random-encoder control (Step 6 of the
    roadmap), with the architecture otherwise byte-for-byte identical to
    the trained version.
    """

    def __init__(
        self,
        pretrained_ssl_checkpoint_path: Optional[str | Path],
        trainable_encoder_stages: tuple[str, ...] = (),
        projector_hidden_dim: int = 256,
        embedding_dim: int = 256,
        norm_type: str = "batch",
        local_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(norm_type=norm_type)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.out_channels, projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, embedding_dim),
        )

        # Step 4's local/structural branch: one linear layer, applied
        # identically to every ink cell (shared weights across space -
        # DetailSemNet's own local branch is a single linear layer, not a
        # deep MLP). `None` (default) omits it entirely, so Steps 1-3's
        # model construction is byte-for-byte unaffected.
        self.local_projection: Optional[nn.Linear] = (
            nn.Linear(self.encoder.out_channels, local_embedding_dim)
            if local_embedding_dim is not None else None
        )

        if pretrained_ssl_checkpoint_path is not None:
            self._load_pretrained_encoder(Path(pretrained_ssl_checkpoint_path))
        else:
            print("No SSL checkpoint provided - encoder left at random initialization.")

        self._configure_partial_finetuning(trainable_encoder_stages)

    def _load_pretrained_encoder(self, checkpoint_path: Path) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained SSL encoder from: {checkpoint_path}")
        print(f"Missing keys:    {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

    def _configure_partial_finetuning(self, trainable_encoder_stages: tuple[str, ...]) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for stage_name in trainable_encoder_stages:
            if not hasattr(self.encoder, stage_name):
                raise ValueError(f"Unknown encoder stage: {stage_name}")
            for parameter in getattr(self.encoder, stage_name).parameters():
                parameter.requires_grad = True

        frozen_params = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        projector_params = sum(p.numel() for p in self.projector.parameters())

        print(f"Trainable encoder stages:   {trainable_encoder_stages if trainable_encoder_stages else 'none (fully frozen)'}")
        print(f"Frozen encoder params:      {frozen_params:,}")
        print(f"Trainable encoder params:   {trainable_params:,}")
        print(f"Trainable projector params: {projector_params:,}")
        if self.local_projection is not None:
            local_params = sum(p.numel() for p in self.local_projection.parameters())
            print(f"Trainable local projection params: {local_params:,} (dim={self.local_projection.out_features})")

    def forward_global(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Method A: (B, 1, 256, 256) -> L2-normalized (B, embedding_dim)."""
        pooled = self.encoder(image_tensor, pool=True)
        projected = self.projector(pooled)
        return F.normalize(projected, p=2, dim=1)

    def forward_dense(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Method B: (B, 1, 256, 256) -> raw backbone dense grid,
        (B, grid, grid, 256). No projector, no normalization - the
        piece-by-piece matching (Step 2) operates on these raw features,
        same convention as `similarity_analyzer.extract_dense_features`."""
        dense = self.encoder(image_tensor, pool=False)  # (B, 256, grid, grid)
        return dense.permute(0, 2, 3, 1)  # (B, grid, grid, 256)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Default forward = Method A (global), for drop-in compatibility
        with code that expects a single embedding per image (e.g. the dual
        triplet loss during training)."""
        return self.forward_global(image_tensor)


def load_downstream_model(
    run_name: str,
    checkpoint_epoch: Optional[int],
    device: torch.device,
    trainable_encoder_stages: tuple[str, ...] = (),
    projector_hidden_dim: int = 256,
    embedding_dim: int = 256,
    norm_type: str = "batch",
    local_embedding_dim: Optional[int] = None,
) -> DownstreamVerificationModel:
    """Locate `encoder_epoch<N>.pt` for `run_name` the same way the
    analysis tools do (`similarity_analyzer.find_encoder_checkpoint`),
    build the model, move to `device`. `checkpoint_epoch=None` picks the
    latest checkpoint found. `local_embedding_dim=None` (default) omits
    the Step 4 local branch entirely, matching Steps 1-3."""
    checkpoint_path, epoch = find_encoder_checkpoint(RESULTS_TRAINING_DIR / run_name, checkpoint_epoch)
    model = DownstreamVerificationModel(
        pretrained_ssl_checkpoint_path=checkpoint_path,
        trainable_encoder_stages=trainable_encoder_stages,
        projector_hidden_dim=projector_hidden_dim,
        embedding_dim=embedding_dim,
        norm_type=norm_type,
        local_embedding_dim=local_embedding_dim,
    )
    print(f"Run: {run_name} / epoch {epoch}")
    return model.to(device)


def load_random_init_downstream_model(
    trainable_encoder_stages: tuple[str, ...] = (),
    projector_hidden_dim: int = 256,
    embedding_dim: int = 256,
    norm_type: str = "batch",
    device: Optional[torch.device] = None,
    local_embedding_dim: Optional[int] = None,
) -> DownstreamVerificationModel:
    """The random-encoder control (Step 6): architecture identical to
    `load_downstream_model`'s, no pretrained weights loaded."""
    model = DownstreamVerificationModel(
        pretrained_ssl_checkpoint_path=None,
        trainable_encoder_stages=trainable_encoder_stages,
        projector_hidden_dim=projector_hidden_dim,
        embedding_dim=embedding_dim,
        norm_type=norm_type,
        local_embedding_dim=local_embedding_dim,
    )
    return model.to(device) if device is not None else model
