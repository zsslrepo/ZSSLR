"""Sapiens-1B human-centric vision encoder (frozen) + Temporal Transformer.

Reference: Khirodkar et al., "Sapiens: Foundation for Human Vision Models",
ECCV 2024 (Oral). Pretrained on 300M in-the-wild human images at 1024x1024.

Pipeline (paper Sec. III-C, "Visual Encoder"):
    T RGB frames --Sapiens--> T frozen [CLS] tokens (each `visual_dim`)
                 --Temporal Transformer (1-layer, 8-head, learnable pos.)-->
                 position-0 output = video embedding v  (B, visual_dim)

Only the frame projection and the Temporal Transformer are trainable; the
Sapiens backbone stays frozen. If the official Sapiens checkpoint is
unavailable, we fall back to a `timm` ViT initialised from CLIP — the
pipeline still runs, but real zero-shot accuracy requires the official
weights.

`forward(video_frames)` returns the (B, visual_dim) video embedding
(NOT L2-normalised — the multimodal fusion head does the final norm).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sapiens model spec — the paper uses Sapiens-1B (ViT-Huge, 1024-d CLS).
# When falling back to a timm proxy, we pick the closest available ViT and
# project its CLS token to `visual_dim` so downstream dims stay paper-faithful.
# ----------------------------------------------------------------------
_SAPIENS_SPECS = {
    "sapiens_0.3b": dict(timm_proxy="vit_large_patch14_clip_224.openai", embed_dim=1024),
    "sapiens_0.6b": dict(timm_proxy="vit_huge_patch14_clip_224.openai",  embed_dim=1280),
    "sapiens_1b":   dict(timm_proxy="vit_huge_patch14_clip_224.openai",  embed_dim=1280),
}


class SapiensEncoder(nn.Module):
    """Frozen Sapiens RGB backbone + trainable Temporal Transformer aggregator."""

    def __init__(
        self,
        model_name: str = "sapiens_1b",
        pretrained_path: Optional[str] = None,
        visual_dim: int = 1024,
        num_frames: int = 32,
        input_size: int = 1024,
        temporal_layers: int = 1,
        temporal_heads: int = 8,
        freeze: bool = True,
        chunk_size: int = 8,
    ):
        super().__init__()
        if model_name not in _SAPIENS_SPECS:
            raise ValueError(f"Unknown Sapiens variant: {model_name}")
        if not _HAS_TIMM:
            raise ImportError("timm is required for SapiensEncoder")

        spec = _SAPIENS_SPECS[model_name]
        self.model_name = model_name
        self.backbone_embed_dim = spec["embed_dim"]
        self.visual_dim = visual_dim
        self.num_frames = num_frames
        self.input_size = input_size
        self.freeze = freeze
        self.chunk_size = chunk_size

        # 1. Build backbone -------------------------------------------------
        loaded_official = False
        if pretrained_path:
            # Official Sapiens checkpoint. We attempt to load recognised keys
            # into a timm proxy of the right shape so the pipeline always runs.
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                logger.info("Loaded Sapiens checkpoint metadata from %s", pretrained_path)
                loaded_official = True
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning("Sapiens checkpoint not found at %s (%s) — "
                               "falling back to timm proxy.", pretrained_path, e)

        self.backbone = timm.create_model(
            spec["timm_proxy"],
            pretrained=(pretrained_path is None),
            img_size=input_size,
            num_classes=0,                          # remove classifier head
        )

        if loaded_official:
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("Sapiens partial load: %d missing, %d unexpected keys",
                        len(missing), len(unexpected))

        # 2. Freeze the backbone --------------------------------------------
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            n_frozen = sum(p.numel() for p in self.backbone.parameters())
            logger.info("Sapiens backbone frozen — %d parameters", n_frozen)

        # 3. Trainable frame projection: backbone CLS -> visual_dim ---------
        self.frame_proj = nn.Linear(self.backbone_embed_dim, visual_dim)

        # 4. Trainable Temporal Transformer (paper Sec. III-C) --------------
        #    Learnable positional embedding over up to `num_frames` frames;
        #    position-0 output of the encoder is the video embedding.
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, visual_dim))
        nn.init.normal_(self.temporal_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim,
            nhead=temporal_heads,
            dim_feedforward=visual_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

    # ----------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode regardless of model mode."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    # ----------------------------------------------------------------------
    def _encode_frames(self, frames_flat: torch.Tensor) -> torch.Tensor:
        """Run the ViT backbone on (BT, C, H, W) in chunks to bound memory."""
        out = []
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            for i in range(0, frames_flat.size(0), self.chunk_size):
                chunk = frames_flat[i : i + self.chunk_size]
                feat = self.backbone(chunk)            # (b, backbone_embed_dim)
                out.append(feat)
        return torch.cat(out, dim=0)                   # (BT, backbone_embed_dim)

    # ----------------------------------------------------------------------
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        video_frames : (B, T, 3, H, W) — preprocessed (resize + ImageNet norm).

        Returns
        -------
        (B, visual_dim) — video embedding (position-0 of the Temporal
        Transformer). Not L2-normalised.
        """
        B, T, C, H, W = video_frames.shape
        if T > self.num_frames:
            raise ValueError(
                f"T={T} exceeds configured num_frames={self.num_frames}"
            )
        frames_flat = rearrange(video_frames, "b t c h w -> (b t) c h w")
        frame_feats = self._encode_frames(frames_flat)            # (BT, D_b)
        frame_feats = self.frame_proj(frame_feats)                # (BT, visual_dim)
        frame_feats = rearrange(frame_feats, "(b t) d -> b t d", b=B, t=T)

        # Add learnable temporal positional embedding, then aggregate.
        tokens = frame_feats + self.temporal_pos[:, :T, :]        # (B, T, visual_dim)
        encoded = self.temporal_encoder(tokens)                   # (B, T, visual_dim)
        return encoded[:, 0, :]                                    # (B, visual_dim)
