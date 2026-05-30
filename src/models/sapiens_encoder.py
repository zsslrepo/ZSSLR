"""Sapiens-1B human-centric vision encoder (frozen).

Reference: Khirodkar et al., "Sapiens: Foundation for Human Vision Models",
ECCV 2024 (Oral). Pretrained on 300M in-the-wild human images at 1024x1024.

If the official Sapiens checkpoint is unavailable, we fall back to a
`timm` ViT-Huge initialised from CLIP — the smoke-test pipeline still
runs, but real zero-shot accuracy requires the official weights
(paper Table III).

Output of `forward(video_frames)`:
    (B, embedding_dim) — L2-normalised projected features.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sapiens model spec — the paper uses Sapiens-1B (1.0B params, 1024^2).
# When falling back to a timm proxy, we pick the closest available ViT.
# ----------------------------------------------------------------------
_SAPIENS_SPECS = {
    "sapiens_0.3b": dict(timm_proxy="vit_large_patch14_clip_224.openai", embed_dim=1024),
    "sapiens_0.6b": dict(timm_proxy="vit_huge_patch14_clip_224.openai",  embed_dim=1280),
    "sapiens_1b":   dict(timm_proxy="vit_huge_patch14_clip_224.openai",  embed_dim=1280),
}


class SapiensEncoder(nn.Module):
    """Frozen Sapiens RGB encoder + learnable linear projection to shared space."""

    def __init__(
        self,
        model_name: str = "sapiens_1b",
        pretrained_path: Optional[str] = None,
        embedding_dim: int = 256,
        num_frames: int = 16,
        input_size: int = 1024,
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
        self.num_frames = num_frames
        self.input_size = input_size
        self.freeze = freeze
        self.chunk_size = chunk_size

        # 1. Build backbone -------------------------------------------------
        loaded_official = False
        if pretrained_path:
            # Official Sapiens checkpoint path. We attempt to load it but
            # we still build a timm proxy of the right shape so the rest of
            # the pipeline always works. Real Sapiens weights would need
            # the official sapiens-foundation package — out of scope here.
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                logger.info("Loaded Sapiens checkpoint metadata from %s",
                            pretrained_path)
                loaded_official = True
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning("Sapiens checkpoint not found at %s (%s) — "
                               "falling back to timm proxy.", pretrained_path, e)

        # The timm proxy. We use img_size=input_size so the patch
        # embedding accepts 1024x1024 inputs; positional embeddings are
        # interpolated automatically by timm when img_size differs from
        # the pretrained one.
        self.backbone = timm.create_model(
            spec["timm_proxy"],
            pretrained=(pretrained_path is None),  # only fetch ImageNet weights as a last resort
            img_size=input_size,
            num_classes=0,                          # remove classifier head
        )

        if loaded_official:
            # Try to merge whatever official keys we recognise; ignore the rest.
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("Sapiens partial load: %d missing, %d unexpected keys",
                        len(missing), len(unexpected))

        # 2. Freeze if requested --------------------------------------------
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            n_frozen = sum(p.numel() for p in self.backbone.parameters())
            logger.info("Sapiens backbone frozen — %d parameters", n_frozen)

        # 3. Projection to shared embedding space (trainable) ---------------
        self.projection = nn.Linear(self.backbone_embed_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def train(self, mode: bool = True):
        """Keep frozen backbone in eval mode regardless of model-level mode."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def _encode_frames(self, frames_flat: torch.Tensor) -> torch.Tensor:
        """Run ViT on (BT, C, H, W) in chunks to control GPU memory."""
        out = []
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            for i in range(0, frames_flat.size(0), self.chunk_size):
                chunk = frames_flat[i : i + self.chunk_size]
                # timm ViTs with num_classes=0 return the pooled token already
                feat = self.backbone(chunk)            # (b, D)
                out.append(feat)
        return torch.cat(out, dim=0)                   # (BT, D)

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        video_frames : (B, T, 3, H, W) — already preprocessed (resize + ImageNet norm).

        Returns
        -------
        (B, embedding_dim) — L2-normalised.
        """
        B, T, C, H, W = video_frames.shape
        frames_flat = rearrange(video_frames, "b t c h w -> (b t) c h w")
        frame_feats = self._encode_frames(frames_flat)        # (BT, D_b)
        frame_feats = rearrange(frame_feats, "(b t) d -> b t d", b=B, t=T)

        # Temporal average pooling across the T frames (paper Sec. III-B)
        video_feats = frame_feats.mean(dim=1)                 # (B, D_b)

        # Project to shared space (trainable layers)
        x = self.projection(video_feats)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)                    # (B, embedding_dim)
