"""MotionBERT skeleton encoder (frozen).

Reference: Zhu et al., "MotionBERT: A Unified Perspective on Learning
Human Motion Representations", ICCV 2023.

The encoder is a DSTformer (Dual-Stream Transformer) operating on 2D
keypoint sequences. We follow the paper's protocol: random joint
masking (15%) and small Gaussian noise (sigma=0.02) during training to
simulate MediaPipe imperfections.

If the official DSTformer checkpoint is unavailable, an
architecturally-compatible-but-untrained DSTformer is constructed.
The smoke test passes; real performance requires the official weights.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DSTformer building blocks
# ----------------------------------------------------------------------
class DSTFormerBlock(nn.Module):
    """One spatial + one temporal attention head + MLP, pre-norm residual."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, J, D)
        B, T, J, D = x.shape

        # Spatial attention — across joints within each frame
        xs = rearrange(x, "b t j d -> (b t) j d")
        ys, _ = self.attn_s(self.norm_s(xs), self.norm_s(xs), self.norm_s(xs))
        xs = xs + self.dropout(ys)
        x = rearrange(xs, "(b t) j d -> b t j d", b=B, t=T)

        # Temporal attention — across time for each joint
        xt = rearrange(x, "b t j d -> (b j) t d")
        yt, _ = self.attn_t(self.norm_t(xt), self.norm_t(xt), self.norm_t(xt))
        xt = xt + self.dropout(yt)
        x = rearrange(xt, "(b j) t d -> b t j d", b=B, j=J)

        # MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class DSTFormer(nn.Module):
    """Stack of N DSTFormer blocks operating on (B, T, J, 2) inputs."""

    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 2,
        embed_dim: int = 512,
        num_layers: int = 5,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.joint_embed = nn.Linear(in_dim, embed_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_seq_len, 1, embed_dim))
        self.blocks = nn.ModuleList([
            DSTFormerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, J, 2)
        B, T, J, _ = x.shape
        x = self.joint_embed(x)
        x = x + self.spatial_pos[:, :, :J, :]
        x = x + self.temporal_pos[:, :T, :, :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                          # (B, T, J, D)


# ----------------------------------------------------------------------
# Public encoder
# ----------------------------------------------------------------------
class MotionBERTEncoder(nn.Module):
    """Wraps DSTformer with augmentation, pooling, and projection."""

    def __init__(
        self,
        num_joints: int = 17,
        embed_dim: int = 512,
        num_layers: int = 5,
        pretrained_path: Optional[str] = None,
        motion_dim: int = 512,
        joint_mask_ratio: float = 0.15,
        joint_noise_std: float = 0.02,
        freeze: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_mask_ratio = joint_mask_ratio
        self.joint_noise_std = joint_noise_std
        self.freeze = freeze

        self.backbone = DSTFormer(
            num_joints=num_joints,
            in_dim=2,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                logger.info("MotionBERT loaded from %s — %d missing, %d unexpected",
                            pretrained_path, len(missing), len(unexpected))
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning("MotionBERT checkpoint not found at %s (%s). "
                               "Falling back to untrained DSTformer (smoke test only).",
                               pretrained_path, e)

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            n = sum(p.numel() for p in self.backbone.parameters())
            logger.info("MotionBERT backbone frozen — %d parameters", n)

        self.motion_dim = motion_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, motion_dim),
            nn.LayerNorm(motion_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Augmentation (training-only)
    # ------------------------------------------------------------------
    def _augment(self, skeleton: torch.Tensor) -> torch.Tensor:
        """Joint masking + Gaussian noise (paper Sec. III-B)."""
        if not self.training:
            return skeleton
        if self.joint_mask_ratio > 0:
            B, T, J, _ = skeleton.shape
            keep = (torch.rand(B, T, J, 1, device=skeleton.device)
                    > self.joint_mask_ratio).float()
            skeleton = skeleton * keep
        if self.joint_noise_std > 0:
            skeleton = skeleton + torch.randn_like(skeleton) * self.joint_noise_std
        return skeleton

    # ------------------------------------------------------------------
    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        skeleton : (B, T, J, 2) — MediaPipe Holistic 2D keypoints.

        Returns
        -------
        (B, motion_dim) — motion embedding (NOT L2-normalised; the
        multimodal fusion head applies the final norm).
        """
        skeleton = self._augment(skeleton)

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feats = self.backbone(skeleton)              # (B, T, J, D)

        # Pool over joints first, then over time (paper Sec. III-C)
        feats = feats.mean(dim=2)                        # (B, T, D)
        feats = feats.mean(dim=1)                        # (B, D)

        return self.projection(feats)                    # (B, motion_dim)
