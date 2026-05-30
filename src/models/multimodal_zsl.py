"""End-to-end model: Sapiens (RGB) + MotionBERT (skeleton) + AzBERT (text).

Paper Eq. (1): v = L2Norm(W_rgb * v_rgb + W_skel * v_skel)
Paper Eq. (2): Symmetric InfoNCE with learnable temperature.

Only `azbert` (text encoder) and the linear projection heads inside the
two visual encoders are trainable; the visual transformer backbones are
kept frozen — paper Sec. III-B / V-C.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sapiens_encoder import SapiensEncoder
from .motionbert_encoder import MotionBERTEncoder
from .azbert_encoder import AzBERTEncoder
from ..losses.contrastive_losses import SymmetricInfoNCE

logger = logging.getLogger(__name__)


class MultimodalZSLModel(nn.Module):
    """Three-encoder contrastive model for zero-shot sign recognition."""

    def __init__(
        self,
        sapiens_config: Optional[Dict] = None,
        motionbert_config: Optional[Dict] = None,
        azbert_config: Optional[Dict] = None,
        embedding_dim: int = 256,
        temperature: float = 0.07,
        temperature_learnable: bool = True,
    ):
        super().__init__()
        sapiens_cfg = sapiens_config or {}
        motionbert_cfg = motionbert_config or {}
        azbert_cfg = azbert_config or {}

        # ------------------------------------------------------------------
        # Visual encoders (frozen by default — see paper Sec. V-C)
        # ------------------------------------------------------------------
        self.sapiens = SapiensEncoder(
            model_name=sapiens_cfg.get("model_name", "sapiens_1b"),
            pretrained_path=sapiens_cfg.get("pretrained_path"),
            embedding_dim=embedding_dim,
            num_frames=sapiens_cfg.get("num_frames", 16),
            input_size=sapiens_cfg.get("input_size", 1024),
            freeze=sapiens_cfg.get("freeze", True),
            chunk_size=sapiens_cfg.get("chunk_size", 8),
        )

        self.motionbert = MotionBERTEncoder(
            num_joints=motionbert_cfg.get("num_joints", 133),
            embed_dim=motionbert_cfg.get("embed_dim", 512),
            num_layers=motionbert_cfg.get("num_layers", 8),
            pretrained_path=motionbert_cfg.get("pretrained_path"),
            projection_dim=embedding_dim,
            joint_mask_ratio=motionbert_cfg.get("joint_mask_ratio", 0.15),
            joint_noise_std=motionbert_cfg.get("joint_noise_std", 0.02),
            freeze=motionbert_cfg.get("freeze", True),
        )

        # ------------------------------------------------------------------
        # Text encoder (trainable)
        # ------------------------------------------------------------------
        self.azbert = AzBERTEncoder(
            model_name=azbert_cfg.get("model_name", "language-ml-lab/AzerBert"),
            embedding_dim=embedding_dim,
            max_length=azbert_cfg.get("max_length", 128),
            num_prompt_templates=azbert_cfg.get("num_prompt_templates", 5),
            freeze=azbert_cfg.get("freeze", False),
        )

        # ------------------------------------------------------------------
        # Loss (paper Eq. 2)
        # ------------------------------------------------------------------
        self.criterion = SymmetricInfoNCE(
            temperature=temperature,
            learnable_temp=temperature_learnable,
        )

        self.embedding_dim = embedding_dim

    # ----------------------------------------------------------------------
    def encode_visual(
        self,
        video_frames: torch.Tensor,
        skeleton: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements paper Eq. (1).

        Args
        ----
        video_frames : (B, T, 3, H, W) — already preprocessed.
        skeleton     : (B, T, J, 2) — optional. If None, returns RGB-only.

        Returns
        -------
        (B, embedding_dim) — L2-normalised.
        """
        v_rgb = self.sapiens(video_frames)              # (B, D), L2-normed
        if skeleton is None:
            return v_rgb
        v_skel = self.motionbert(skeleton)              # (B, D), L2-normed
        v = v_rgb + v_skel
        return F.normalize(v, p=2, dim=-1)

    # ----------------------------------------------------------------------
    def encode_text(
        self,
        descriptions: Sequence[str],
        use_prompt_ensemble: bool = True,
    ) -> torch.Tensor:
        return self.azbert(descriptions, use_prompt_ensemble=use_prompt_ensemble)

    # ----------------------------------------------------------------------
    def forward(
        self,
        video_frames: torch.Tensor,
        descriptions: Sequence[str],
        skeleton: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training forward: computes embeddings and symmetric InfoNCE loss."""
        v = self.encode_visual(video_frames, skeleton)
        t = self.encode_text(descriptions)
        return self.criterion(v, t)

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def zero_shot_classify(
        self,
        video_frames: torch.Tensor,
        class_descriptions: Dict[str, str],
        skeleton: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args
        ----
        video_frames : (B, T, 3, H, W).
        class_descriptions : ordered dict {class_name: description}.

        Returns
        -------
        dict with keys: 'pred_classes' (B,), 'pred_indices' (B,),
        'similarities' (B, C), 'visual_embeds' (B, D), 'text_embeds' (C, D).
        """
        was_training = self.training
        self.eval()

        v = self.encode_visual(video_frames, skeleton)        # (B, D)
        class_names = list(class_descriptions.keys())
        descs = [class_descriptions[c] for c in class_names]
        t = self.encode_text(descs)                            # (C, D)

        sim = v @ t.t()                                        # (B, C)
        idx = sim.argmax(dim=1)                                # (B,)

        if was_training:
            self.train()

        return {
            "pred_classes":  [class_names[i] for i in idx.cpu().tolist()],
            "pred_indices":  idx.cpu().numpy(),
            "similarities":  sim.cpu().numpy(),
            "visual_embeds": v.cpu().numpy(),
            "text_embeds":   t.cpu().numpy(),
            "class_names":   class_names,
        }

    # ----------------------------------------------------------------------
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only parameters whose `requires_grad` is True."""
        return [p for p in self.parameters() if p.requires_grad]

    # ----------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def n(mod): return sum(p.numel() for p in mod.parameters())
        def nt(mod): return sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return {
            "sapiens_total":      n(self.sapiens),
            "sapiens_trainable":  nt(self.sapiens),
            "motionbert_total":   n(self.motionbert),
            "motionbert_trainable": nt(self.motionbert),
            "azbert_total":       n(self.azbert),
            "azbert_trainable":   nt(self.azbert),
            "total":              n(self),
            "trainable":          nt(self),
        }
