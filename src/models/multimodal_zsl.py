"""End-to-end model: Sapiens (RGB) + MotionBERT (skeleton) + AzBERT (text).

Paper Eq. (1) — multimodal fusion:
    v = Sapiens + Temporal Transformer      (B, visual_dim=1024)
    m = MotionBERT                          (B, motion_dim=512)
    h = ReLU(W1 [v; m] + b1)                (B, embedding_dim=512)
    z_v = L2Norm(W2 h + b2)                 (B, embedding_dim)

Paper Eq. (2): symmetric InfoNCE aligns z_v with the text embedding z_t.

Only the AzBERT text encoder, the Sapiens frame-projection + Temporal
Transformer, the MotionBERT projection, and the fusion head are trainable;
the two foundation backbones stay frozen — paper Sec. III-C.
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
        embedding_dim: int = 512,
        visual_dim: int = 1024,
        motion_dim: int = 512,
        temperature: float = 0.07,
        temperature_learnable: bool = False,
    ):
        super().__init__()
        sapiens_cfg = sapiens_config or {}
        motionbert_cfg = motionbert_config or {}
        azbert_cfg = azbert_config or {}

        # ------------------------------------------------------------------
        # Visual encoders (frozen backbones — see paper Sec. III-C)
        # ------------------------------------------------------------------
        self.sapiens = SapiensEncoder(
            model_name=sapiens_cfg.get("model_name", "sapiens_1b"),
            pretrained_path=sapiens_cfg.get("pretrained_path"),
            visual_dim=visual_dim,
            num_frames=sapiens_cfg.get("num_frames", 32),
            input_size=sapiens_cfg.get("input_size", 1024),
            temporal_layers=sapiens_cfg.get("temporal_layers", 1),
            temporal_heads=sapiens_cfg.get("temporal_heads", 8),
            freeze=sapiens_cfg.get("freeze", True),
            chunk_size=sapiens_cfg.get("chunk_size", 8),
        )

        self.motionbert = MotionBERTEncoder(
            num_joints=motionbert_cfg.get("num_joints", 17),
            embed_dim=motionbert_cfg.get("embed_dim", 512),
            num_layers=motionbert_cfg.get("num_layers", 5),
            pretrained_path=motionbert_cfg.get("pretrained_path"),
            motion_dim=motion_dim,
            joint_mask_ratio=motionbert_cfg.get("joint_mask_ratio", 0.15),
            joint_noise_std=motionbert_cfg.get("joint_noise_std", 0.02),
            freeze=motionbert_cfg.get("freeze", True),
        )

        # ------------------------------------------------------------------
        # Multimodal fusion head (paper Eq. 1): concat -> MLP -> L2-norm
        # ------------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + motion_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ------------------------------------------------------------------
        # Text encoder (trainable) — projects to the shared embedding_dim.
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
        self.visual_dim = visual_dim
        self.motion_dim = motion_dim

    # ----------------------------------------------------------------------
    def encode_visual(
        self,
        video_frames: torch.Tensor,
        skeleton: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements paper Eq. (1): concat visual + motion, MLP-fuse, L2-norm.

        Args
        ----
        video_frames : (B, T, 3, H, W) — already preprocessed.
        skeleton     : (B, T, J, 2) — optional. If None, the motion stream is
                       zeroed (matches the "without skeleton" ablation).

        Returns
        -------
        (B, embedding_dim) — L2-normalised fused video embedding.
        """
        v = self.sapiens(video_frames)                  # (B, visual_dim)
        if skeleton is None:
            m = torch.zeros(v.size(0), self.motion_dim,
                            device=v.device, dtype=v.dtype)
        else:
            m = self.motionbert(skeleton)               # (B, motion_dim)
        fused = torch.cat([v, m], dim=-1)               # (B, visual_dim + motion_dim)
        z = self.fusion(fused)                          # (B, embedding_dim)
        return F.normalize(z, p=2, dim=-1)

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
            "fusion_total":       n(self.fusion),
            "fusion_trainable":   nt(self.fusion),
            "total":              n(self),
            "trainable":          nt(self),
        }
