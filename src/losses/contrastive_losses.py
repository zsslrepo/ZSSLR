"""Symmetric InfoNCE — paper Eq. (2).

Only this loss is actually used in the paper. The original repository
contained four unused variants (Asymmetric, NT-Xent, Weighted, Focal)
which we removed to keep the codebase faithful to what is reported.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    r"""CLIP-style symmetric contrastive loss with learnable temperature.

    For an L2-normalised batch of visual embeddings :math:`v \in \mathbb{R}^{N \times D}`
    and text embeddings :math:`t \in \mathbb{R}^{N \times D}`,

    .. math::
        \mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N} \Big[
            -\log\frac{\exp(v_i^\top t_i / \tau)}{\sum_j \exp(v_i^\top t_j / \tau)}
            -\log\frac{\exp(t_i^\top v_i / \tau)}{\sum_j \exp(t_i^\top v_j / \tau)}
        \Big]

    `logit_scale = 1 / tau` is parameterised in log-space for stability,
    following Radford et al. (CLIP, 2021).
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        log_scale = np.log(1.0 / temperature)
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.tensor(log_scale, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(log_scale, dtype=torch.float32))
        self.learnable_temp = learnable_temp

    # ----------------------------------------------------------------------
    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args
        ----
        visual : (N, D) — L2-normalised.
        text   : (N, D) — L2-normalised.

        Returns
        -------
        loss : scalar tensor.
        metrics : dict of scalar Python floats (for logging).
        """
        if visual.shape != text.shape:
            raise ValueError(f"visual {tuple(visual.shape)} vs text {tuple(text.shape)}")
        N = visual.size(0)
        if N < 2:
            raise ValueError("Symmetric InfoNCE requires batch size >= 2")

        # Clamp logit_scale to prevent fp16/bf16 overflow (Radford CLIP trick)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_v2t = logit_scale * visual @ text.t()           # (N, N)
        logits_t2v = logits_v2t.t()

        labels = torch.arange(N, device=visual.device)
        loss_v = F.cross_entropy(logits_v2t, labels)
        loss_t = F.cross_entropy(logits_t2v, labels)
        loss = 0.5 * (loss_v + loss_t)

        with torch.no_grad():
            acc_v = (logits_v2t.argmax(dim=1) == labels).float().mean().item()
            acc_t = (logits_t2v.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            "loss":         loss.item(),
            "loss_v2t":     loss_v.item(),
            "loss_t2v":     loss_t.item(),
            "acc_v2t":      acc_v,
            "acc_t2v":      acc_t,
            "acc":          0.5 * (acc_v + acc_t),
            "temperature":  float(1.0 / logit_scale.item()),
            "logit_scale":  float(logit_scale.item()),
        }
        return loss, metrics
