"""Azerbaijani BERT (azBERT) text encoder with batched prompt ensemble.

Reference: language-ml-lab/AzerBert on the Hugging Face Hub
(6-layer Transformer, 512-dim hidden states, 30K subword vocabulary).

Key implementation note
-----------------------
The original repo encoded each prompt template sequentially (Python loop
of size N * 5 forward passes through BERT). We batch all N * 5 prompts
into a single tokenizer call and a single BERT forward, which is ~5x
faster and produces identical embeddings.

`forward(descriptions)` returns a (B, embedding_dim) tensor that is
L2-normalised. Five Azerbaijani prompt templates wrap each description;
their projected CLS embeddings are averaged, then L2-normalised again.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Five Azerbaijani prompt templates (prompt ensemble, paper Sec. III-C).
# Each must contain the literal "{description}" placeholder.
# ----------------------------------------------------------------------
PROMPT_TEMPLATES_AZ: List[str] = [
    "Azərbaycan İşarə Dilində {description}",
    "{description} – AzSL işarəsi",
    "Kar şəxs {description} işarəsini göstərir",
    "Bu işarə {description} mənasını verir",
    "İşarə dili: {description}",
]


class AzBERTEncoder(nn.Module):
    """Trainable AzerBert + prompt ensemble + projection."""

    def __init__(
        self,
        model_name: str = "language-ml-lab/AzerBert",
        embedding_dim: int = 512,
        max_length: int = 128,
        num_prompt_templates: int = 5,
        freeze: bool = False,
        prompt_templates: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.max_length = max_length
        self.freeze = freeze

        if num_prompt_templates < 1 or num_prompt_templates > len(PROMPT_TEMPLATES_AZ):
            raise ValueError(
                f"num_prompt_templates must be in [1, {len(PROMPT_TEMPLATES_AZ)}]"
            )

        templates = list(prompt_templates) if prompt_templates is not None \
            else PROMPT_TEMPLATES_AZ[:num_prompt_templates]
        for t in templates:
            if "{description}" not in t:
                raise ValueError(
                    f"Prompt template missing '{{description}}' placeholder: {t!r}"
                )
        self.prompt_templates = templates
        self.num_prompts = len(templates)

        # ------------------------------------------------------------------
        # Load Hugging Face model + tokenizer with graceful fallback.
        # ------------------------------------------------------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.backbone.config.hidden_size
            logger.info("Loaded AzBERT from %s (hidden=%d)",
                        model_name, self.hidden_size)
        except (OSError, EnvironmentError) as e:
            logger.warning("Could not download %s (%s). "
                           "Falling back to bert-base-multilingual-cased.",
                           model_name, e)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            cfg = AutoConfig.from_pretrained("bert-base-multilingual-cased")
            self.backbone = AutoModel.from_pretrained("bert-base-multilingual-cased")
            self.hidden_size = cfg.hidden_size

        # ------------------------------------------------------------------
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            logger.info("AzBERT backbone frozen.")
        else:
            logger.info("AzBERT backbone is trainable.")

        # Projection to the shared (vision/text) embedding space
        self.projection = nn.Linear(self.hidden_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    def _wrap_with_prompts(self, descriptions: Sequence[str]) -> List[str]:
        """Return a (N * num_prompts,) flat list of prompt-wrapped strings."""
        wrapped = []
        for d in descriptions:
            for t in self.prompt_templates:
                wrapped.append(t.format(description=d))
        return wrapped

    # ------------------------------------------------------------------
    def forward(
        self,
        descriptions: Sequence[str],
        use_prompt_ensemble: bool = True,
    ) -> torch.Tensor:
        """
        Args
        ----
        descriptions : list of N strings.
        use_prompt_ensemble : if True, wrap each in `self.num_prompts`
            templates and average their embeddings.

        Returns
        -------
        (N, embedding_dim) — L2-normalised.
        """
        if not isinstance(descriptions, (list, tuple)):
            raise TypeError("descriptions must be a list/tuple of strings")
        N = len(descriptions)
        if N == 0:
            raise ValueError("descriptions list is empty")

        # Build flat list of prompts
        if use_prompt_ensemble:
            flat_prompts = self._wrap_with_prompts(descriptions)
            M = self.num_prompts
        else:
            flat_prompts = list(descriptions)
            M = 1

        # Single tokenizer call, single BERT forward
        enc = self.tokenizer(
            flat_prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask)

        cls = outputs.last_hidden_state[:, 0, :]          # (N*M, hidden)
        projected = self.projection(cls)                  # (N*M, D)
        projected = self.layer_norm(projected)
        # NOTE: do **not** L2-normalise before averaging — paper averages
        # the projected CLS vectors then normalises once at the end.
        projected = projected.view(N, M, -1).mean(dim=1)  # (N, D)
        return F.normalize(projected, p=2, dim=-1)
