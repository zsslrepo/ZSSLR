"""Zero-shot evaluator.

Encodes every test video and every class description, computes the full
similarity matrix, and returns Top-K / mAP / confusion matrix /
per-class accuracy. Also saves embeddings and predictions as .npz so
the visualisation notebook can reuse them without re-running the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.multimodal_zsl import MultimodalZSLModel
from ..utils.metrics import (
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix,
    compute_mean_average_precision,
    compute_per_class_accuracy,
    compute_top_k_accuracy,
)

logger = logging.getLogger(__name__)


class ZeroShotEvaluator:
    """Encodes everything once, then computes all metrics."""

    def __init__(
        self,
        model: MultimodalZSLModel,
        device: str = "cuda",
        text_batch_size: int = 32,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.text_batch_size = text_batch_size
        self.results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Batched text encoding with prompt ensemble."""
        chunks = []
        for i in range(0, len(descriptions), self.text_batch_size):
            chunk = list(descriptions[i : i + self.text_batch_size])
            emb = self.model.encode_text(chunk).detach().cpu().numpy()
            chunks.append(emb)
        return np.vstack(chunks)                            # (C, D)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        class_descriptions: Dict[str, str],
        split_name: str = "test",
    ) -> Dict:
        """Run zero-shot evaluation and cache results in `self.results`."""
        class_names = list(class_descriptions.keys())
        descriptions = [class_descriptions[c] for c in class_names]

        logger.info("Encoding %d class descriptions...", len(class_names))
        text_embeds = self._encode_descriptions(descriptions)   # (C, D)

        logger.info("Encoding videos...")
        vis_list, sim_list, pred_list, tgt_list = [], [], [], []
        vid_list, gloss_list = [], []

        for batch in tqdm(dataloader, desc=f"Eval ({split_name})"):
            frames = batch["frames"].to(self.device)
            skeleton = batch["skeleton"].to(self.device) \
                if batch["skeleton"] is not None else None
            labels = batch["label"].numpy()

            v = self.model.encode_visual(frames, skeleton).cpu().numpy()
            sims = v @ text_embeds.T                          # (B, C)
            preds = sims.argmax(axis=1)

            vis_list.append(v)
            sim_list.append(sims)
            pred_list.extend(preds.tolist())
            tgt_list.extend(labels.tolist())
            vid_list.extend(batch["video_id"])
            gloss_list.extend(batch["gloss"])

        vis_embeds = np.vstack(vis_list)
        similarities = np.vstack(sim_list)
        predictions = np.array(pred_list, dtype=np.int64)
        targets = np.array(tgt_list, dtype=np.int64)

        # --- Metrics --------------------------------------------------
        metrics: Dict[str, float] = {}
        metrics["top1"] = compute_accuracy(predictions, targets)
        metrics["top5"] = compute_top_k_accuracy(similarities, targets, k=5)
        metrics["top10"] = compute_top_k_accuracy(similarities, targets, k=10)
        metrics["mAP"] = compute_mean_average_precision(similarities, targets)

        cm = compute_confusion_matrix(predictions, targets, len(class_names))
        per_class = compute_per_class_accuracy(predictions, targets, class_names)
        report = compute_classification_report(predictions, targets, class_names)
        metrics.update({
            f"{k}": v for k, v in report.items()
            if isinstance(v, (int, float, np.floating))
        })

        self.results[split_name] = {
            "metrics": metrics,
            "predictions": predictions,
            "targets": targets,
            "similarities": similarities,
            "vis_embeds": vis_embeds,
            "text_embeds": text_embeds,
            "video_ids": vid_list,
            "glosses": gloss_list,
            "class_names": class_names,
            "confusion_matrix": cm,
            "per_class_accuracy": per_class,
        }

        # --- Pretty print ---------------------------------------------
        print(f"\n{'=' * 56}")
        print(f"Zero-Shot Evaluation — {split_name}")
        print(f"{'=' * 56}")
        print(f"  Samples:        {len(targets):>8}")
        print(f"  Classes:        {len(class_names):>8}")
        print(f"  Top-1 accuracy: {metrics['top1'] * 100:>7.2f}%")
        print(f"  Top-5 accuracy: {metrics['top5'] * 100:>7.2f}%")
        print(f"  Top-10 acc.:    {metrics['top10'] * 100:>7.2f}%")
        print(f"  mAP:            {metrics['mAP'] * 100:>7.2f}%")
        print(f"  Macro F1:       {metrics.get('macro_f1', 0.0) * 100:>7.2f}%")
        print(f"{'=' * 56}\n")

        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_per_signer(
        self,
        split_name: str,
        signer_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Break down Top-1 accuracy by signer using a cached evaluation."""
        if split_name not in self.results:
            raise RuntimeError(
                f"call evaluate(split_name={split_name!r}) before this"
            )
        r = self.results[split_name]
        correct: Dict[str, int] = {}
        total: Dict[str, int] = {}
        for i, vid in enumerate(r["video_ids"]):
            signer = signer_map.get(vid, "unknown")
            total[signer] = total.get(signer, 0) + 1
            if r["predictions"][i] == r["targets"][i]:
                correct[signer] = correct.get(signer, 0) + 1
        acc = {s: correct.get(s, 0) / total[s] for s in total}
        if acc:
            vals = list(acc.values())
            logger.info("Per-signer mean=%.4f std=%.4f n=%d",
                        float(np.mean(vals)), float(np.std(vals)), len(vals))
        return acc

    # ------------------------------------------------------------------
    def save_results(
        self,
        save_dir: Union[str, Path],
        split_name: str = "test",
    ) -> None:
        if split_name not in self.results:
            raise RuntimeError(f"no results for split {split_name!r}")
        save = Path(save_dir)
        save.mkdir(parents=True, exist_ok=True)
        r = self.results[split_name]

        metrics_scalar = {
            k: float(v) for k, v in r["metrics"].items()
            if isinstance(v, (int, float, np.floating))
        }
        (save / "metrics.json").write_text(
            json.dumps(metrics_scalar, indent=2),
            encoding="utf-8",
        )

        np.savez_compressed(
            save / "embeddings.npz",
            visual=r["vis_embeds"],
            text=r["text_embeds"],
            class_names=np.array(r["class_names"]),
        )
        np.savez_compressed(
            save / "predictions.npz",
            predictions=r["predictions"],
            targets=r["targets"],
            similarities=r["similarities"],
            video_ids=np.array(r["video_ids"]),
            glosses=np.array(r["glosses"]),
        )
        # Confusion matrix as a separate file (potentially huge)
        np.savez_compressed(
            save / "confusion_matrix.npz",
            cm=r["confusion_matrix"],
            class_names=np.array(r["class_names"]),
        )
        logger.info("Saved evaluation outputs to %s", save)
