"""Standalone metric functions (no model dependency).

All functions take numpy arrays and return Python floats / numpy arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ----------------------------------------------------------------------
def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Top-1 accuracy."""
    return float(accuracy_score(targets, predictions))


# ----------------------------------------------------------------------
def compute_top_k_accuracy(
    similarities: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
) -> float:
    """Top-k accuracy from similarity matrix."""
    if similarities.ndim != 2:
        raise ValueError(f"similarities must be (N, C), got {similarities.shape}")
    k = min(k, similarities.shape[1])
    top_k = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    return float(np.any(top_k == targets[:, None], axis=1).mean())


# ----------------------------------------------------------------------
def compute_mean_average_precision(
    similarities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """mAP for single-positive retrieval (one ground truth per query)."""
    N, C = similarities.shape
    aps: List[float] = []
    for i in range(N):
        order = np.argsort(-similarities[i])             # high to low
        rel = (order == targets[i]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        precision_at_k = np.cumsum(rel) / np.arange(1, C + 1)
        aps.append(float((precision_at_k * rel).sum() / rel.sum()))
    return float(np.mean(aps)) if aps else 0.0


# ----------------------------------------------------------------------
def compute_retrieval_metrics(
    visual_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute R@K and median rank for both retrieval directions."""
    if visual_embeds.shape[0] != labels.shape[0]:
        raise ValueError(
            f"visual_embeds {visual_embeds.shape} vs labels {labels.shape}"
        )
    sim = visual_embeds @ text_embeds.T                  # (N, M)
    out: Dict[str, float] = {}

    # video-to-text
    v2t_order = np.argsort(-sim, axis=1)
    for k in (1, 5, 10):
        if k > v2t_order.shape[1]:
            continue
        hits = np.any(v2t_order[:, :k] == labels[:, None], axis=1)
        out[f"v2t_r@{k}"] = float(hits.mean())

    # text-to-video — only meaningful when N == M
    if sim.shape[0] == sim.shape[1]:
        t2v_order = np.argsort(-sim.T, axis=1)
        for k in (1, 5, 10):
            if k > t2v_order.shape[1]:
                continue
            hits = np.any(t2v_order[:, :k] == labels[:, None], axis=1)
            out[f"t2v_r@{k}"] = float(hits.mean())

    # Median ranks
    v2t_ranks = np.array(
        [int(np.where(v2t_order[i] == labels[i])[0][0]) for i in range(len(labels))]
    )
    out["v2t_median_rank"] = float(np.median(v2t_ranks))
    return out


# ----------------------------------------------------------------------
def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None,
    normalize: Optional[str] = "true",
) -> np.ndarray:
    if num_classes is None:
        num_classes = int(max(predictions.max(), targets.max())) + 1
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    cm = cm.astype(np.float64)
    if normalize == "true":
        s = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, s, out=np.zeros_like(cm), where=s > 0)
    elif normalize == "pred":
        s = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, s, out=np.zeros_like(cm), where=s > 0)
    elif normalize == "all":
        s = cm.sum()
        cm = cm / s if s > 0 else cm
    return cm


# ----------------------------------------------------------------------
def compute_per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Accuracy for each ground-truth class."""
    out: Dict[str, float] = {}
    for c in np.unique(targets):
        mask = targets == c
        key = class_names[c] if class_names and c < len(class_names) else str(c)
        out[key] = float((predictions[mask] == c).mean())
    return out


# ----------------------------------------------------------------------
def compute_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=np.unique(targets), zero_division=0
    )
    out = {
        "accuracy":           compute_accuracy(predictions, targets),
        "macro_precision":    float(precision.mean()),
        "macro_recall":       float(recall.mean()),
        "macro_f1":           float(f1.mean()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall":    float(np.average(recall, weights=support)),
        "weighted_f1":        float(np.average(f1, weights=support)),
    }
    return out
