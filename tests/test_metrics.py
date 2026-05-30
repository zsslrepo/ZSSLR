"""Unit tests for the standalone metric functions."""

import numpy as np
import pytest

from src.utils.metrics import (
    compute_accuracy,
    compute_top_k_accuracy,
    compute_mean_average_precision,
    compute_confusion_matrix,
    compute_per_class_accuracy,
)


def test_accuracy_perfect():
    preds = np.array([0, 1, 2, 3, 4])
    tgts = preds.copy()
    assert compute_accuracy(preds, tgts) == 1.0


def test_accuracy_chance():
    preds = np.array([0, 1, 2, 3, 4])
    tgts = np.array([1, 2, 3, 4, 0])
    assert compute_accuracy(preds, tgts) == 0.0


def test_top_k_accuracy_top1_matches_argmax():
    # When k=1 it must equal top-1 accuracy of argmax predictions
    sims = np.array([
        [0.9, 0.1, 0.0],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.0, 0.9, 0.1],
    ])
    tgts = np.array([0, 1, 2, 2])             # last one is wrong
    assert compute_top_k_accuracy(sims, tgts, k=1) == 0.75
    assert compute_top_k_accuracy(sims, tgts, k=2) == 1.0


def test_mean_average_precision_range():
    sims = np.random.RandomState(0).rand(10, 5)
    tgts = np.random.RandomState(0).randint(0, 5, 10)
    mAP = compute_mean_average_precision(sims, tgts)
    assert 0.0 <= mAP <= 1.0


def test_confusion_matrix_row_sums_to_one_when_normalized_true():
    preds = np.array([0, 0, 1, 1, 2, 2])
    tgts = np.array([0, 1, 1, 1, 2, 0])
    cm = compute_confusion_matrix(preds, tgts, num_classes=3, normalize="true")
    # Rows sum to 1 (or 0 if class is absent)
    sums = cm.sum(axis=1)
    for s in sums:
        assert pytest.approx(s, abs=1e-6) == 1.0 or s == 0


def test_per_class_accuracy_keys_match_names():
    preds = np.array([0, 1, 1])
    tgts = np.array([0, 1, 0])
    pca = compute_per_class_accuracy(preds, tgts, class_names=["A", "B"])
    assert "A" in pca and "B" in pca
    assert pca["A"] == 0.5
    assert pca["B"] == 1.0
