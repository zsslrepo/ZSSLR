"""Unit tests for the symmetric InfoNCE loss."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import SymmetricInfoNCE


def _random_normed(B: int = 16, D: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(B, D, generator=g)
    t = torch.randn(B, D, generator=g)
    return F.normalize(v, dim=-1), F.normalize(t, dim=-1)


def test_loss_finite_and_positive():
    v, t = _random_normed(B=16, D=32)
    loss_fn = SymmetricInfoNCE(temperature=0.07, learnable_temp=True)
    loss, metrics = loss_fn(v, t)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert 0.0 <= metrics["acc"] <= 1.0


def test_loss_lower_when_aligned():
    """If text equals visual, loss is at its minimum (very small)."""
    v, _ = _random_normed(B=8, D=32)
    t = v.clone()
    loss_fn = SymmetricInfoNCE(temperature=0.07, learnable_temp=False)
    loss_aligned, _ = loss_fn(v, t)

    v2, t2 = _random_normed(B=8, D=32, seed=7)
    loss_random, _ = loss_fn(v2, t2)
    assert loss_aligned.item() < loss_random.item()


def test_loss_temperature_learnable_grad():
    v, t = _random_normed()
    loss_fn = SymmetricInfoNCE(temperature=0.07, learnable_temp=True)
    loss, _ = loss_fn(v, t)
    loss.backward()
    assert loss_fn.logit_scale.grad is not None
    assert torch.isfinite(loss_fn.logit_scale.grad)


def test_loss_rejects_mismatched_shapes():
    loss_fn = SymmetricInfoNCE()
    v = torch.randn(4, 16)
    t = torch.randn(5, 16)
    with pytest.raises(ValueError):
        loss_fn(v, t)


def test_loss_rejects_batch_of_one():
    loss_fn = SymmetricInfoNCE()
    v, t = _random_normed(B=1, D=16)
    with pytest.raises(ValueError):
        loss_fn(v, t)
