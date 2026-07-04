"""Tests for the model components.

We use very small dimensions (tiny ViT, 64-dim embeddings) so the test
suite runs in seconds on CPU.
"""

import pytest
import torch

# transformers / timm pulls in heavy imports — guard so the suite still runs
# in environments where some are unavailable.
timm = pytest.importorskip("timm")
transformers = pytest.importorskip("transformers")

from src.losses import SymmetricInfoNCE
from src.models import MultimodalZSLModel


def _build_tiny_model():
    sapiens_cfg = dict(
        model_name="sapiens_0.3b", pretrained_path=None,
        input_size=32, num_frames=4, temporal_layers=1, temporal_heads=4,
        freeze=True, chunk_size=2,
    )
    motionbert_cfg = dict(
        num_joints=17, embed_dim=32, num_layers=1,
        pretrained_path=None, joint_mask_ratio=0.15, joint_noise_std=0.02,
        freeze=True,
    )
    azbert_cfg = dict(
        model_name="language-ml-lab/AzerBert",
        max_length=16, num_prompt_templates=2, freeze=False,
    )
    return MultimodalZSLModel(
        sapiens_config=sapiens_cfg,
        motionbert_config=motionbert_cfg,
        azbert_config=azbert_cfg,
        embedding_dim=32,
        visual_dim=32,
        motion_dim=32,
        temperature=0.07,
    )


def test_model_builds():
    m = _build_tiny_model()
    counts = m.count_parameters()
    assert counts["trainable"] > 0
    assert counts["sapiens_trainable"] < counts["sapiens_total"]
    assert counts["motionbert_trainable"] < counts["motionbert_total"]


def test_visual_embeddings_are_l2_normed():
    m = _build_tiny_model().eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    skeleton = torch.rand(2, 4, 17, 2)
    v = m.encode_visual(frames, skeleton)
    norms = v.norm(dim=-1)
    assert v.shape == (2, 32)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_text_embeddings_shape_with_prompt_ensemble():
    m = _build_tiny_model().eval()
    descs = ["Baş barmaq alına toxunur.", "İki əl açılır."]
    t = m.encode_text(descs)
    assert t.shape == (2, 32)
    norms = t.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_loss_is_finite_and_backprops():
    m = _build_tiny_model().train()
    frames = torch.randn(3, 4, 3, 32, 32)
    skeleton = torch.rand(3, 4, 17, 2)
    descriptions = [
        "Baş barmaq alına toxunur.",
        "İki əl ev şəklində birləşir.",
        "Sağ əl üzə yaxın hərəkət edir.",
    ]
    loss, metrics = m(frames, descriptions, skeleton)
    assert torch.isfinite(loss)
    loss.backward()
    # azBERT must receive grads
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in m.azbert.parameters()
        if p.requires_grad
    )


def test_zero_shot_classify_runs_eval_mode_independent():
    m = _build_tiny_model()
    m.train()
    frames = torch.randn(2, 4, 3, 32, 32)
    skeleton = torch.rand(2, 4, 17, 2)
    classes = {"a": "Baş barmaq alına toxunur.", "b": "İki əl açılır."}
    out = m.zero_shot_classify(frames, classes, skeleton)
    assert out["similarities"].shape == (2, 2)
    assert set(out["pred_classes"]).issubset({"a", "b"})
    # zero_shot_classify must restore train mode
    assert m.training is True


def test_visual_encoder_freezing():
    m = _build_tiny_model()
    # No grads on backbone params
    assert not any(p.requires_grad for p in m.sapiens.backbone.parameters())
    assert not any(p.requires_grad for p in m.motionbert.backbone.parameters())
    # The projection heads ARE trainable
    assert m.sapiens.frame_proj.weight.requires_grad
    # Temporal Transformer is trainable
    assert any(p.requires_grad for p in m.sapiens.temporal_encoder.parameters())
    # Fusion head is trainable
    assert any(p.requires_grad for p in m.fusion.parameters())
    assert m.motionbert.projection[0].weight.requires_grad
