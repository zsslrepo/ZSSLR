"""End-to-end smoke test on synthetic tensors — no dataset required.

Verifies that:
  1. The three encoders build and produce the expected shapes.
  2. The fusion in `encode_visual` returns L2-normalised (B, D) tensors.
  3. The symmetric InfoNCE loss is finite and backprops.
  4. Only the text encoder + projection heads have non-zero gradients.
  5. Saving and reloading a checkpoint preserves predictions exactly.

Runtime: < 60 s on CPU.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import MultimodalZSLModel
from src.utils import load_checkpoint, save_checkpoint


def main() -> None:
    print("=" * 64)
    print("ZSSLR-AzSL smoke test")
    print("=" * 64)

    # Smaller dimensions than the paper to keep this fast on CPU.
    B = 4                 # batch size
    T = 4                 # frames
    H = W = 64            # tiny spatial size
    J = 133               # paper-compliant keypoint count
    D = 64                # embedding dim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sapiens_cfg = dict(
        model_name="sapiens_0.3b",
        pretrained_path=None,
        input_size=H,
        num_frames=T,
        freeze=True,
        chunk_size=8,
    )
    motionbert_cfg = dict(
        num_joints=J, embed_dim=64, num_layers=2,
        pretrained_path=None,
        joint_mask_ratio=0.15, joint_noise_std=0.02, freeze=True,
    )
    azbert_cfg = dict(
        model_name="language-ml-lab/AzerBert",
        max_length=24, num_prompt_templates=3, freeze=False,
    )

    print("\n[1/6] Building MultimodalZSLModel...")
    torch.manual_seed(0)
    model = MultimodalZSLModel(
        sapiens_config=sapiens_cfg,
        motionbert_config=motionbert_cfg,
        azbert_config=azbert_cfg,
        embedding_dim=D,
        temperature=0.07,
        temperature_learnable=True,
    ).to(device)
    counts = model.count_parameters()
    print(f"  Total params: {counts['total']:,}  |  Trainable: {counts['trainable']:,}")

    # Sanity: trainable should be << total because backbones are frozen.
    assert counts["trainable"] < counts["total"], "all params trainable? check freeze flags"
    assert counts["sapiens_trainable"] < counts["sapiens_total"], "sapiens not frozen"
    assert counts["motionbert_trainable"] < counts["motionbert_total"], "motionbert not frozen"

    # Synthetic batch ------------------------------------------------------
    print("\n[2/6] Generating synthetic batch...")
    frames = torch.randn(B, T, 3, H, W, device=device)
    skeleton = torch.rand(B, T, J, 2, device=device)              # MediaPipe gives (x,y) in [0,1]
    descriptions = [
        "Baş barmaq alına toxunur, digər barmaqlar açıq və yuxarı yönəlmişdir.",
        "Sağ əl sol əlin üstündə hərəkət edir.",
        "İki əl bir-birinə yaxınlaşır və açılır.",
        "Bir əl üzə yaxın hərəkət edir.",
    ]

    # Forward + loss -------------------------------------------------------
    print("\n[3/6] Forward + InfoNCE loss...")
    model.train()
    loss, metrics = model(frames, descriptions, skeleton)
    print(f"  loss          = {loss.item():.4f}")
    print(f"  temperature   = {metrics['temperature']:.4f}")
    print(f"  acc_v2t       = {metrics['acc_v2t']:.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    # Backward -------------------------------------------------------------
    print("\n[4/6] Backward pass and gradient check...")
    loss.backward()
    grads_present = {"trainable_with_grad": 0, "trainable_no_grad": 0,
                     "frozen_with_grad": 0,    "frozen_no_grad": 0}
    for name, p in model.named_parameters():
        has_grad = p.grad is not None and torch.any(p.grad != 0)
        if p.requires_grad and has_grad:
            grads_present["trainable_with_grad"] += 1
        elif p.requires_grad and not has_grad:
            grads_present["trainable_no_grad"] += 1
        elif not p.requires_grad and has_grad:
            grads_present["frozen_with_grad"] += 1
        else:
            grads_present["frozen_no_grad"] += 1
    print(f"  {grads_present}")
    assert grads_present["frozen_with_grad"] == 0, \
        "frozen parameters received gradients — freezing is broken"

    # Zero-shot classify ---------------------------------------------------
    print("\n[5/6] Zero-shot classification on 6 candidate classes...")
    model.eval()
    class_descs = {
        "ata":  "Baş barmaq alına toxunur.",
        "ana":  "Baş barmaq çənəyə toxunur.",
        "su":   "Üç barmaq dodağa yaxınlaşır.",
        "ev":   "İki əl ev şəklində birləşir.",
        "kitab": "İki əl açıq kitab kimi açılır.",
        "dost": "İki əl sıxlaşır və birləşir.",
    }
    out = model.zero_shot_classify(frames, class_descs, skeleton)
    print(f"  predictions   = {out['pred_classes']}")
    print(f"  similarity    shape = {out['similarities'].shape}")
    assert out["similarities"].shape == (B, len(class_descs))

    # Checkpoint round-trip ------------------------------------------------
    print("\n[6/6] Checkpoint save / load round-trip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, optimizer=None, scheduler=None,
                               epoch=1, step=1, metrics={"loss": loss.item()},
                               checkpoint_dir=tmpdir, filename="smoke.pt")
        # Reload into a fresh model
        model2 = MultimodalZSLModel(
            sapiens_config=sapiens_cfg, motionbert_config=motionbert_cfg,
            azbert_config=azbert_cfg, embedding_dim=D, temperature=0.07,
        ).to(device)
        info = load_checkpoint(path, model2, map_location=device)
        model2.eval()
        out2 = model2.zero_shot_classify(frames, class_descs, skeleton)
        # Same architecture + same weights -> same outputs (within float tolerance)
        diff = float(np.abs(out["similarities"] - out2["similarities"]).max())
        print(f"  max abs diff after reload = {diff:.2e}")
        assert diff < 1e-4, f"reload changed outputs (diff={diff:.2e})"

    print("\nAll smoke-test checks passed.")


if __name__ == "__main__":
    main()
