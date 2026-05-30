# ZSSLR — Multimodal Zero-Shot Word-Level Sign Language Recognition

> Reference implementation for *“Word-Level Isolated Zero-Shot Sign Language Recognition via Frozen Human-Centric Foundation Models.”*
> **Sapiens-1B** (RGB) + a lightweight **Temporal Transformer** + **MotionBERT** (skeleton) + **BERT-base** (text), trained with a symmetric InfoNCE loss for zero-shot recognition of unseen sign glosses. Evaluated on a stratified, signer-disjoint zero-shot split of **WLASL**, with strong **CLIP** and **SignCLIP** baselines under an identical contrastive protocol.

**Paper:** Word-Level Isolated Zero-Shot Sign Language Recognition via Frozen Human-Centric Foundation Models · **Repo:** <https://github.com/Lala2398/ZSSLR>

---

## Headline results (WLASL, 500 unseen glosses; mean ± std over 3 seeds)

| Model | Top-1 (%) | Top-5 (%) | mAP (%) | H-mean (%) |
|---|---|---|---|---|
| **Ours (Sapiens + Temporal Transf. + MotionBERT + prompt ens.)** | **35.2 ± 0.4** | **57.4 ± 0.5** | **41.0 ± 0.4** | **28.6 ± 0.5** |
| SignCLIP (visual + text enc.) | 34.1 ± 0.4 | 56.0 ± 0.5 | 39.5 ± 0.4 | 27.3 ± 0.5 |
| Full CLIP (ViT-L/14 + CLIP text) | 31.8 ± 0.4 | 53.6 ± 0.5 | 37.1 ± 0.5 | 25.7 ± 0.5 |
| CLIP visual-swap (ViT-L/14 + BERT text) | 30.1 ± 0.5 | 52.0 ± 0.6 | 35.4 ± 0.5 | 24.3 ± 0.6 |

The full model leads all baselines. The skeleton-motion stream of MotionBERT supplies the final **+1.1 pp** margin over sign-specific SignCLIP pretraining, and the Temporal Transformer adds **+1.8 pp** over plain mean-pooling (35.2 vs. 33.4).

---

## What this repository contains

This is a **reference / starter repository**. WLASL videos are publicly distributable via the [official WLASL release](https://github.com/dxli94/WLASL), but the official **Sapiens-1B** and **MotionBERT** checkpoints require separate registration/download and are not redistributed here. What we *do* provide:

1. **Reference implementation** of the model exactly as described in the paper — frozen Sapiens-1B + MotionBERT encoders, a trainable 1-layer/8-head Temporal Transformer for visual aggregation, a trainable BERT-base text encoder, a symmetric InfoNCE loss, and a 3-template prompt ensemble.
2. **The CLIP and SignCLIP baselines**, implemented under the *identical* contrastive protocol so the comparison isolates the effect of pretraining domain and the skeleton-motion stream.
3. **A smoke-test pipeline** that runs end-to-end on synthetic dummy data — verifies the code is correct without requiring the dataset or pretrained weights.
4. **Two evaluation protocols** — traditional ZSL (unseen search space) and Generalized ZSL (full-vocabulary search space, reporting `Au`, `As`, and `H-mean`).
5. **Reproducible figures and tables** generated from saved `.npz` outputs; when no checkpoint is available the notebooks fall back to the numbers reported in the paper (clearly labelled).

> **Paper-faithful numbers** — see Section IV of the paper. We do not fabricate experimental results; every figure flags whether it uses real model outputs or reported numbers.

---

## Repository layout

```
zsslr/
├── README.md                          # this file
├── LICENSE                            # MIT
├── setup.py                           # pip install -e .
├── requirements.txt
├── configs/
│   ├── default.yaml                   # full model — single source of truth for hyperparameters
│   ├── clip_baseline.yaml             # ViT-L/14 + CLIP text encoder
│   └── signclip_baseline.yaml         # SignCLIP visual + text encoder
├── scripts/
│   ├── setup_environment.sh           # one-shot venv + deps installer
│   ├── extract_skeletons.py           # MediaPipe Holistic → 17-joint H36M .npy per video
│   ├── build_split.py                 # stratified, signer-disjoint zero-shot split + manifest
│   ├── train.py                       # main training entry point (full / clip / signclip)
│   ├── evaluate.py                    # ZSL + GZSL evaluation, saves .npz
│   ├── noise_robustness.py            # Gaussian skeleton-noise sweep (Table II)
│   └── smoke_test.py                  # end-to-end test on synthetic data
├── src/
│   ├── models/
│   │   ├── sapiens_encoder.py         # frozen Sapiens-1B (with CLIP-ViT fallback)
│   │   ├── temporal_transformer.py    # trainable 1-layer, 8-head temporal aggregator
│   │   ├── motionbert_encoder.py      # frozen MotionBERT DSTformer (with proxy fallback)
│   │   ├── bert_encoder.py            # trainable BERT-base + prompt ensemble (batched)
│   │   ├── clip_encoder.py            # CLIP ViT-L/14 visual + CLIP text (baseline)
│   │   ├── signclip_encoder.py        # SignCLIP visual + text (baseline)
│   │   └── multimodal_zsl.py          # the full model: fusion + InfoNCE
│   ├── losses/
│   │   └── contrastive_losses.py      # SymmetricInfoNCE (single source, reused)
│   ├── data/
│   │   ├── dataset.py                 # SignLanguageDataset + ZeroShotSignDataset
│   │   ├── preprocessing.py           # VideoPreprocessor, SkeletonExtractor, prompts
│   │   └── splits.py                  # build / load seen-unseen gloss splits + manifest hash
│   ├── evaluation/
│   │   └── evaluator.py               # ZeroShotEvaluator (ZSL + GZSL)
│   └── utils/
│       ├── metrics.py                 # top-k, mAP, H-mean, confusion matrix
│       ├── checkpoint.py              # robust load (handles 4 key conventions)
│       ├── logging_utils.py           # logger + W&B + TensorBoard
│       └── visualization.py           # figure generators (real-data only)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # inspect WLASL splits and descriptions
│   ├── 02_visualizations.ipynb        # paper figures (real-data OR reported)
│   ├── 03_ablation_analysis.ipynb     # ablation tables and plots (Table III)
│   └── 04_sensitivity_analysis.ipynb  # τ / batch-size heatmap (Table IV), noise curve (Table II)
└── tests/
    ├── test_models.py                 # shapes, frozen-param check, temporal aggregation
    ├── test_dataset.py                # description-key matching, collate
    └── test_losses.py                 # symmetric InfoNCE behaviour
```

---

## Quick start

### 1. Environment

```bash
git clone https://github.com/Lala2398/ZSSLR
cd ZSSLR
bash scripts/setup_environment.sh
# OR manually:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Verify the code with synthetic data (no dataset needed)

```bash
python scripts/smoke_test.py
```

This runs the entire pipeline — encoders, temporal aggregation, fusion, InfoNCE loss, and both ZSL and GZSL inference — on randomly generated tensors. If this passes, your environment is correct. **Expected runtime: under 60 s on CPU.**

### 3. Prepare your data

Expected layout:

```
data/
└── wlasl/
    ├── videos/{gloss}/{video_id}.mp4
    ├── skeletons/{gloss}/{video_id}.npy        # (T, 17, 2); see step 5
    ├── descriptions.json     {gloss: "English text description"}
    └── splits/
        ├── seen_glosses.txt        # 1,500 glosses, one per line
        ├── unseen_glosses.txt      # 500 glosses
        ├── signer_map.json         # {video_id: signer_id}
        └── manifest.json           # gloss assignments + corpus hash (for leakage checks)
```

`descriptions.json` is keyed by **gloss**, not by individual video — every video of the same gloss shares the same description. Descriptions are short definitions from WordNet / ASL dictionaries, e.g. `"The sign for book. A set of written pages fastened together."`

### 4. Build the zero-shot split

```bash
python scripts/build_split.py \
    --videos_dir data/wlasl/videos \
    --num_seen 1500 --num_unseen 500 \
    --min_videos_per_gloss 5 \
    --seed 42 \
    --output_dir data/wlasl/splits
```

Glosses are bucketed by video frequency and the unseen set is sampled proportionally per bucket, so the unseen split matches the full-vocabulary frequency profile. Signers are partitioned **disjointly** across train / val / test-seen. WordNet synsets are used to keep near-synonyms on the same side of the split. The result is written to `manifest.json` with a corpus hash so anyone can reproduce or audit it.

### 5. Extract skeletons (one-time, slow)

```bash
python scripts/extract_skeletons.py \
    --video_dir  data/wlasl/videos \
    --output_dir data/wlasl/skeletons
```

Uses MediaPipe Holistic to produce 2D landmarks, then maps them to a **17-joint H36M-compatible skeleton** as required by MotionBERT. Output is a `(T, 17, 2)` array per clip (T = 32 frames, uniformly sampled at 30 fps; looping/padding as needed; frames resized to 224×224).

### 6. Train

```bash
# Full model (default)
python scripts/train.py --config configs/default.yaml
# with overrides:
python scripts/train.py --config configs/default.yaml --batch_size 64 --temperature 0.07 --use_wandb

# Baselines, identical contrastive protocol:
python scripts/train.py --config configs/clip_baseline.yaml
python scripts/train.py --config configs/signclip_baseline.yaml
```

Only the **BERT text encoder, the Temporal Transformer, and the linear projection / fusion heads** are updated. **Sapiens-1B and MotionBERT stay frozen** — this is the paper’s configuration and isolates the contribution of the pretrained representations. (For the CLIP / SignCLIP baselines, both of their encoders are frozen and only the projection heads + Temporal Transformer are trained.)

Default training: AdamW (`lr=1e-4`, `weight_decay=1e-5`), 100 epochs, batch size 64, temperature τ = 0.07, on a single NVIDIA A100 (80 GB). Early stopping on the signer-disjoint validation set. Each experiment is repeated with 3 random seeds.

### 7. Evaluate (ZSL + GZSL) and save embeddings

```bash
python scripts/evaluate.py \
    --checkpoint     outputs/<exp>/checkpoints/best.pt \
    --video_dir      data/wlasl/videos \
    --skeleton_dir   data/wlasl/skeletons \
    --descriptions   data/wlasl/descriptions.json \
    --splits_dir     data/wlasl/splits \
    --protocol       both \
    --output_dir     outputs/evaluation
```

This writes `embeddings.npz`, `predictions.npz`, and `metrics.json` (top-1/top-5/mAP for ZSL, and `Au` / `As` / `H-mean` for GZSL) — exactly the files the visualization notebook expects.

### 8. Generate paper figures

```bash
jupyter notebook notebooks/02_visualizations.ipynb
```

Each figure either reads `outputs/evaluation/*.npz` (real data) or, if no `.npz` is found, plots the numbers reported in the paper — and labels the figure accordingly so reviewers can tell at a glance which is which.

---

## Evaluation protocols

- **Traditional ZSL.** A test video from the unseen set is matched against the **unseen** gloss text embeddings only: `ĝ = argmax_{g∈G_u} z_v · z_t(g)`. Reported as top-1, top-5, mAP.
- **Generalized ZSL (GZSL).** Retrieval is performed over the **full** 2,000-class vocabulary. We report accuracy on unseen videos `Au`, accuracy on held-out-signer seen videos `As`, and their harmonic mean `H-mean = 2·As·Au/(As+Au)`, which penalises models biased toward either split.

---

## Baselines

| Baseline | Visual | Text | Trained components |
|---|---|---|---|
| **CLIP visual-swap** | CLIP ViT-L/14 (frozen) | BERT-base | Temporal Transf. + projections |
| **Full CLIP** | CLIP ViT-L/14 (frozen) | CLIP text (frozen) | Temporal Transf. + projections |
| **SignCLIP** | SignCLIP visual (frozen) | SignCLIP text (frozen) | Temporal Transf. + projections |

All three share our fusion head, InfoNCE loss (τ = 0.07, B = 64), prompt ensemble, and zero-shot inference protocol, so any difference is attributable to the visual/text backbone choice. Note SignCLIP is pretrained on **continuous, sentence-level** signing, whereas this task is **isolated, word-level** — a domain mismatch quantified directly in the paper.

---

## Ablations (Table III, 500 unseen glosses, mean ± std over 3 seeds)

| Variant | Top-1 (%) | Top-5 (%) | mAP (%) | H-mean (%) |
|---|---|---|---|---|
| Full model | 35.2 ± 0.4 | 57.4 ± 0.5 | 41.0 ± 0.4 | 28.6 ± 0.5 |
| Mean-pool visual (no Temporal Transf.) | 33.4 ± 0.5 | 55.8 ± 0.6 | 39.2 ± 0.5 | 27.1 ± 0.6 |
| w/o skeleton encoder | 28.2 ± 0.6 | 49.3 ± 0.7 | 33.1 ± 0.6 | 22.7 ± 0.7 |
| w/o visual encoder | 25.1 ± 0.7 | 46.0 ± 0.8 | 30.5 ± 0.7 | 20.4 ± 0.8 |
| ST-GCN instead of MotionBERT | 27.5 ± 0.6 | 48.7 ± 0.7 | 32.0 ± 0.6 | 22.2 ± 0.7 |
| Single prompt (no ensemble) | 29.7 ± 0.5 | 51.2 ± 0.6 | 34.5 ± 0.5 | 23.9 ± 0.6 |

Skeleton-noise robustness (Table II): top-1 degrades gracefully to σ = 5 px (−1.5 pp), then −4.3 pp at σ = 10 px and −9.8 pp at σ = 20 px. Temperature/batch sweep (Table IV): τ = 0.07, B = 64 is optimal under the A100 memory budget.

---

## Checkpoint compatibility

`scripts/evaluate.py` and `src/utils/checkpoint.py` handle four checkpoint conventions:

```python
{"model_state_dict": state_dict}   # this repo
{"model_state":      state_dict}   # earlier repo iterations
{"model":            state_dict}   # some HuggingFace exports
state_dict                         # bare torch.save(model.state_dict())
```

`module.` prefixes from Distributed Data Parallel are stripped automatically.

---

## Notes on faithfulness to the paper

- **Sapiens-1B fallback.** The official Sapiens checkpoint requires registration. If `checkpoints/sapiens_1b.pt` is not found, the code falls back to `vit_huge_patch14_clip_224` from `timm`. The fallback is documented in `src/models/sapiens_encoder.py` and yields lower zero-shot accuracy than the official Sapiens.
- **MotionBERT fallback.** Similar — if the official DSTformer checkpoint (depth = 5, d = 512) is missing, an architecturally compatible but untrained DSTformer is constructed; smoke tests pass but real performance requires the official checkpoint.
- **Temporal Transformer.** The T = 32 frame-level Sapiens `[CLS]` tokens are treated as a sequence and fed to a single-layer, 8-head Transformer encoder with a learnable positional embedding; the position-0 output is the video-level embedding `v ∈ R¹⁰²⁴`. Fewer than 5M trainable parameters; only this module (not Sapiens) is trained. Replacing mean-pooling with it gives +1.8 pp top-1.
- **17 H36M joints.** Skeletons are mapped to the 17-joint H36M layout required by MotionBERT and stored as `(T, 17, 2)`.
- **3-template prompt ensemble.** `"A person signing the word {gloss}."`, `"The ASL sign for {gloss}."`, `"{gloss}. {description}"`. The final text embedding is the mean of the per-template `[CLS]` embeddings, computed in a **single batched forward pass** (3N tokens per batch), then projected `768 → 512` with L2-norm. The ensemble adds +5.5 pp over a single template.
- **Symmetric InfoNCE** (τ = 0.07) is implemented once in `src/losses/contrastive_losses.py` and reused — not duplicated — in `src/models/multimodal_zsl.py` and in both baselines.
- **Dimensions.** `v ∈ R¹⁰²⁴`, `m ∈ R⁵¹²`, `t ∈ R⁷⁶⁸`; fusion concatenates `[v; m] ∈ R¹⁵³⁶ → 512` (ReLU → Linear → L2-norm); `z_v, z_t ∈ R⁵¹²`.

---

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@inproceedings{zsslr_2026,
  title     = {Word-Level Isolated Zero-Shot Sign Language Recognition
               via Frozen Human-Centric Foundation Models},
  author    = {Alishzade Nigar, Ibadullayeva Lala, Iskandarli Rajab,
               Babayev Jabrayil, Mammadov Elnur, Elekberli Mehemmed,
               Hasanli Yusif and Aliyev Elvin},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2026},
  note      = {Code: https://github.com/Lala2398/ZSSLR}
}
```
