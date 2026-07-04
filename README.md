# ZSSLR-AzSL — Multimodal Zero-Shot Word-Level Sign Language Recognition

> Reference implementation of a **frozen human-centric foundation-model** approach to
> word-level *zero-shot* sign language recognition, adapted to **Azerbaijani Sign
> Language (AzSL)**.
>
> **Sapiens-1B** (RGB) + a lightweight **Temporal Transformer** + **MotionBERT**
> (skeleton) + a trainable **AzerBert** text encoder, trained with a symmetric
> InfoNCE loss so the model can recognise sign glosses it has **never seen during
> training**. Data: the public **AzSLD *words200*** corpus.

**Repo:** <https://github.com/zsslrepo/ZSSLR/>

---

## What this repository is

This is a **research / reference repository** for a university scientific work. Its
goal is to be **honest, reproducible, and actually runnable** on the one real dataset
we have — AzSLD.

The method follows the paper *"Word-Level Isolated Zero-Shot Sign Language
Recognition via Frozen Human-Centric Foundation Models"*, but the **language and data
are Azerbaijani**, not English/WLASL. Concretely:

- **Visual stream** — frozen **Sapiens-1B** (ViT-Huge) encodes each of `T = 32`
  frames; a trainable **1-layer, 8-head Temporal Transformer** aggregates them into a
  single 1024-d video embedding.
- **Skeleton stream** — frozen **MotionBERT** (DSTformer, depth 5) encodes a
  **17-joint H36M** skeleton sequence into a 512-d motion embedding.
- **Text stream** — trainable **AzerBert** (`language-ml-lab/AzerBert`) with a
  **5-template Azerbaijani prompt ensemble** produces a 512-d gloss embedding.
- **Fusion** — `concat[v(1024); m(512)] → Linear(1536→512) → ReLU → Linear(512→512)
  → L2-norm` (paper Eq. 1).
- **Loss** — **symmetric InfoNCE**, fixed temperature `τ = 0.07` (paper Eq. 2).

Only the **text encoder, the Temporal Transformer, and the fusion / projection heads**
are trained. Sapiens-1B and MotionBERT stay **frozen**.

> **No fabricated numbers.** This README contains no results table. Reported accuracy
> requires the official Sapiens-1B / MotionBERT checkpoints and a GPU (see
> *Foundation-model weights* below). Run the pipeline on your machine and fill in your
> own measured numbers.

---

## Repository layout

Every path below exists in the repo (no placeholder / fictional files):

```
ZSSLR/
├── README.md                       # this file
├── LICENSE                         # MIT
├── setup.py                        # pip install -e .
├── requirements.txt
├── .gitignore                      # excludes videos/skeletons/checkpoints; keeps splits + descriptions
├── configs/
│   └── default.yaml                # single source of truth for hyperparameters (paper-faithful)
├── scripts/
│   ├── setup_environment.sh        # venv + deps installer
│   ├── build_split.py              # frequency-stratified, gloss-disjoint zero-shot split
│   ├── build_descriptions.py       # per-gloss Azerbaijani descriptions -> descriptions.json
│   ├── extract_skeletons.py        # MediaPipe Pose -> 17-joint H36M .npy per video
│   ├── train.py                    # training entry point
│   ├── evaluate.py                 # zero-shot (ZSL / GZSL) evaluation
│   └── smoke_test.py               # end-to-end test on synthetic data (CPU)
├── src/
│   ├── models/
│   │   ├── sapiens_encoder.py      # frozen Sapiens-1B + trainable Temporal Transformer (timm fallback)
│   │   ├── motionbert_encoder.py   # frozen MotionBERT DSTformer, 17-joint (untrained fallback)
│   │   ├── azbert_encoder.py       # trainable AzerBert + 5-template Azerbaijani prompt ensemble
│   │   └── multimodal_zsl.py       # full model: concat-MLP fusion + InfoNCE
│   ├── losses/
│   │   └── contrastive_losses.py   # SymmetricInfoNCE (single source, reused)
│   ├── data/
│   │   ├── dataset.py              # SignLanguageDataset + ZeroShotSignDataset
│   │   ├── preprocessing.py        # VideoPreprocessor, SkeletonExtractor (17-joint), prompts
│   │   └── splits.py               # build / load seen-unseen gloss splits
│   ├── evaluation/
│   │   └── evaluator.py            # ZeroShotEvaluator (ZSL + GZSL)
│   └── utils/
│       ├── metrics.py              # top-k, mAP, H-mean
│       ├── checkpoint.py           # robust checkpoint loading
│       ├── logging_utils.py        # logger + optional W&B / TensorBoard
│       └── visualization.py        # figure generators
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_visualizations.ipynb
│   └── 03_ablation_analysis.ipynb
├── data/
│   └── azsld/
│       ├── descriptions.json       # committed (200 glosses)
│       └── splits/                 # committed: seen/unseen_glosses.txt + split_manifest.json
└── tests/
    ├── test_models.py              # shapes, frozen-param check, temporal aggregation
    ├── test_dataset.py             # split filtering, collate
    ├── test_losses.py              # symmetric InfoNCE behaviour
    └── test_metrics.py             # top-k / mAP / H-mean
```

---

## Dataset — AzSLD *words200*

- **Source:** AzSLD (Azerbaijani Sign Language Dataset), *words200* subset — 200
  Azerbaijani word glosses, ~8,500 videos. Public release on Zenodo
  (record `14222948`).
- **Layout expected by the code:**

```
data/
└── azsld/
    ├── videos/{GLOSS}/{video_id}.mp4          # link/copy AzSLD words200 here
    ├── skeletons/{GLOSS}/{video_id}.npy       # (T, 17, 2); produced by extract_skeletons.py
    ├── descriptions.json                      # {gloss: "short Azerbaijani description"}
    └── splits/
        ├── seen_glosses.txt                   # 150 glosses, one per line
        ├── unseen_glosses.txt                 #  50 glosses
        └── split_manifest.json                # gloss assignments + counts (audit)
```

`descriptions.json` is keyed by **gloss** (not per video) — every video of a gloss
shares its description. The splits and descriptions are **committed** to the repo so
the zero-shot partition is reproducible; the large binaries (videos, skeletons) are
**not** committed (see `.gitignore`).

The zero-shot split is **gloss-disjoint** (150 seen / 50 unseen) and
**frequency-stratified**: glosses are bucketed by video count and the unseen set is
sampled proportionally, so it matches the corpus frequency profile. AzSLD *words200*
carries **no signer identity** (`cam: null`), so a signer-disjoint split is not
possible; validation is instead a **video-disjoint** held-out fraction of the seen
set (used only for early stopping).

---

## Quick start

### 1. Environment

```bash
git clone https://github.com/zsslrepo/ZSSLR/
cd ZSSLR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Verify the code with synthetic data (no dataset, no weights needed)

```bash
python scripts/smoke_test.py
```

Runs the whole pipeline — encoders, temporal aggregation, concat-MLP fusion, InfoNCE
loss, and zero-shot inference — on random tensors with tiny dimensions. If it passes,
your environment and the architecture wiring are correct. Runs on CPU.

Unit tests:

```bash
pytest tests/
```

### 3. Point the code at the AzSLD videos

Link (or copy) the AzSLD *words200* video tree so each gloss is a sub-directory:

```bash
mkdir -p data/azsld
ln -s /path/to/AzSLD_Words_200 data/azsld/videos
```

### 4. Build the zero-shot split

```bash
python scripts/build_split.py \
    --video_dir data/azsld/videos \
    --num_unseen 50 \
    --seed 42 \
    --output_dir data/azsld/splits
```

Writes `seen_glosses.txt`, `unseen_glosses.txt`, and `split_manifest.json`
(frequency-stratified, gloss-disjoint). The seen count is simply *total − unseen*
(≈150 for words200). Pass `--manifest data/azsld/manifest.json` instead of
`--video_dir` to build from an AzSLD-style manifest.

### 5. Build gloss descriptions

```bash
python scripts/build_descriptions.py \
    --video_dir data/azsld/videos \
    --output data/azsld/descriptions.json
    # optional: --override my_manual_descriptions.json
```

The gloss list is taken from `--video_dir` (all glosses discovered on disk); you can
instead pass `--manifest data/azsld/manifest.json` or `--glosses_file glosses.txt`.

By default the description of a gloss is the normalised gloss word itself; supply a
manual dictionary via `--override` to improve text quality (descriptions affect
accuracy).

### 6. Extract skeletons (one-time, slow)

```bash
python scripts/extract_skeletons.py \
    --video_dir  data/azsld/videos \
    --output_dir data/azsld/skeletons
```

Uses MediaPipe Pose, then maps landmarks to a **17-joint H36M-compatible** skeleton
as required by MotionBERT. Output is `(T, 17, 2)` per clip.

### 7. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Default: AdamW (`lr = 1e-4`, `weight_decay = 1e-5`), 100 epochs, batch size 64,
`τ = 0.07`, `T = 32` frames. Early stopping on the video-disjoint validation split.
Sapiens-1B and MotionBERT are frozen.

### 8. Evaluate (zero-shot on unseen glosses)

```bash
python scripts/evaluate.py \
    --checkpoint     outputs/<exp>/checkpoints/best.pt \
    --config         configs/default.yaml
```

Writes embeddings, predictions, and `metrics.json` (top-1 / top-5 / top-10, mAP) for
the unseen-only zero-shot protocol; set the evaluation split to `mixed` for GZSL.

---

## Foundation-model weights (important)

Reported paper-level accuracy requires the **official** foundation-model checkpoints
and a GPU:

- **Sapiens-1B.** If `checkpoints/sapiens_1b.pt` is missing, `sapiens_encoder.py`
  falls back to a `timm` ViT-Huge backbone (random / ImageNet-CLIP init). The
  pipeline runs end-to-end and smoke tests pass, but zero-shot accuracy is far below
  the official Sapiens.
- **MotionBERT.** If the official DSTformer checkpoint (depth 5, d = 512) is missing,
  an architecturally compatible but **untrained** DSTformer is built — smoke tests
  pass, real performance needs the official weights.

The same code scales up unchanged once the official checkpoints and a GPU are
available.

---

## Method details

- **Temporal Transformer.** The `T = 32` frame embeddings from Sapiens are treated as
  a sequence, given a learnable positional embedding, and passed through a single
  1-layer / 8-head Transformer encoder (pre-norm, GELU). The position-0 output is the
  video embedding `v ∈ R¹⁰²⁴`. Only this module (not Sapiens) is trained.
- **17 H36M joints.** Skeletons are mapped to the 17-joint H36M layout MotionBERT
  expects and stored as `(T, 17, 2)`.
- **Azerbaijani prompt ensemble.** 5 templates around each gloss/description; the text
  embedding is the mean of the per-template `[CLS]` embeddings, projected to 512-d and
  L2-normalised.
- **Symmetric InfoNCE** (`τ = 0.07`, fixed) is defined once in
  `src/losses/contrastive_losses.py` and reused by the model.
- **Dimensions.** `v ∈ R¹⁰²⁴`, `m ∈ R⁵¹²`; fusion `[v; m] ∈ R¹⁵³⁶ → 512`;
  `z_v, z_t ∈ R⁵¹²`.

---

## Limitations

- Results depend on the official frozen checkpoints; the fallbacks exist only so the
  code is runnable and testable without them.
- AzSLD *words200* has no signer metadata, so the split is gloss-disjoint /
  video-disjoint rather than signer-disjoint.
- Gloss descriptions are auto-generated by default; richer, hand-written Azerbaijani
  descriptions are expected to improve the text side.

---

## License

MIT — see `LICENSE`.
