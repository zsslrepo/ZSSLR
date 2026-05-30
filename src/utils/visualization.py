"""All figures for the paper, generated from real model outputs OR from
the numbers reported in the paper (clearly labelled in the figure title).

Two ways to use this module:

1.  After running `scripts/evaluate.py`, call:
        FigureGenerator.from_npz('outputs/evaluation').generate_all()
    This produces 8 figures purely from real arrays in
    `embeddings.npz` / `predictions.npz`.

2.  Without a checkpoint, call:
        ReportedFigures().generate_all('outputs/figures')
    This reproduces the figures using the exact numbers from
    Tables I-III and Sec. V of the paper. Every such figure has a
    “Source: paper Table X” caption so reviewers can verify.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared design tokens (IEEE-paper-friendly palette).
# ----------------------------------------------------------------------
PALETTE = {
    "primary":  "#1f4e79",
    "accent":   "#c0504d",
    "success":  "#2ecc71",
    "warning":  "#f59d56",
    "muted":    "#7f7f7f",
    "purple":   "#8064a2",
    "teal":     "#4bacc6",
    "lgray":    "#d9d9d9",
}

_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "lines.linewidth":    2.2,
}


def _apply_rc():
    plt.rcParams.update(_RC)


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved figure: %s.{pdf,png}", name)


# ======================================================================
# 1) Real-data figure generator.
# ======================================================================
class FigureGenerator:
    """Generates 8 paper figures from real `.npz` files."""

    def __init__(
        self,
        data: Dict,
        signer_map: Optional[Dict[str, str]] = None,
        output_dir: Union[str, Path] = "outputs/figures",
    ):
        _apply_rc()
        self.d = data
        self.signer_map = signer_map or {}
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.N = self.d["vis_embeds"].shape[0]
        self.C = self.d["text_embeds"].shape[0]
        logger.info(
            "FigureGenerator initialised: N=%d samples, C=%d classes",
            self.N, self.C,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_npz(
        cls,
        eval_dir: Union[str, Path],
        signer_map_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "FigureGenerator":
        eval_dir = Path(eval_dir)
        emb = np.load(eval_dir / "embeddings.npz", allow_pickle=True)
        pred = np.load(eval_dir / "predictions.npz", allow_pickle=True)
        data = {
            "vis_embeds":   emb["visual"],
            "text_embeds":  emb["text"],
            "class_names":  emb["class_names"].tolist(),
            "similarities": pred["similarities"],
            "predictions":  pred["predictions"],
            "targets":      pred["targets"],
            "video_ids":    pred["video_ids"].tolist(),
            "glosses":      pred["glosses"].tolist() if "glosses" in pred.files else None,
        }
        signer_map = {}
        if signer_map_path is not None and Path(signer_map_path).exists():
            with open(signer_map_path, encoding="utf-8") as f:
                signer_map = json.load(f)
        return cls(
            data,
            signer_map=signer_map,
            output_dir=output_dir or (eval_dir / "figures"),
        )

    # ------------------------------------------------------------------
    def generate_all(self) -> None:
        self.fig_a_embedding_space()
        self.fig_b_similarity_matrix()
        self.fig_c_topk_bar()
        self.fig_d_confusion_matrix()
        if self.signer_map:
            self.fig_e_per_signer()
        self.fig_f_similarity_distributions()
        self.fig_g_modality_gap()
        self.fig_h_retrieval_ranks()

    # ------------------------------------------------------------------
    def _topk_acc(self, k: int) -> float:
        sims = self.d["similarities"]
        tgts = self.d["targets"]
        k = min(k, sims.shape[1])
        top_k = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        return float(np.any(top_k == tgts[:, None], axis=1).mean())

    # ------------------------------------------------------------------
    def fig_a_embedding_space(self) -> None:
        """Panel A: UMAP/t-SNE of visual + text embeddings."""
        vis = self.d["vis_embeds"]
        txt = self.d["text_embeds"]
        tgts = self.d["targets"]

        combined = np.vstack([vis, txt])
        try:
            import umap                                     # noqa: F401
            from umap import UMAP
            reducer = UMAP(
                n_components=2, random_state=42,
                metric="cosine",
                n_neighbors=min(15, max(2, combined.shape[0] - 1)),
                min_dist=0.1,
            )
            algo = "UMAP"
        except ImportError:
            from sklearn.manifold import TSNE
            perp = max(5, min(30, combined.shape[0] // 4))
            reducer = TSNE(
                n_components=2, random_state=42,
                metric="cosine", perplexity=perp, init="random",
            )
            algo = "t-SNE"

        emb2d = reducer.fit_transform(combined)
        vis2d, txt2d = emb2d[: self.N], emb2d[self.N:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel A1 — visual embeddings coloured by gloss class
        ax = axes[0]
        unique = np.unique(tgts)
        K = min(len(unique), 20)
        cmap = plt.cm.tab20(np.linspace(0, 1, K))
        for i, c in enumerate(unique[:K]):
            mask = tgts == c
            name = (self.d["class_names"][c]
                    if c < len(self.d["class_names"]) else str(c))
            ax.scatter(
                vis2d[mask, 0], vis2d[mask, 1],
                color=cmap[i % K], alpha=0.6, s=18,
                label=name, rasterized=True,
            )
        ax.set_title(f"A   {algo} — Visual Embeddings\n(coloured by gloss)")
        ax.set_xticks([]); ax.set_yticks([])
        if K <= 12:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

        # Panel A2 — modality gap
        ax2 = axes[1]
        n_lines = min(80, self.N)
        idx = np.random.default_rng(0).choice(self.N, n_lines, replace=False)
        for vi in idx:
            ti = int(self.d["targets"][vi])
            if ti < len(txt2d):
                ax2.plot(
                    [vis2d[vi, 0], txt2d[ti, 0]],
                    [vis2d[vi, 1], txt2d[ti, 1]],
                    color="#ccc", lw=0.5, alpha=0.4, zorder=1,
                )
        ax2.scatter(vis2d[:, 0], vis2d[:, 1], c=PALETTE["primary"],
                    alpha=0.4, s=14, label="Visual", rasterized=True, zorder=2)
        ax2.scatter(txt2d[:, 0], txt2d[:, 1], c=PALETTE["accent"],
                    alpha=0.85, s=30, marker="s", label="Text",
                    rasterized=True, zorder=3)

        # Average cosine distance to the closest matched-class text emb
        gaps = []
        for i, t in enumerate(tgts):
            if t < len(txt):
                gaps.append(1.0 - float(np.dot(vis[i], txt[t])))
        avg_gap = float(np.mean(gaps)) if gaps else 0.0

        ax2.set_title(f"B   Modality Gap\navg cosine distance = {avg_gap:.3f}")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.legend(fontsize=9)
        ax2.grid(False)

        plt.suptitle(
            f"Real Embedding Space — N={self.N} samples, C={self.C} classes",
            fontsize=14, fontweight="bold",
        )
        _save(fig, self.out, "figA_embedding_space")

    # ------------------------------------------------------------------
    def fig_b_similarity_matrix(self, max_show: int = 50) -> None:
        sims = self.d["similarities"]
        tgts = self.d["targets"]
        N_show = min(max_show, sims.shape[0])
        C_show = min(max_show, sims.shape[1])

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(
            sims[:N_show, :C_show],
            aspect="auto", cmap="RdYlBu_r",
            vmin=-0.1, vmax=1.0,
        )
        plt.colorbar(im, ax=ax, label="cosine similarity")
        for i in range(N_show):
            t = int(tgts[i])
            if t < C_show:
                ax.scatter(t, i, marker="x", s=18, color="black", lw=0.8)
        ax.set_xlabel("Class (text embedding)")
        ax.set_ylabel("Test sample (video)")
        ax.set_title(
            f"Similarity Matrix (first {N_show}×{C_show}) — "
            f"black ×: ground truth"
        )
        ax.grid(False)
        _save(fig, self.out, "figB_similarity_matrix")

    # ------------------------------------------------------------------
    def fig_c_topk_bar(self) -> None:
        ks = [1, 5, 10]
        accs = [self._topk_acc(k) * 100 for k in ks]
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [PALETTE["primary"], PALETTE["teal"], PALETTE["success"]]
        bars = ax.bar([f"Top-{k}" for k in ks], accs, color=colors, alpha=0.88)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{a:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Zero-Shot Top-K Accuracy (real)")
        ax.set_ylim(0, min(100, max(accs) + 12))
        _save(fig, self.out, "figC_topk_accuracy")

    # ------------------------------------------------------------------
    def fig_d_confusion_matrix(self, max_classes: int = 30) -> None:
        from .metrics import compute_confusion_matrix
        preds = self.d["predictions"]
        tgts = self.d["targets"]
        unique = np.unique(np.concatenate([preds, tgts]))
        if len(unique) > max_classes:
            top = np.argsort(np.bincount(tgts.astype(int)))[-max_classes:]
            keep = np.isin(tgts, top) & np.isin(preds, top)
            preds_f, tgts_f = preds[keep], tgts[keep]
            label_subset = top
        else:
            preds_f, tgts_f = preds, tgts
            label_subset = unique

        if len(label_subset) == 0:
            logger.warning("No classes to plot in confusion matrix")
            return

        # Re-index labels into 0..k-1 for plotting
        mapping = {c: i for i, c in enumerate(sorted(label_subset))}
        preds_remap = np.array([mapping[c] for c in preds_f])
        tgts_remap = np.array([mapping[c] for c in tgts_f])
        names = [
            self.d["class_names"][c] if c < len(self.d["class_names"]) else str(c)
            for c in sorted(label_subset)
        ]

        cm = compute_confusion_matrix(
            preds_remap, tgts_remap, num_classes=len(names), normalize="true",
        )

        fig, ax = plt.subplots(figsize=(min(12, 0.4 * len(names) + 4),
                                        min(12, 0.4 * len(names) + 4)))
        im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"Confusion Matrix (top {len(names)} classes)")
        ax.grid(False)
        _save(fig, self.out, "figD_confusion_matrix")

    # ------------------------------------------------------------------
    def fig_e_per_signer(self) -> None:
        per_signer_correct: Dict[str, int] = {}
        per_signer_total: Dict[str, int] = {}
        for i, vid in enumerate(self.d["video_ids"]):
            s = self.signer_map.get(vid, "unknown")
            per_signer_total[s] = per_signer_total.get(s, 0) + 1
            if self.d["predictions"][i] == self.d["targets"][i]:
                per_signer_correct[s] = per_signer_correct.get(s, 0) + 1
        if not per_signer_total:
            logger.info("No signer_map entries match video_ids; skipping fig E.")
            return

        signers = sorted(per_signer_total.keys())
        accs = [
            per_signer_correct.get(s, 0) / per_signer_total[s] * 100
            for s in signers
        ]
        mean = float(np.mean(accs))
        std = float(np.std(accs))

        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(signers)), 5))
        colors = [PALETTE["primary"] if a >= mean else PALETTE["muted"]
                  for a in accs]
        ax.bar(range(len(signers)), accs, color=colors, alpha=0.88)
        ax.axhline(mean, color=PALETTE["accent"], lw=2, ls="--",
                   label=f"Mean = {mean:.1f}%")
        ax.axhspan(mean - std, mean + std, color=PALETTE["accent"], alpha=0.1,
                   label=f"±1σ = {std:.1f}%")
        for i, (a, t) in enumerate(zip(accs, [per_signer_total[s] for s in signers])):
            ax.text(i, a + 0.5, f"{a:.0f}%\n(n={t})",
                    ha="center", fontsize=8)
        ax.set_xticks(range(len(signers)))
        ax.set_xticklabels(signers, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title("Per-Signer Zero-Shot Accuracy")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(accs) * 1.2 if accs else 100)
        plt.tight_layout()
        _save(fig, self.out, "figE_per_signer")

    # ------------------------------------------------------------------
    def fig_f_similarity_distributions(self) -> None:
        vis = self.d["vis_embeds"]
        tgts = self.d["targets"]

        rng = np.random.default_rng(42)
        n_pairs = min(5000, self.N * (self.N - 1) // 2)
        if n_pairs < 5:
            logger.info("Too few samples for intra/inter distribution plot")
            return
        i = rng.integers(0, self.N, n_pairs)
        j = rng.integers(0, self.N, n_pairs)
        same = i == j
        i, j = i[~same], j[~same]
        sims = np.sum(vis[i] * vis[j], axis=-1)
        same_class = tgts[i] == tgts[j]
        intra = sims[same_class]
        inter = sims[~same_class]

        fig, ax = plt.subplots(figsize=(9, 5))
        for arr, label, color in (
            (intra, f"Intra-class  (μ={intra.mean():.3f})", PALETTE["success"]),
            (inter, f"Inter-class  (μ={inter.mean():.3f})", PALETTE["primary"]),
        ):
            if len(arr) < 5:
                continue
            xs = np.linspace(arr.min() - 0.05, arr.max() + 0.05, 300)
            kde = gaussian_kde(arr, bw_method=0.15)
            ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
            ax.plot(xs, kde(xs), color=color, lw=2.4, label=label)
        ax.set_xlabel("Cosine similarity (visual–visual)")
        ax.set_ylabel("Density")
        ax.set_title("Intra- vs Inter-class Similarity (real)")
        ax.legend(fontsize=10)
        plt.tight_layout()
        _save(fig, self.out, "figF_similarity_distributions")

    # ------------------------------------------------------------------
    def fig_g_modality_gap(self) -> None:
        vis = self.d["vis_embeds"]
        txt = self.d["text_embeds"]
        tgts = self.d["targets"]
        sims = self.d["similarities"]

        gt_sims, neg_sims = [], []
        for i, t in enumerate(tgts):
            if t >= len(txt):
                continue
            gt_sims.append(float(np.dot(vis[i], txt[t])))
            row = sims[i].copy()
            row[t] = -np.inf
            neg_sims.append(float(row.max()))
        gt_sims = np.asarray(gt_sims)
        neg_sims = np.asarray(neg_sims)
        if len(gt_sims) == 0:
            logger.info("No valid GT pairs for modality-gap plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A — scatter
        ax = axes[0]
        correct = gt_sims > neg_sims
        ax.scatter(neg_sims[correct], gt_sims[correct], c=PALETTE["success"],
                   alpha=0.6, s=20,
                   label=f"Correct (n={int(correct.sum())})", zorder=3)
        ax.scatter(neg_sims[~correct], gt_sims[~correct], c=PALETTE["accent"],
                   marker="X", alpha=0.7, s=25,
                   label=f"Error (n={int((~correct).sum())})", zorder=4)
        lo = float(min(gt_sims.min(), neg_sims.min())) - 0.05
        hi = float(max(gt_sims.max(), neg_sims.max())) + 0.05
        ax.plot([lo, hi], [lo, hi], ls="--", color="black", alpha=0.5,
                label="decision boundary")
        ax.set_xlabel("Best non-GT similarity")
        ax.set_ylabel("GT similarity")
        ax.set_title("A   GT vs Best-Negative Similarity")
        ax.legend(fontsize=9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        # Panel B — KDE of GT sims
        ax2 = axes[1]
        if len(gt_sims) >= 5:
            xs = np.linspace(gt_sims.min() - 0.05, gt_sims.max() + 0.05, 300)
            kde = gaussian_kde(gt_sims, bw_method=0.15)
            ax2.fill_between(xs, kde(xs), alpha=0.3, color=PALETTE["primary"])
            ax2.plot(xs, kde(xs), color=PALETTE["primary"], lw=2.4)
        ax2.axvline(gt_sims.mean(), color=PALETTE["accent"], ls="--", lw=2,
                    label=f"mean = {gt_sims.mean():.3f}")
        ax2.set_xlabel("Cosine similarity (video ↔ GT description)")
        ax2.set_ylabel("Density")
        ax2.set_title("B   Video–GT Similarity Distribution")
        ax2.legend(fontsize=9)

        plt.suptitle("Modality Gap & Alignment (real)", fontsize=14, fontweight="bold")
        _save(fig, self.out, "figG_modality_gap")

    # ------------------------------------------------------------------
    def fig_h_retrieval_ranks(self) -> None:
        sims = self.d["similarities"]
        tgts = self.d["targets"]
        ranks = []
        for i, t in enumerate(tgts):
            order = np.argsort(-sims[i])
            r = int(np.where(order == t)[0][0]) + 1
            ranks.append(r)
        ranks = np.asarray(ranks)
        median = int(np.median(ranks))
        r1 = float((ranks == 1).mean() * 100)
        r5 = float((ranks <= 5).mean() * 100)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A
        ax = axes[0]
        bins = np.logspace(0, np.log10(self.C + 1), 40)
        ax.hist(ranks, bins=bins, color=PALETTE["primary"],
                alpha=0.85, edgecolor="white")
        ax.set_xscale("log")
        ax.axvline(median, color=PALETTE["accent"], lw=2.5, ls="--",
                   label=f"Median rank = {median}")
        ax.set_xlabel("Rank of GT class (log)")
        ax.set_ylabel("Count")
        ax.set_title(f"A   GT Retrieval Rank Distribution\n"
                     f"R@1 = {r1:.1f}%  R@5 = {r5:.1f}%  Median = {median}")
        ax.legend(fontsize=9)

        # Panel B
        ax2 = axes[1]
        ks = np.arange(1, min(self.C + 1, 51))
        recall = [(ranks <= k).mean() * 100 for k in ks]
        ax2.plot(ks, recall, color=PALETTE["primary"], lw=2.4)
        ax2.fill_between(ks, 0, recall, alpha=0.18, color=PALETTE["primary"])
        for km in (1, 5, 10):
            if km <= len(ks):
                ax2.scatter([km], [recall[km - 1]], color=PALETTE["accent"],
                            s=80, zorder=4)
                ax2.text(km + 0.5, recall[km - 1] - 2,
                         f"R@{km} = {recall[km-1]:.1f}%",
                         color=PALETTE["accent"], fontweight="bold", fontsize=9)
        ax2.set_xlabel("k")
        ax2.set_ylabel("Recall@k (%)")
        ax2.set_title("B   Cumulative Recall@k")
        ax2.set_xlim(0, len(ks) + 1); ax2.set_ylim(0, 105)

        plt.suptitle("Zero-Shot Retrieval Analysis (real)",
                     fontsize=14, fontweight="bold")
        _save(fig, self.out, "figH_retrieval_ranks")


# ======================================================================
# 2) Reported-paper-numbers figure generator (no checkpoint required).
# ======================================================================
class ReportedFigures:
    """Reproduces Tables I–III and several Sec. V plots from the paper.

    Every figure carries a 'Source: paper Table X' subtitle so reviewers
    can tell at a glance that the numbers come from the paper, not from
    a current model run.
    """

    # ----- Table I (Sec. V-A) ------------------------------------------
    methods = [
        "Random Guess",
        "ZSSLR Baseline [Bilge et al.]",
        "CLIP-ArASL [Alasmari]",
        "Ours (Sapiens only)",
        "Ours (Sapiens + MotionBERT)",
    ]
    azsld_top1 = [2.0, 51.4, 58.7, 72.3, 79.8]
    msz_top1   = [1.0, 46.2, 52.1, 66.8, 73.5]

    # ----- Table II (ablations) ----------------------------------------
    abl_names = [
        "Sapiens only (single template)",
        "Sapiens only (5-template ensemble)",
        "Full w/o skeleton noise aug.",
        "Full w/o prompt ensemble",
        "Full w/ fine-tuned visual enc.",
        "Full model (frozen visual enc.)",
    ]
    abl_top1 = [68.1, 72.3, 77.1, 75.4, 74.2, 79.8]

    # Temperature sensitivity (Sec. V-C)
    temp_tau = [0.05, 0.07, 0.20]
    temp_acc = [79.5, 79.8, 79.0]    # 0.07 here is the *init* (learned -> 0.11)

    # ----- Table III (backbone comparison) ------------------------------
    backbones = ["DINOv2 ViT-L/14", "CLIP ViT-L/14", "Sapiens-1B (ours)"]
    bb_top1 = [53.8, 56.4, 79.8]

    # ----- Noise sensitivity (Sec. V-F) --------------------------------
    noise_sigma = [0.0, 0.02, 0.05]
    noise_acc = [79.8, 77.1, 71.4]

    # ----- Per-signer summary (Sec. V-E) -------------------------------
    persigner_mean = 79.8
    persigner_std = 5.2
    persigner_min = 73.1
    persigner_max = 84.6
    persigner_n = 10

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        _apply_rc()

    def generate_all(self, output_dir: Union[str, Path]) -> None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        self.fig_table1_top1(out)
        self.fig_table2_ablation(out)
        self.fig_table3_backbones(out)
        self.fig_temperature_sensitivity(out)
        self.fig_noise_sensitivity(out)
        self.fig_per_signer(out)

    # ------------------------------------------------------------------
    def fig_table1_top1(self, out: Path) -> None:
        x = np.arange(len(self.methods))
        w = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        b1 = ax.bar(x - w / 2, self.azsld_top1, w,
                    label="AzSLD (50 unseen)",
                    color=PALETTE["primary"], alpha=0.9)
        b2 = ax.bar(x + w / 2, self.msz_top1, w,
                    label="MS-ZSSLR-W (100 unseen)",
                    color=PALETTE["accent"], alpha=0.9)
        for b, v in list(zip(b1, self.azsld_top1)) + list(zip(b2, self.msz_top1)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.methods, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Top-1 Zero-Shot Accuracy (%)")
        ax.set_ylim(0, 95)
        ax.set_title("Table I — Zero-Shot Top-1 Accuracy by Method")
        ax.text(
            0.99, -0.32, "Source: paper Table I (numbers reported by the authors)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["muted"],
        )
        ax.legend(fontsize=9, loc="upper left")
        _save(fig, out, "fig_table1_top1")

    # ------------------------------------------------------------------
    def fig_table2_ablation(self, out: Path) -> None:
        full_acc = 79.8
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(self.abl_names))
        colors = [
            PALETTE["success"] if a == full_acc
            else (PALETTE["warning"] if a >= full_acc - 4
                  else PALETTE["accent"])
            for a in self.abl_top1
        ]
        ax.barh(y, self.abl_top1, color=colors, alpha=0.9, height=0.6)
        ax.axvline(full_acc, color=PALETTE["success"], ls="--", lw=2,
                   alpha=0.6, label="Full model")
        for i, a in enumerate(self.abl_top1):
            ax.text(a + 0.3, i, f"{a:.1f}%", va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(self.abl_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Top-1 Accuracy on AzSLD 50 unseen (%)")
        ax.set_xlim(60, 85)
        ax.set_title("Table II — Ablation Study (AzSLD 50 unseen)")
        ax.text(
            0.99, -0.18,
            "Source: paper Table II",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["muted"],
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, out, "fig_table2_ablation")

    # ------------------------------------------------------------------
    def fig_table3_backbones(self, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [PALETTE["muted"], PALETTE["muted"], PALETTE["primary"]]
        bars = ax.bar(self.backbones, self.bb_top1, color=colors, alpha=0.88)
        for b, v in zip(bars, self.bb_top1):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_ylim(0, 90)
        ax.set_title("Table III — Visual Backbone Comparison")
        ax.text(
            0.99, -0.20, "Source: paper Table III",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["muted"],
        )
        plt.tight_layout()
        _save(fig, out, "fig_table3_backbones")

    # ------------------------------------------------------------------
    def fig_temperature_sensitivity(self, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.temp_tau, self.temp_acc, "o-",
                color=PALETTE["primary"], lw=2.4, markersize=10)
        for t, a in zip(self.temp_tau, self.temp_acc):
            ax.text(t, a + 0.15, f"{a:.1f}%", ha="center", fontsize=9)
        ax.set_xlabel("Temperature τ (initial)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Temperature Sensitivity (Sec. V-C)")
        ax.set_ylim(78, 81)
        ax.text(
            0.99, -0.18,
            "Source: paper Sec. V-C / Table II final block",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["muted"],
        )
        _save(fig, out, "fig_temperature_sensitivity")

    # ------------------------------------------------------------------
    def fig_noise_sensitivity(self, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.noise_sigma, self.noise_acc, "s-",
                color=PALETTE["accent"], lw=2.4, markersize=10)
        for s, a in zip(self.noise_sigma, self.noise_acc):
            ax.text(s, a + 0.4, f"{a:.1f}%", ha="center", fontsize=9)
        ax.set_xlabel("Added test-time skeleton noise σ")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Skeleton Noise Sensitivity (Sec. V-F)")
        ax.text(
            0.99, -0.18, "Source: paper Sec. V-F",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["muted"],
        )
        _save(fig, out, "fig_noise_sensitivity")

    # ------------------------------------------------------------------
    def fig_per_signer(self, out: Path) -> None:
        # Reconstruct an illustrative distribution from reported mean/std/min/max,
        # explicitly labelled in the title so reviewers don't confuse it with raw data.
        rng = np.random.default_rng(0)
        samples = np.clip(
            rng.normal(self.persigner_mean, self.persigner_std, 1000),
            self.persigner_min, self.persigner_max,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(samples, bins=20, color=PALETTE["primary"], alpha=0.75,
                edgecolor="white")
        ax.axvline(self.persigner_mean, color=PALETTE["accent"], lw=2,
                   ls="--", label=f"mean = {self.persigner_mean:.1f}%")
        ax.axvline(self.persigner_min, color=PALETTE["muted"], lw=1.5,
                   ls=":", label=f"min = {self.persigner_min:.1f}%")
        ax.axvline(self.persigner_max, color=PALETTE["muted"], lw=1.5,
                   ls=":", label=f"max = {self.persigner_max:.1f}%")
        ax.set_xlabel("Per-signer Top-1 accuracy (%)")
        ax.set_ylabel("Density (illustrative sample)")
        ax.set_title("Per-Signer Accuracy — reported distribution "
                     f"(n={self.persigner_n}, μ={self.persigner_mean}, σ={self.persigner_std})")
        ax.text(
            0.99, -0.18,
            "Source: paper Sec. V-E. Histogram is an illustrative sample "
            "from a Gaussian with the reported (μ, σ, min, max); the paper "
            "does not list per-signer values.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=PALETTE["muted"],
        )
        ax.legend(fontsize=9)
        _save(fig, out, "fig_per_signer_reported")
