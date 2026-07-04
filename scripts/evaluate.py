"""Zero-shot evaluation script.

Loads a trained checkpoint, encodes all unseen-class descriptions and
test videos, computes Top-1/5/10 + mAP + per-signer accuracy, and
saves embeddings/predictions/metrics to `--output_dir`.

Usage
-----
    python scripts/evaluate.py \
        --checkpoint outputs/exp01/checkpoints/best.pt \
        --config     configs/default.yaml \
        --video_dir       data/azsld/videos \
        --skeleton_dir    data/azsld/skeletons \
        --descriptions    data/azsld/descriptions.json \
        --unseen_glosses  data/azsld/splits/unseen_glosses.txt \
        --signer_map      data/azsld/splits/signer_map.json \
        --output_dir      outputs/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    DescriptionLoader,
    ZeroShotSignDataset,
    collate_fn,
    load_split,
)
from src.evaluation import ZeroShotEvaluator
from src.models import MultimodalZSLModel
from src.utils import load_checkpoint, setup_logger
from src.utils.visualization import FigureGenerator


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate ZSSLR-AzSL")
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--config",          type=str, default="configs/default.yaml")
    p.add_argument("--video_dir",       type=str, required=True)
    p.add_argument("--skeleton_dir",    type=str, default=None)
    p.add_argument("--descriptions",    type=str, required=True)
    p.add_argument("--seen_glosses",    type=str, default=None)
    p.add_argument("--unseen_glosses",  type=str, required=True)
    p.add_argument("--signer_map",      type=str, default=None)
    p.add_argument("--output_dir",      type=str, default="outputs/evaluation")
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_figures",      action="store_true",
                   help="Skip figure generation after computing metrics.")
    return p.parse_args()


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evaluate", log_dir=out / "logs")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------------------- Load descriptions + splits -------------
    descs = DescriptionLoader(args.descriptions)
    descriptions = dict(descs.items())

    seen = load_split(args.seen_glosses) if args.seen_glosses else []
    unseen = load_split(args.unseen_glosses)

    all_glosses = sorted(set(seen) | set(unseen) | set(descriptions.keys()))
    gloss_to_label = {g: i for i, g in enumerate(all_glosses)}

    signer_map = {}
    if args.signer_map:
        with open(args.signer_map, encoding="utf-8") as f:
            signer_map = json.load(f)

    # ---------------------- Build evaluation dataset ---------------
    pp = cfg["data"]["preprocessing"]
    eval_ds = ZeroShotSignDataset(
        video_dir=args.video_dir,
        skeleton_dir=args.skeleton_dir,
        descriptions=descriptions,
        gloss_to_label=gloss_to_label,
        seen_glosses=seen,
        unseen_glosses=unseen,
        split="unseen",
        num_frames=pp["num_frames"],
        target_size=pp["spatial_size"],
        sampling=pp["sampling"],
        normalize_mean=pp["normalize_mean"],
        normalize_std=pp["normalize_std"],
    )
    if len(eval_ds) == 0:
        raise RuntimeError(
            "Evaluation dataset is empty. "
            "Check that `--video_dir` contains unseen-gloss videos and that "
            "`--descriptions` covers them."
        )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=cfg["hardware"]["pin_memory"],
    )

    # ---------------------- Build + load model ---------------------
    logger.info("Building model...")
    model = MultimodalZSLModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        visual_dim=cfg["model"].get("visual_dim", 1024),
        motion_dim=cfg["model"].get("motion_dim", 512),
        temperature=cfg["model"]["temperature_init"],
        temperature_learnable=cfg["model"]["temperature_learnable"],
        sapiens_config=cfg["model"]["visual"]["sapiens"],
        motionbert_config=cfg["model"]["visual"]["motionbert"],
        azbert_config=cfg["model"]["text"],
    )
    info = load_checkpoint(args.checkpoint, model, map_location=args.device)
    logger.info("Loaded checkpoint (epoch=%d, step=%d)",
                info["epoch"], info["step"])
    model.to(args.device)

    # ---------------------- Evaluate -------------------------------
    eval_descriptions = {g: descriptions[g] for g in unseen if g in descriptions}
    evaluator = ZeroShotEvaluator(model, device=args.device)
    metrics = evaluator.evaluate(eval_loader, eval_descriptions, split_name="unseen")

    if signer_map:
        per_signer = evaluator.evaluate_per_signer("unseen", signer_map)
        (out / "per_signer.json").write_text(
            json.dumps(per_signer, indent=2),
            encoding="utf-8",
        )

    evaluator.save_results(out, split_name="unseen")

    # ---------------------- Figures --------------------------------
    if not args.no_figures:
        logger.info("Generating figures from real arrays...")
        fg = FigureGenerator.from_npz(
            eval_dir=out,
            signer_map_path=args.signer_map,
            output_dir=out / "figures",
        )
        fg.generate_all()

    logger.info("Done. Outputs at %s", out)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
