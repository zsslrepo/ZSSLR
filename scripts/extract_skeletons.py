"""Run MediaPipe Pose on every video and save the (T, 17, 2) H36M keypoint
sequence as `{output_dir}/{gloss_id}/{video_stem}.npy` (MotionBERT input).

Usage
-----
    python scripts/extract_skeletons.py \
        --video_dir  data/azsld/videos \
        --output_dir data/azsld/skeletons \
        [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import discover_videos
from src.data.preprocessing import SkeletonExtractor
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract MediaPipe Holistic skeletons")
    p.add_argument("--video_dir",  type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-extract even if the .npy already exists.")
    p.add_argument("--complexity", type=int, default=2,
                   help="MediaPipe Holistic model_complexity (0/1/2).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("extract_skeletons")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_videos(args.video_dir)
    logger.info("Discovered %d videos in %s", len(pairs), args.video_dir)

    extractor = SkeletonExtractor(model_complexity=args.complexity)

    n_done = n_skipped = n_failed = 0
    try:
        for gloss, vp in tqdm(pairs, desc="MediaPipe Holistic"):
            out_path = out_root / gloss / f"{vp.stem}.npy"
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                continue
            try:
                extractor.process_video(vp, out_path)
                n_done += 1
            except (IOError, RuntimeError) as e:
                logger.warning("Failed on %s: %s", vp, e)
                n_failed += 1
    finally:
        extractor.close()

    logger.info("Extracted: %d, skipped: %d, failed: %d", n_done, n_skipped, n_failed)


if __name__ == "__main__":
    main()
