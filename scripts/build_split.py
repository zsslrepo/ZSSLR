"""Build a deterministic zero-shot seen/unseen gloss split.

The unseen glosses are sampled *frequency-stratified* (paper Sec. III-A):
glosses are bucketed by video count and the unseen set is drawn
proportionally from each bucket, so seen and unseen share a similar
frequency profile. Only glosses with at least `--min_videos_per_gloss`
videos are eligible.

Gloss counts come either from an AzSLD-style `manifest.json`
(list of {"gloss": ..., ...}) or by scanning a `--video_dir` whose
immediate sub-directories are gloss ids.

Usage
-----
    # From the local AzSLD manifest:
    python scripts/build_split.py \
        --manifest /home/mahammad/Desktop/AZSLD/data/manifest.json \
        --num_unseen 50 --min_videos_per_gloss 5 --seed 42 \
        --output_dir data/azsld/splits

    # Or by scanning a video directory:
    python scripts/build_split.py \
        --video_dir data/azsld/videos \
        --num_unseen 50 --output_dir data/azsld/splits
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import discover_videos
from src.data.splits import write_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build zero-shot seen/unseen gloss split")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str, help="AzSLD-style manifest.json")
    src.add_argument("--video_dir", type=str, help="Directory of <gloss>/<video> folders")
    p.add_argument("--num_unseen", type=int, default=50)
    p.add_argument("--min_videos_per_gloss", type=int, default=5)
    p.add_argument("--num_buckets", type=int, default=4,
                   help="Frequency buckets for stratified unseen sampling.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="data/azsld/splits")
    return p.parse_args()


def gloss_counts_from_manifest(path: str) -> Counter:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Counter(entry["gloss"] for entry in data)


def gloss_counts_from_video_dir(video_dir: str) -> Counter:
    counts: Counter = Counter()
    for gloss, _ in discover_videos(video_dir):
        counts[gloss] += 1
    return counts


def stratified_unseen(
    counts: Counter, num_unseen: int, num_buckets: int, seed: int
) -> List[str]:
    """Bucket glosses by frequency and sample the unseen set proportionally."""
    glosses = sorted(counts)                       # deterministic order
    n = len(glosses)
    if num_unseen >= n:
        raise ValueError(f"num_unseen ({num_unseen}) >= eligible glosses ({n})")

    # Rank by frequency, then split into `num_buckets` contiguous buckets.
    by_freq = sorted(glosses, key=lambda g: (counts[g], g))
    buckets: Dict[int, List[str]] = defaultdict(list)
    for i, g in enumerate(by_freq):
        buckets[min(i * num_buckets // n, num_buckets - 1)].append(g)

    rng = random.Random(seed)
    unseen: List[str] = []
    for b in range(num_buckets):
        bucket = buckets[b]
        # Proportional quota for this bucket (at least what rounding gives).
        quota = round(num_unseen * len(bucket) / n)
        quota = min(quota, len(bucket))
        unseen.extend(rng.sample(bucket, quota))

    # Fix rounding drift to land exactly on num_unseen.
    remaining = [g for g in by_freq if g not in set(unseen)]
    while len(unseen) < num_unseen and remaining:
        unseen.append(remaining.pop(rng.randrange(len(remaining))))
    while len(unseen) > num_unseen:
        unseen.pop()

    return sorted(unseen)


def main() -> None:
    args = parse_args()

    if args.manifest:
        counts = gloss_counts_from_manifest(args.manifest)
    else:
        counts = gloss_counts_from_video_dir(args.video_dir)

    eligible = Counter({g: c for g, c in counts.items()
                        if c >= args.min_videos_per_gloss})
    dropped = len(counts) - len(eligible)

    unseen = stratified_unseen(eligible, args.num_unseen, args.num_buckets, args.seed)
    seen = sorted(set(eligible) - set(unseen))

    out = Path(args.output_dir)
    write_split(out / "seen_glosses.txt", seen)
    write_split(out / "unseen_glosses.txt", unseen)

    summary = {
        "seed": args.seed,
        "total_glosses": len(counts),
        "eligible_glosses": len(eligible),
        "dropped_below_min": dropped,
        "min_videos_per_gloss": args.min_videos_per_gloss,
        "num_seen": len(seen),
        "num_unseen": len(unseen),
        "num_buckets": args.num_buckets,
    }
    (out / "split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote seen/unseen splits to {out}/")


if __name__ == "__main__":
    main()
