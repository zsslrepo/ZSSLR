"""Build `descriptions.json` — one text description per gloss.

The text encoder wraps each description in the Azerbaijani prompt ensemble
(e.g. "Azərbaycan İşarə Dilində {description}"), so a description can be as
simple as the readable word. By default we use the lower-cased gloss word;
pass `--override` with a JSON `{gloss: description}` map to supply richer,
hand-written definitions (recommended for better zero-shot quality).

Glosses are collected from a manifest, a video directory, or split files.

Usage
-----
    python scripts/build_descriptions.py \
        --manifest /home/mahammad/Desktop/AZSLD/data/manifest.json \
        --output data/azsld/descriptions.json

    # With hand-written definitions merged on top of the defaults:
    python scripts/build_descriptions.py \
        --video_dir data/azsld/videos \
        --override data/azsld/descriptions_manual.json \
        --output data/azsld/descriptions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import discover_videos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build gloss descriptions.json")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str, help="AzSLD-style manifest.json")
    src.add_argument("--video_dir", type=str, help="Directory of <gloss>/<video> folders")
    src.add_argument("--glosses_file", type=str, help="Newline-separated gloss list")
    p.add_argument("--override", type=str, default=None,
                   help="JSON {gloss: description} merged over the defaults.")
    p.add_argument("--output", type=str, default="data/azsld/descriptions.json")
    return p.parse_args()


def collect_glosses(args) -> List[str]:
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            data = json.load(f)
        return sorted({entry["gloss"] for entry in data})
    if args.video_dir:
        return sorted({g for g, _ in discover_videos(args.video_dir)})
    return sorted({
        line.strip()
        for line in Path(args.glosses_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    })


def default_description(gloss: str) -> str:
    """Readable word form used inside the Azerbaijani prompt ensemble."""
    return gloss.replace("_", " ").strip().lower()


def main() -> None:
    args = parse_args()
    glosses = collect_glosses(args)

    descriptions: Dict[str, str] = {g: default_description(g) for g in glosses}

    if args.override:
        override = json.loads(Path(args.override).read_text(encoding="utf-8"))
        n_before = len(override)
        descriptions.update({g: d for g, d in override.items() if g in descriptions})
        print(f"Merged {n_before} manual descriptions "
              f"({sum(1 for g in override if g in descriptions)} matched glosses).")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Wrote {len(descriptions)} gloss descriptions to {out}")


if __name__ == "__main__":
    main()
