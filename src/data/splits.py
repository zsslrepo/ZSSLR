"""Helpers for seen/unseen gloss splits.

The paper splits AzSLD's 450 glosses into 400 seen / 50 unseen
deterministically. We store the split as two newline-separated text
files: `seen_glosses.txt` and `unseen_glosses.txt`.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
def load_split(path: Union[str, Path]) -> List[str]:
    """Read a .txt of gloss ids (one per line, blanks/`#` comments allowed)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if len(out) != len(set(out)):
        raise ValueError(f"Duplicate gloss ids in {p}")
    return out


# ----------------------------------------------------------------------
def write_split(path: Union[str, Path], glosses: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(glosses) + "\n", encoding="utf-8")
    logger.info("Wrote %d glosses to %s", len(glosses), p)


# ----------------------------------------------------------------------
def build_random_split(
    all_glosses: Sequence[str],
    num_unseen: int = 50,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Sample `num_unseen` glosses without replacement to form the unseen set.

    Returns (seen, unseen), both sorted alphabetically for reproducibility.
    """
    glosses = sorted(set(all_glosses))
    if num_unseen >= len(glosses):
        raise ValueError(
            f"num_unseen ({num_unseen}) >= total glosses ({len(glosses)})"
        )
    rng = random.Random(seed)
    unseen = sorted(rng.sample(glosses, num_unseen))
    seen = sorted(set(glosses) - set(unseen))
    return seen, unseen
