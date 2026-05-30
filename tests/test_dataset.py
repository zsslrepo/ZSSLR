"""Tests for the dataset / splits / preprocessing layer.

We use synthetic videos created with `imageio` to avoid bundling real data.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from src.data import (
    SignLanguageDataset,
    ZeroShotSignDataset,
    build_random_split,
    collate_fn,
    load_split,
    write_split,
)


# ---------------------------------------------------------------------- helpers
def _write_dummy_video(path: Path, n_frames: int = 8, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (size, size))
    rng = np.random.default_rng(int.from_bytes(path.name.encode(), "little") & 0xffff)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def tiny_corpus(tmp_path: Path):
    """Create a 4-gloss dummy AzSLD: 3 videos each for ata/ana/su/ev."""
    video_dir = tmp_path / "videos"
    glosses = ["ata", "ana", "su", "ev"]
    for g in glosses:
        for i in range(3):
            _write_dummy_video(video_dir / g / f"signer{i:02d}.mp4")

    descriptions = {
        "ata": "Baş barmaq alına toxunur.",
        "ana": "Baş barmaq çənəyə toxunur.",
        "su":  "Üç barmaq dodağa yaxınlaşır.",
        "ev":  "İki əl ev şəklində birləşir.",
    }
    desc_path = tmp_path / "descriptions.json"
    desc_path.write_text(json.dumps(descriptions, ensure_ascii=False), encoding="utf-8")

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    write_split(splits_dir / "seen_glosses.txt",   ["ata", "ana", "su"])
    write_split(splits_dir / "unseen_glosses.txt", ["ev"])

    return dict(
        video_dir=video_dir, descriptions=descriptions,
        descriptions_path=desc_path, splits_dir=splits_dir,
    )


# ---------------------------------------------------------------------- tests
def test_load_and_write_split(tmp_path: Path):
    p = tmp_path / "g.txt"
    write_split(p, ["a", "b", "c"])
    assert load_split(p) == ["a", "b", "c"]


def test_build_random_split_disjoint():
    seen, unseen = build_random_split([f"g{i:03d}" for i in range(450)], num_unseen=50, seed=42)
    assert len(seen) == 400 and len(unseen) == 50
    assert set(seen).isdisjoint(unseen)


def test_dataset_finds_videos_by_gloss_dir(tiny_corpus):
    ds = SignLanguageDataset(
        video_dir=tiny_corpus["video_dir"],
        descriptions=tiny_corpus["descriptions"],
        num_frames=4,
        target_size=32,
    )
    # 4 glosses * 3 videos each = 12 samples
    assert len(ds) == 12
    item = ds[0]
    assert item["frames"].shape == (4, 3, 32, 32)
    assert item["skeleton"] is None
    assert item["description"] == tiny_corpus["descriptions"][item["gloss"]]


def test_collate_handles_missing_skeleton(tiny_corpus):
    ds = SignLanguageDataset(
        video_dir=tiny_corpus["video_dir"],
        descriptions=tiny_corpus["descriptions"],
        num_frames=4, target_size=32,
    )
    batch = collate_fn([ds[i] for i in range(3)])
    assert batch["frames"].shape == (3, 4, 3, 32, 32)
    assert batch["skeleton"] is None
    assert isinstance(batch["description"], list) and len(batch["description"]) == 3
    assert batch["label"].shape == torch.Size([3])


def test_zero_shot_dataset_filters_by_split(tiny_corpus):
    seen = load_split(tiny_corpus["splits_dir"] / "seen_glosses.txt")
    unseen = load_split(tiny_corpus["splits_dir"] / "unseen_glosses.txt")

    train_ds = ZeroShotSignDataset(
        video_dir=tiny_corpus["video_dir"],
        descriptions=tiny_corpus["descriptions"],
        num_frames=4, target_size=32,
        seen_glosses=seen, unseen_glosses=unseen, split="seen",
    )
    test_ds = ZeroShotSignDataset(
        video_dir=tiny_corpus["video_dir"],
        descriptions=tiny_corpus["descriptions"],
        num_frames=4, target_size=32,
        seen_glosses=seen, unseen_glosses=unseen, split="unseen",
    )
    train_glosses = {s.gloss for s in train_ds.samples}
    test_glosses = {s.gloss for s in test_ds.samples}
    assert train_glosses == set(seen)
    assert test_glosses == set(unseen)
    assert train_glosses.isdisjoint(test_glosses)
