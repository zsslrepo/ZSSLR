"""PyTorch datasets for AzSL / MS-ZSSLR-W sign language videos.

Critical conventions
--------------------
* `descriptions.json` is keyed by **gloss_id**, not by per-video filename
  — every video that depicts the same gloss shares the same description.
  The old repo confused these and would silently drop most samples.
* Each video file lives at `{video_dir}/{gloss_id}/{signer}_{instance}.mp4`.
  We infer `gloss_id` from the **immediate parent directory** of the
  video file.
* Skeletons are pre-extracted with `scripts/extract_skeletons.py` and
  stored at `{skeleton_dir}/{gloss_id}/{signer}_{instance}.npy` —
  (T_full, 17, 2) H36M keypoints per file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: discover videos and infer (gloss_id, video_path) pairs.
# ----------------------------------------------------------------------
def discover_videos(
    video_dir: Union[str, Path],
    extensions: Sequence[str] = (".mp4", ".avi", ".mov", ".mkv"),
) -> List[Tuple[str, Path]]:
    """Walk `video_dir` and return list of (gloss_id, file_path)."""
    root = Path(video_dir)
    out: List[Tuple[str, Path]] = []
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            # Gloss id = directory name immediately above the file.
            # If videos are not nested, fall back to file stem.
            if p.parent != root:
                gloss = p.parent.name
            else:
                gloss = p.stem
            out.append((gloss, p))
    out.sort(key=lambda x: (x[0], x[1].name))
    return out


# ----------------------------------------------------------------------
@dataclass
class Sample:
    video_id: str            # filename stem, unique across the dataset
    gloss: str               # class id (used to look up description)
    video_path: Path
    skeleton_path: Optional[Path] = None


# ----------------------------------------------------------------------
class SignLanguageDataset(Dataset):
    """Loads video frames + optional skeletons for sign language training."""

    def __init__(
        self,
        video_dir: Union[str, Path],
        descriptions: Dict[str, str],
        gloss_to_label: Optional[Dict[str, int]] = None,
        skeleton_dir: Optional[Union[str, Path]] = None,
        num_frames: int = 16,
        target_size: int = 1024,
        sampling: str = "uniform",
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
        glosses: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.video_dir = Path(video_dir)
        self.skeleton_dir = Path(skeleton_dir) if skeleton_dir else None
        self.descriptions = descriptions
        self.num_frames = num_frames
        self.target_size = target_size
        self.sampling = sampling
        self.transform = transform
        self.mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.std = torch.tensor(normalize_std).view(3, 1, 1)

        # If the caller did not provide a gloss-to-label map, build a
        # stable, alphabetical one over the glosses that have descriptions.
        all_glosses = sorted(descriptions.keys()) if glosses is None else sorted(set(glosses))
        if gloss_to_label is None:
            gloss_to_label = {g: i for i, g in enumerate(all_glosses)}
        self.gloss_to_label = gloss_to_label
        self.label_to_gloss = {v: k for k, v in gloss_to_label.items()}

        # Discover videos and filter
        glosses_filter = set(all_glosses)
        discovered = discover_videos(self.video_dir)
        self.samples: List[Sample] = []
        skipped_no_desc = 0
        skipped_not_in_split = 0
        for gloss, vp in discovered:
            if gloss not in descriptions:
                skipped_no_desc += 1
                continue
            if gloss not in glosses_filter:
                skipped_not_in_split += 1
                continue
            sp = None
            if self.skeleton_dir is not None:
                cand = self.skeleton_dir / gloss / f"{vp.stem}.npy"
                if cand.exists():
                    sp = cand
            self.samples.append(Sample(
                video_id=vp.stem,
                gloss=gloss,
                video_path=vp,
                skeleton_path=sp,
            ))

        logger.info(
            "SignLanguageDataset: %d videos kept "
            "(%d missing description, %d outside split). %d glosses, "
            "skeletons available for %d samples.",
            len(self.samples), skipped_no_desc, skipped_not_in_split,
            len(self.gloss_to_label),
            sum(1 for s in self.samples if s.skeleton_path is not None),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _sample_indices(self, total: int) -> np.ndarray:
        T = self.num_frames
        if total <= 0:
            raise ValueError(f"empty video")
        if total < T:
            # Repeat the last frame to pad
            base = np.arange(total)
            pad = np.full(T - total, total - 1, dtype=int)
            return np.concatenate([base, pad])
        if self.sampling == "uniform":
            return np.linspace(0, total - 1, T, dtype=int)
        if self.sampling == "random":
            return np.sort(np.random.choice(total, T, replace=False))
        if self.sampling == "consecutive":
            start = np.random.randint(0, total - T + 1)
            return np.arange(start, start + T)
        raise ValueError(f"Unknown sampling strategy: {self.sampling!r}")

    # ------------------------------------------------------------------
    def _load_video(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise IOError(f"cannot read video {path}")
        idx_set = self._sample_indices(total)
        wanted = set(int(i) for i in idx_set)

        # We read sequentially and decode only frames we need.
        frames_by_idx: Dict[int, np.ndarray] = {}
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[0] != self.target_size or frame.shape[1] != self.target_size:
                    frame = cv2.resize(frame, (self.target_size, self.target_size))
                frames_by_idx[frame_idx] = frame
            frame_idx += 1
        cap.release()

        # Assemble in sampled order, padding with the last available frame.
        last = None
        out_frames = []
        for i in idx_set:
            i = int(i)
            f = frames_by_idx.get(i, last)
            if f is None:
                # Could happen if the very first sampled idx wasn't decoded.
                f = np.zeros(
                    (self.target_size, self.target_size, 3), dtype=np.uint8
                )
            last = f
            out_frames.append(f)

        arr = np.stack(out_frames, axis=0)                     # (T, H, W, 3) uint8
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)

        # ImageNet normalisation (broadcast over T)
        tensor = (tensor - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor                                          # (T, 3, H, W)

    # ------------------------------------------------------------------
    def _load_skeleton(self, path: Optional[Path], num_frames_video: int = -1) -> Optional[torch.Tensor]:
        if path is None:
            return None
        try:
            arr = np.load(path)                                # (T_full, 17, 2)
        except (FileNotFoundError, ValueError):
            return None
        if arr.ndim != 3 or arr.shape[2] != 2:
            logger.warning("Skeleton at %s has unexpected shape %s, skipping.",
                           path, arr.shape)
            return None
        T_full = arr.shape[0]
        idx = self._sample_indices(T_full)
        sampled = arr[idx]                                     # (T, J, 2)
        return torch.from_numpy(sampled).float()

    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict:
        s = self.samples[i]
        frames = self._load_video(s.video_path)
        skeleton = self._load_skeleton(s.skeleton_path)
        description = self.descriptions[s.gloss]
        label = self.gloss_to_label[s.gloss]
        return {
            "frames":      frames,                   # (T, 3, H, W)
            "skeleton":    skeleton,                 # (T, J, 2) or None
            "description": description,              # str
            "label":       label,                    # int
            "video_id":    s.video_id,               # str
            "gloss":       s.gloss,                  # str
        }


# ======================================================================
class ZeroShotSignDataset(SignLanguageDataset):
    """Same as `SignLanguageDataset` but restricted to a seen/unseen split.

    For zero-shot training, instantiate with `glosses=seen_glosses`.
    For zero-shot evaluation, instantiate with `glosses=unseen_glosses`
    and pass the **full gloss-to-label** map so label indices are
    consistent with the description vocabulary used at inference.
    """

    def __init__(
        self,
        *args,
        seen_glosses: Optional[Sequence[str]] = None,
        unseen_glosses: Optional[Sequence[str]] = None,
        split: str = "unseen",                       # 'seen' | 'unseen' | 'all'
        **kwargs,
    ):
        seen_set = set(seen_glosses or [])
        unseen_set = set(unseen_glosses or [])
        if split == "seen":
            glosses = seen_glosses
        elif split == "unseen":
            glosses = unseen_glosses
        elif split == "all":
            glosses = list(seen_set | unseen_set) or None
        else:
            raise ValueError(f"split must be 'seen'/'unseen'/'all', got {split!r}")

        kwargs["glosses"] = glosses
        super().__init__(*args, **kwargs)
        self.split = split
        self.seen_set = seen_set
        self.unseen_set = unseen_set


# ----------------------------------------------------------------------
def collate_fn(batch: List[Dict]) -> Dict:
    """Collate samples, handling videos with missing skeletons."""
    frames = torch.stack([b["frames"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    descriptions = [b["description"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    glosses = [b["gloss"] for b in batch]

    skeletons = [b["skeleton"] for b in batch]
    if all(s is not None for s in skeletons):
        skeleton = torch.stack(skeletons)
    else:
        skeleton = None                              # mixed batch: drop skeleton

    return {
        "frames":      frames,
        "skeleton":    skeleton,
        "description": descriptions,
        "label":       labels,
        "video_id":    video_ids,
        "gloss":       glosses,
    }
