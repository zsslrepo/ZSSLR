"""Preprocessing utilities for video frames, skeletons, and prompts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Video preprocessor (used by extract_skeletons.py — the SignLanguageDataset
# does its own preprocessing inline for efficiency).
# ----------------------------------------------------------------------
class VideoPreprocessor:
    """Sample frames + resize + ImageNet normalise."""

    def __init__(
        self,
        target_size: int = 1024,
        num_frames: int = 16,
        sampling: str = "uniform",
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.num_frames = num_frames
        self.sampling = sampling
        self.mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.std = torch.tensor(normalize_std).view(3, 1, 1)

    def sample_indices(self, total_frames: int) -> np.ndarray:
        T = self.num_frames
        if total_frames < T:
            base = np.arange(total_frames)
            pad = np.full(T - total_frames, total_frames - 1, dtype=int)
            return np.concatenate([base, pad])
        if self.sampling == "uniform":
            return np.linspace(0, total_frames - 1, T, dtype=int)
        if self.sampling == "random":
            return np.sort(np.random.choice(total_frames, T, replace=False))
        if self.sampling == "consecutive":
            start = np.random.randint(0, total_frames - T + 1)
            return np.arange(start, start + T)
        raise ValueError(f"Unknown sampling: {self.sampling}")

    def __call__(self, video_path: Union[str, Path]) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise IOError(f"cannot read video {video_path}")
        wanted = set(int(i) for i in self.sample_indices(total))
        frames = {}
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.target_size, self.target_size))
                frames[i] = frame
            i += 1
        cap.release()

        idx_list = sorted(wanted)
        arr = np.stack([frames[i] for i in idx_list], axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        tensor = (tensor - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)
        return tensor                                     # (T, 3, H, W)


# ----------------------------------------------------------------------
# Skeleton extractor — produces (T, 133, 2) per video.
# Layout:  33 body + 58 face (uniform subsample from 478) + 21 left + 21 right
# = 33 + 58 + 21 + 21 = 133  (matches paper Sec. III-B)
# ----------------------------------------------------------------------
class SkeletonExtractor:
    """Extract 133 normalised (x, y) keypoints per frame using MediaPipe Holistic."""

    POSE_JOINTS = 33
    FACE_JOINTS = 58
    HAND_JOINTS = 21
    TOTAL_JOINTS = POSE_JOINTS + FACE_JOINTS + 2 * HAND_JOINTS   # 133

    def __init__(self, model_complexity: int = 2):
        try:
            import mediapipe as mp
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "mediapipe is required for SkeletonExtractor. "
                "Install with `pip install mediapipe`."
            ) from e
        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_keypoints(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(rgb)

        pts: List[List[float]] = []

        # 1) Body pose (33)
        if res.pose_landmarks:
            pts.extend([[lm.x, lm.y] for lm in res.pose_landmarks.landmark])
        else:
            pts.extend([[0.0, 0.0]] * self.POSE_JOINTS)

        # 2) Face (uniform sub-sample to 58)
        if res.face_landmarks:
            full = [[lm.x, lm.y] for lm in res.face_landmarks.landmark]
            n = len(full)
            if n >= self.FACE_JOINTS:
                idx = np.linspace(0, n - 1, self.FACE_JOINTS, dtype=int)
                pts.extend([full[i] for i in idx])
            else:                                       # pragma: no cover
                pts.extend(full + [[0.0, 0.0]] * (self.FACE_JOINTS - n))
        else:
            pts.extend([[0.0, 0.0]] * self.FACE_JOINTS)

        # 3) Left hand (21)
        if res.left_hand_landmarks:
            pts.extend([[lm.x, lm.y] for lm in res.left_hand_landmarks.landmark])
        else:
            pts.extend([[0.0, 0.0]] * self.HAND_JOINTS)

        # 4) Right hand (21)
        if res.right_hand_landmarks:
            pts.extend([[lm.x, lm.y] for lm in res.right_hand_landmarks.landmark])
        else:
            pts.extend([[0.0, 0.0]] * self.HAND_JOINTS)

        arr = np.asarray(pts[: self.TOTAL_JOINTS], dtype=np.float32)
        if arr.shape != (self.TOTAL_JOINTS, 2):
            raise RuntimeError(
                f"Expected ({self.TOTAL_JOINTS},2) keypoints, got {arr.shape}"
            )
        return arr

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        seq: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            seq.append(self.extract_keypoints(frame))
        cap.release()

        if not seq:
            raise IOError(f"No frames decoded from {video_path}")

        sk = np.stack(seq, axis=0)                       # (T, 133, 2)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), sk)
        return sk

    def close(self):
        try:
            self.holistic.close()
        except Exception:                                # pragma: no cover
            pass


# ----------------------------------------------------------------------
# Description loader
# ----------------------------------------------------------------------
class DescriptionLoader:
    """Load `gloss_id -> description` map from a JSON file."""

    def __init__(self, descriptions_file: Union[str, Path]):
        self.path = Path(descriptions_file)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            self.descriptions: Dict[str, str] = dict(data)
        elif isinstance(data, list):
            # List of {gloss, description}
            self.descriptions = {item["gloss"]: item["description"] for item in data}
        else:
            raise ValueError(
                f"Unexpected description format in {self.path}: {type(data)}"
            )
        logger.info("Loaded %d gloss descriptions from %s",
                    len(self.descriptions), self.path)

    def __getitem__(self, gloss: str) -> str:
        return self.descriptions[gloss]

    def __contains__(self, gloss: str) -> bool:
        return gloss in self.descriptions

    def __len__(self) -> int:
        return len(self.descriptions)

    def keys(self):
        return self.descriptions.keys()

    def items(self):
        return self.descriptions.items()


# ----------------------------------------------------------------------
# Prompt builder — wrap a single description into Azerbaijani templates.
# ----------------------------------------------------------------------
class PromptBuilder:
    """Wrap descriptions with Azerbaijani prompt templates."""

    def __init__(self, templates: Optional[Sequence[str]] = None):
        from ..models.azbert_encoder import PROMPT_TEMPLATES_AZ
        self.templates = list(templates) if templates is not None else list(PROMPT_TEMPLATES_AZ)
        for t in self.templates:
            if "{description}" not in t:
                raise ValueError(f"template missing placeholder: {t!r}")

    def __call__(self, description: str) -> List[str]:
        return [t.format(description=description) for t in self.templates]
