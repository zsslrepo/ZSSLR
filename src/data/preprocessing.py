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
# Skeleton extractor — produces (T, 17, 2) per video in H36M joint order.
#
# The paper feeds MotionBERT a "17-joint H36M-compatible skeleton"
# (Sec. III-B). We run MediaPipe Pose (33 body landmarks) and map them to
# the standard 17-joint Human3.6M layout used by MotionBERT / VideoPose3D:
#
#   0 Hip(pelvis) 1 RHip 2 RKnee 3 RAnkle 4 LHip 5 LKnee 6 LAnkle
#   7 Spine 8 Thorax 9 Neck/Nose 10 Head 11 LShoulder 12 LElbow 13 LWrist
#   14 RShoulder 15 RElbow 16 RWrist
#
# Derived joints (pelvis, spine, thorax, head) are computed as midpoints.
# ----------------------------------------------------------------------
class SkeletonExtractor:
    """Extract 17 normalised (x, y) H36M keypoints per frame via MediaPipe Pose."""

    TOTAL_JOINTS = 17

    # MediaPipe Pose landmark indices used in the mapping.
    _NOSE = 0
    _L_EAR, _R_EAR = 7, 8
    _L_SHOULDER, _R_SHOULDER = 11, 12
    _L_ELBOW, _R_ELBOW = 13, 14
    _L_WRIST, _R_WRIST = 15, 16
    _L_HIP, _R_HIP = 23, 24
    _L_KNEE, _R_KNEE = 25, 26
    _L_ANKLE, _R_ANKLE = 27, 28

    def __init__(self, model_complexity: int = 2):
        try:
            import mediapipe as mp
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "mediapipe is required for SkeletonExtractor. "
                "Install with `pip install mediapipe`."
            ) from e
        self.mp = mp
        # Holistic exposes the same 33-landmark pose model; we only use pose.
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _mid(a, b):
        return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

    def _pose_to_h36m(self, pose: List[List[float]]) -> List[List[float]]:
        """Map 33 MediaPipe pose (x, y) to the 17-joint H36M order."""
        pelvis = self._mid(pose[self._L_HIP], pose[self._R_HIP])
        thorax = self._mid(pose[self._L_SHOULDER], pose[self._R_SHOULDER])
        spine = self._mid(pelvis, thorax)
        head = self._mid(pose[self._L_EAR], pose[self._R_EAR])
        return [
            pelvis,                       # 0 Hip
            pose[self._R_HIP],            # 1 RHip
            pose[self._R_KNEE],           # 2 RKnee
            pose[self._R_ANKLE],          # 3 RAnkle
            pose[self._L_HIP],            # 4 LHip
            pose[self._L_KNEE],           # 5 LKnee
            pose[self._L_ANKLE],          # 6 LAnkle
            spine,                        # 7 Spine
            thorax,                       # 8 Thorax
            pose[self._NOSE],             # 9 Neck/Nose
            head,                         # 10 Head
            pose[self._L_SHOULDER],       # 11 LShoulder
            pose[self._L_ELBOW],          # 12 LElbow
            pose[self._L_WRIST],          # 13 LWrist
            pose[self._R_SHOULDER],       # 14 RShoulder
            pose[self._R_ELBOW],          # 15 RElbow
            pose[self._R_WRIST],          # 16 RWrist
        ]

    def extract_keypoints(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(rgb)

        if res.pose_landmarks:
            pose = [[lm.x, lm.y] for lm in res.pose_landmarks.landmark]
            h36m = self._pose_to_h36m(pose)
        else:
            h36m = [[0.0, 0.0]] * self.TOTAL_JOINTS

        arr = np.asarray(h36m, dtype=np.float32)
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

        sk = np.stack(seq, axis=0)                       # (T, 17, 2)
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
