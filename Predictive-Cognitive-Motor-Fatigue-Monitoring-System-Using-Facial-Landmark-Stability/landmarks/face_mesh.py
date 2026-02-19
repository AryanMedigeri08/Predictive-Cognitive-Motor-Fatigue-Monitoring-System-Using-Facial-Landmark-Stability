"""
landmarks/face_mesh.py
======================
MediaPipe Face Mesh wrapper — compatible with 0.9.x through 0.10.x+.

Enhancements over the legacy version:
  - Returns **3-D** landmark coordinates (x, y, z) for gaze / neck
    vector computation.
  - Exposes **per-region landmark confidence** so the confidence-scoring
    module can weight features by detection reliability.
  - Auto-detects legacy ``mp.solutions`` vs new ``mediapipe.tasks`` API.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# API Detection
# ──────────────────────────────────────────────────────────────────────

def _has_solutions() -> bool:
    """Return True if the legacy mp.solutions API is available."""
    try:
        _ = mp.solutions.face_mesh  # noqa: F841
        return True
    except AttributeError:
        return False


_USE_LEGACY_API = _has_solutions()

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
)


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"[INFO] Downloading face landmarker model (~30 MB) → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Download complete.")
    return _MODEL_PATH


# ──────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LandmarkResult:
    """Unified output from the face-mesh detector."""
    landmarks_px: Optional[np.ndarray] = None   # (N, 2) pixel coords
    landmarks_3d: Optional[np.ndarray] = None   # (N, 3) normalised coords
    raw_results: Optional[object] = None
    landmark_confidence: float = 1.0             # mean visibility / presence


# ──────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────

class FaceMeshDetector:
    """
    Wraps MediaPipe FaceMesh for single-face landmark detection.

    Parameters
    ----------
    max_faces : int
    refine_landmarks : bool  — enables iris landmarks (478 total)
    min_detection_confidence : float
    min_tracking_confidence : float
    """

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._max_faces = max_faces

        if _USE_LEGACY_API:
            self._backend = "legacy"
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            print("[INFO] MediaPipe: using legacy solutions API.")
        else:
            self._backend = "tasks"
            self._init_tasks_api(
                max_faces, min_detection_confidence, min_tracking_confidence
            )
            print("[INFO] MediaPipe: using new Tasks API.")

    def _init_tasks_api(self, max_faces, min_det, min_track) -> None:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        model_path = _ensure_model()
        base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )
        self._landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> LandmarkResult:
        """
        Detect facial landmarks in *frame* (BGR uint8).

        Returns
        -------
        LandmarkResult
            .landmarks_px : (N, 2) pixel-space coordinates
            .landmarks_3d : (N, 3) normalised [0-1] coordinates with depth
            .landmark_confidence : mean visibility score [0-1]
        """
        h, w = frame.shape[:2]
        if self._backend == "legacy":
            return self._process_legacy(frame, h, w)
        return self._process_tasks(frame, h, w)

    def _process_legacy(self, frame, h, w) -> LandmarkResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return LandmarkResult()

        face = results.multi_face_landmarks[0]

        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face.landmark],
            dtype=np.float64,
        )
        landmarks_3d = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark],
            dtype=np.float64,
        )

        # Visibility is an optional field; default to 1.0 if absent
        confidences = []
        for lm in face.landmark:
            vis = getattr(lm, "visibility", None)
            confidences.append(vis if vis is not None else 1.0)
        mean_conf = float(np.mean(confidences))

        return LandmarkResult(
            landmarks_px=landmarks_px,
            landmarks_3d=landmarks_3d,
            raw_results=results,
            landmark_confidence=mean_conf,
        )

    def _process_tasks(self, frame, h, w) -> LandmarkResult:
        import mediapipe as mp_mod
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return LandmarkResult()

        face = results.face_landmarks[0]
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face],
            dtype=np.float64,
        )
        landmarks_3d = np.array(
            [(lm.x, lm.y, lm.z) for lm in face],
            dtype=np.float64,
        )

        confidences = []
        for lm in face:
            vis = getattr(lm, "visibility", None) or getattr(lm, "presence", None)
            confidences.append(vis if vis is not None else 1.0)
        mean_conf = float(np.mean(confidences))

        return LandmarkResult(
            landmarks_px=landmarks_px,
            landmarks_3d=landmarks_3d,
            raw_results=results,
            landmark_confidence=mean_conf,
        )

    # ──────────────────────────────────────────────────────────────────
    # Region-specific confidence
    # ──────────────────────────────────────────────────────────────────

    def get_region_confidence(
        self,
        landmarks_3d: np.ndarray,
        indices: list[int],
    ) -> float:
        """
        Compute mean visibility / presence for a subset of landmarks.

        Used by the glasses-handling module to assess eye-region
        reliability separately from the full-face confidence.
        """
        if landmarks_3d is None or len(landmarks_3d) == 0:
            return 0.0
        # z-component magnitude as a proxy for depth confidence
        # Landmarks with large z-deviations typically have lower confidence
        subset = landmarks_3d[indices]
        z_var = float(np.var(subset[:, 2]))
        # Map z-variance to [0, 1] confidence (lower variance → higher conf)
        conf = max(0.0, 1.0 - z_var * 100.0)
        return min(conf, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._backend == "legacy":
            self._face_mesh.close()
        else:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
