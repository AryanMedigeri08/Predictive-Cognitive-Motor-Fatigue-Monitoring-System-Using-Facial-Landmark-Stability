"""
vision/face_mesh.py
===================
MediaPipe Face Mesh wrapper — compatible with mediapipe 0.9.x through 0.10.x+.

MediaPipe 0.10+ moved from the legacy ``mp.solutions`` API to the new
``mediapipe.tasks`` API.  This module auto-detects which API is available
and uses the appropriate backend transparently.

New-API backend (mediapipe >= 0.10):
  Downloads ``face_landmarker.task`` model on first run (~30 MB).
  Model is cached in the project directory for subsequent runs.

Legacy-API backend (mediapipe < 0.10 / 0.10.x with solutions):
  Uses ``mp.solutions.face_mesh`` directly.
"""

from __future__ import annotations

import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional


# ---------------------------------------------------------------------------
# Detect which MediaPipe API is available
# ---------------------------------------------------------------------------
def _has_solutions() -> bool:
    """Return True if the legacy mp.solutions API is available."""
    try:
        _ = mp.solutions.face_mesh  # noqa: F841
        return True
    except AttributeError:
        return False


_USE_LEGACY_API = _has_solutions()

# Model URL for the new Tasks API
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
_MODEL_PATH = os.path.normpath(_MODEL_PATH)


def _ensure_model() -> str:
    """Download the face landmarker model if not already present."""
    if not os.path.exists(_MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe face landmarker model (~30 MB)...")
        print(f"[INFO] Saving to: {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model downloaded successfully.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Unified result wrapper so downstream code is API-agnostic
# ---------------------------------------------------------------------------
class _FaceResult:
    """Thin wrapper that normalises legacy and new API results."""
    def __init__(self, multi_face_landmarks):
        self.multi_face_landmarks = multi_face_landmarks


class FaceMeshDetector:
    """
    Wraps MediaPipe FaceMesh for single-face landmark detection.

    Automatically uses the legacy ``mp.solutions`` API if available,
    otherwise falls back to the new ``mediapipe.tasks`` FaceLandmarker API.

    Parameters
    ----------
    max_faces : int
        Maximum number of faces to detect.
    refine_landmarks : bool
        Whether to refine iris/lip landmarks (legacy API only; new API always refines).
    min_detection_confidence : float
        Minimum confidence for face detection.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.
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

    def _init_tasks_api(
        self,
        max_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """Initialise the new mediapipe.tasks FaceLandmarker."""
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        model_path = _ensure_model()

        base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[object]]:
        """
        Detect facial landmarks in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.

        Returns
        -------
        landmarks_px : np.ndarray of shape (478, 2) or None
            Pixel-space (x, y) coordinates for each landmark.
            None if no face is detected.
            Note: new API returns 478 landmarks (468 + 10 iris); legacy returns 468.
        raw_results : result object or None
        """
        h, w = frame.shape[:2]

        if self._backend == "legacy":
            return self._process_legacy(frame, h, w)
        else:
            return self._process_tasks(frame, h, w)

    def _process_legacy(self, frame, h, w):
        """Process using the legacy mp.solutions API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
            dtype=np.float64,
        )
        return landmarks_px, results

    def _process_tasks(self, frame, h, w):
        """Process using the new mediapipe.tasks API."""
        import mediapipe as mp_mod
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None, None

        face_landmarks = results.face_landmarks[0]
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks],
            dtype=np.float64,
        )
        return landmarks_px, results

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results: object,
        draw_tesselation: bool = False,
        draw_contours: bool = True,
    ) -> np.ndarray:
        """
        Draw face-mesh landmarks onto *frame* (in-place).

        Only functional with the legacy API; silently skipped for the new API
        (drawing utilities are not yet standardised in mediapipe.tasks).
        """
        if results is None or self._backend != "legacy":
            return frame
        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:
            if draw_tesselation:
                self._mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles
                    .get_default_face_mesh_tesselation_style(),
                )
            if draw_contours:
                self._mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles
                    .get_default_face_mesh_contours_style(),
                )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._backend == "legacy":
            self._face_mesh.close()
        else:
            self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
