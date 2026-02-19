"""
features/head_features.py
=========================
Head pose estimation and deviation tracking relative to calibration
baseline H_0 = (pitch_0, yaw_0, roll_0).

Uses OpenCV solvePnP with 6 facial landmarks mapped to a generic 3-D
face model.  Kalman filters smooth each Euler angle independently.

Head Deviation Score:
    For each axis a ∈ {pitch, yaw, roll}:
        d_a = |angle_a − baseline_a| / threshold_a

    deviation = max(d_pitch, d_yaw, d_roll),  clamped to [0, 1]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from utils.math_utils import rotation_matrix_to_euler_angles, clamp
from utils.smoothing import KalmanFilter1D


@dataclass
class HeadFeatures:
    """Per-frame head pose snapshot."""
    timestamp: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    pitch_raw: float = 0.0
    yaw_raw: float = 0.0
    roll_raw: float = 0.0
    deviation_score: float = 0.0   # [0, 1]  higher → more deviated
    rvec: Optional[np.ndarray] = None   # for axis drawing
    tvec: Optional[np.ndarray] = None


class HeadFeatureExtractor:
    """
    Head pose estimation via PnP and deviation scoring.

    Parameters
    ----------
    model_points_3d : list of (x, y, z)
        Generic 3-D face model coordinates (mm scale).
    landmark_indices : list[int]
        MediaPipe indices corresponding to model points.
    pitch/yaw/roll_threshold : float
        Degrees of deviation that map to a deviation score of 1.0.
    """

    def __init__(
        self,
        model_points_3d: list[tuple[float, float, float]],
        landmark_indices: list[int],
        pitch_threshold: float = 15.0,
        yaw_threshold: float = 20.0,
        roll_threshold: float = 20.0,
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
    ) -> None:
        self._model_pts = np.array(model_points_3d, dtype=np.float64)
        self._lm_indices = landmark_indices
        self._pitch_thresh = pitch_threshold
        self._yaw_thresh = yaw_threshold
        self._roll_thresh = roll_threshold

        kw = dict(process_noise=kalman_process_noise,
                  measurement_noise=kalman_measurement_noise)
        self._kf_pitch = KalmanFilter1D(**kw)
        self._kf_yaw = KalmanFilter1D(**kw)
        self._kf_roll = KalmanFilter1D(**kw)

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Baseline
        self._baseline_pitch: float = 0.0
        self._baseline_yaw: float = 0.0
        self._baseline_roll: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def set_baseline(self, pitch: float, yaw: float, roll: float) -> None:
        """Store the calibration baseline H_0."""
        self._baseline_pitch = pitch
        self._baseline_yaw = yaw
        self._baseline_roll = roll

    # ──────────────────────────────────────────────────────────────────
    # Per-frame update
    # ──────────────────────────────────────────────────────────────────

    def update(
        self,
        landmarks: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> HeadFeatures:
        now = time.monotonic()
        h, w = frame_shape

        if self._camera_matrix is None:
            focal = float(w)
            cx, cy = w / 2.0, h / 2.0
            self._camera_matrix = np.array(
                [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
                dtype=np.float64,
            )

        image_pts = np.array(
            [landmarks[i] for i in self._lm_indices], dtype=np.float64
        )

        success, rvec, tvec = cv2.solvePnP(
            self._model_pts, image_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success:
            return HeadFeatures(timestamp=now)

        R, _ = cv2.Rodrigues(rvec)
        pitch_raw, yaw_raw, roll_raw = rotation_matrix_to_euler_angles(R)

        pitch = self._kf_pitch.update(pitch_raw)
        yaw = self._kf_yaw.update(yaw_raw)
        roll = self._kf_roll.update(roll_raw)

        # Deviation from calibration baseline H_0
        dp = abs(pitch - self._baseline_pitch) / max(self._pitch_thresh, 1.0)
        dy = abs(yaw - self._baseline_yaw) / max(self._yaw_thresh, 1.0)
        dr = abs(roll - self._baseline_roll) / max(self._roll_thresh, 1.0)
        deviation = clamp(max(dp, dy, dr), 0.0, 1.0)

        return HeadFeatures(
            timestamp=now,
            pitch=pitch, yaw=yaw, roll=roll,
            pitch_raw=pitch_raw, yaw_raw=yaw_raw, roll_raw=roll_raw,
            deviation_score=deviation,
            rvec=rvec, tvec=tvec,
        )

    def compute_raw_euler(
        self, landmarks: np.ndarray, frame_shape: tuple[int, int],
    ) -> tuple[float, float, float]:
        """Raw Euler angles without Kalman (for calibration aggregation)."""
        h, w = frame_shape
        if self._camera_matrix is None:
            focal = float(w)
            self._camera_matrix = np.array(
                [[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]],
                dtype=np.float64,
            )
        image_pts = np.array(
            [landmarks[i] for i in self._lm_indices], dtype=np.float64
        )
        ok, rvec, _ = cv2.solvePnP(
            self._model_pts, image_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        R, _ = cv2.Rodrigues(rvec)
        return rotation_matrix_to_euler_angles(R)

    # ──────────────────────────────────────────────────────────────────
    # Axis drawing (for visualization)
    # ──────────────────────────────────────────────────────────────────

    def draw_axes(
        self,
        frame: np.ndarray,
        head_features: HeadFeatures,
        axis_length: float = 100.0,
    ) -> np.ndarray:
        """Draw 3-D coordinate axes at the nose tip."""
        if head_features.rvec is None or self._camera_matrix is None:
            return frame

        rvec = head_features.rvec
        tvec = head_features.tvec

        axis_pts = np.float32([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ])
        nose_3d = np.float32([[0, 0, 0]])

        proj_nose, _ = cv2.projectPoints(
            nose_3d, rvec, tvec, self._camera_matrix, self._dist_coeffs)
        proj_axes, _ = cv2.projectPoints(
            axis_pts, rvec, tvec, self._camera_matrix, self._dist_coeffs)

        origin = tuple(proj_nose[0].ravel().astype(int))
        x_end = tuple(proj_axes[0].ravel().astype(int))
        y_end = tuple(proj_axes[1].ravel().astype(int))
        z_end = tuple(proj_axes[2].ravel().astype(int))

        cv2.line(frame, origin, x_end, (0, 0, 255), 2)   # X red
        cv2.line(frame, origin, y_end, (0, 255, 0), 2)   # Y green
        cv2.line(frame, origin, z_end, (255, 0, 0), 2)   # Z blue

        return frame

    def reset(self) -> None:
        self._kf_pitch.reset()
        self._kf_yaw.reset()
        self._kf_roll.reset()
