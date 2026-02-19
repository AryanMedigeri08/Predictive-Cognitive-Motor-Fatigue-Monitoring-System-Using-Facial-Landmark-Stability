"""
features/gaze_features.py
=========================
Gaze-related feature extraction for cognitive fatigue assessment.

Features:
  - **Gaze vector** — approximate 2-D direction from eye centroid to iris
  - **Gaze drift** — angular deviation of current gaze from baseline G_0
  - **Micro-saccade variance** — variance of gaze velocity over a short
    sliding window.  Elevated variance indicates involuntary, rapid eye
    movements characteristic of cognitive overload.

Mathematical notes
------------------
Gaze drift (angular deviation):
    θ = arccos(clamp(g_t · G_0, -1, 1))         [degrees]

Micro-saccade variance:
    velocity_t = ||g_t − g_{t−1}|| / dt
    σ² = Var(velocity window)

Both metrics are normalised to [0, 1] for the fatigue model.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.math_utils import compute_gaze_vector, angular_deviation, clamp


@dataclass
class GazeFeatures:
    """Per-frame gaze feature snapshot."""
    timestamp: float = 0.0
    gaze_vector_left: Optional[np.ndarray] = None
    gaze_vector_right: Optional[np.ndarray] = None
    gaze_vector_avg: Optional[np.ndarray] = None
    gaze_drift: float = 0.0            # [0, 1] normalised deviation from G_0
    microsaccade_variance: float = 0.0  # [0, 1] normalised


class GazeFeatureExtractor:
    """
    Gaze drift and micro-saccade variance extraction.

    Parameters
    ----------
    left_iris_indices, right_iris_indices : list[int]
        MediaPipe iris landmark indices (5-point: center + 4 boundary).
    left_eye_inner/outer, right_eye_inner/outer : int
        Inner and outer corner landmark indices for each eye.
    drift_window_sec : float
        Sliding window for drift averaging.
    microsaccade_window_sec : float
        Sliding window for micro-saccade velocity variance.
    max_drift_degrees : float
        Drift angle that maps to a normalised score of 1.0.
    max_saccade_variance : float
        Velocity variance that maps to a normalised score of 1.0.
    """

    def __init__(
        self,
        left_iris_indices: list[int],
        right_iris_indices: list[int],
        left_eye_inner: int = 263,
        left_eye_outer: int = 362,
        right_eye_inner: int = 133,
        right_eye_outer: int = 33,
        drift_window_sec: float = 5.0,
        microsaccade_window_sec: float = 2.0,
        max_drift_degrees: float = 15.0,
        max_saccade_variance: float = 50.0,
    ) -> None:
        self._l_iris = left_iris_indices
        self._r_iris = right_iris_indices
        self._l_inner = left_eye_inner
        self._l_outer = left_eye_outer
        self._r_inner = right_eye_inner
        self._r_outer = right_eye_outer

        self._drift_window = drift_window_sec
        self._saccade_window = microsaccade_window_sec
        self._max_drift = max_drift_degrees
        self._max_saccade = max_saccade_variance

        # Baseline gaze vector G_0
        self._gaze_0: Optional[np.ndarray] = None

        # History for micro-saccade variance
        self._prev_gaze: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._velocities: deque[tuple[float, float]] = deque()   # (time, velocity)

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def set_baseline(self, gaze_0: np.ndarray) -> None:
        """Store baseline gaze direction G_0."""
        self._gaze_0 = gaze_0.copy()

    # ──────────────────────────────────────────────────────────────────
    # Per-frame update
    # ──────────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> GazeFeatures:
        now = time.monotonic()

        # ── Compute gaze vectors for each eye ──
        # Iris center is index [0] of each iris set
        l_iris_center = landmarks[self._l_iris[0]]
        r_iris_center = landmarks[self._r_iris[0]]

        gaze_left = compute_gaze_vector(
            l_iris_center, landmarks[self._l_inner], landmarks[self._l_outer])
        gaze_right = compute_gaze_vector(
            r_iris_center, landmarks[self._r_inner], landmarks[self._r_outer])

        # Average gaze vector (binocular mean)
        gaze_avg = (gaze_left + gaze_right) / 2.0
        norm = np.linalg.norm(gaze_avg)
        if norm > 1e-10:
            gaze_avg = gaze_avg / norm

        # ── Gaze drift from baseline ──
        drift_deg = 0.0
        if self._gaze_0 is not None:
            drift_deg = angular_deviation(gaze_avg, self._gaze_0)

        drift_norm = clamp(drift_deg / self._max_drift, 0.0, 1.0)

        # ── Micro-saccade variance ──
        # Velocity = ||gaze_t − gaze_{t-1}|| / dt
        if self._prev_gaze is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                vel = float(np.linalg.norm(gaze_avg - self._prev_gaze)) / dt
                self._velocities.append((now, vel))

        self._prev_gaze = gaze_avg.copy()
        self._prev_time = now

        # Trim to window
        cutoff = now - self._saccade_window
        while self._velocities and self._velocities[0][0] < cutoff:
            self._velocities.popleft()

        # Variance of velocity within window
        if len(self._velocities) >= 2:
            vels = [v for _, v in self._velocities]
            saccade_var = float(np.var(vels))
        else:
            saccade_var = 0.0

        saccade_norm = clamp(saccade_var / self._max_saccade, 0.0, 1.0)

        return GazeFeatures(
            timestamp=now,
            gaze_vector_left=gaze_left,
            gaze_vector_right=gaze_right,
            gaze_vector_avg=gaze_avg,
            gaze_drift=drift_norm,
            microsaccade_variance=saccade_norm,
        )

    def compute_raw_gaze(self, landmarks: np.ndarray) -> np.ndarray:
        """Raw average gaze vector (for calibration aggregation)."""
        l_iris = landmarks[self._l_iris[0]]
        r_iris = landmarks[self._r_iris[0]]
        gl = compute_gaze_vector(l_iris, landmarks[self._l_inner], landmarks[self._l_outer])
        gr = compute_gaze_vector(r_iris, landmarks[self._r_inner], landmarks[self._r_outer])
        avg = (gl + gr) / 2.0
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-10 else avg

    def reset(self) -> None:
        self._prev_gaze = None
        self._prev_time = None
        self._velocities.clear()
