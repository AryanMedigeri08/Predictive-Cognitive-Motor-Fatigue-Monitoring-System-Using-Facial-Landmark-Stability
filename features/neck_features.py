"""
features/neck_features.py
=========================
Neck vector stability tracking.

The neck orientation is approximated by the normalised vector from the
chin landmark (152) to the forehead landmark (10).  Stability is
measured as the variance of this vector's angular displacement over a
sliding window.

Neck Stability Score:
    deviation_t = angular_deviation(neck_vec_t, Neck_0)
    variance    = Var(deviations in window)
    score       = clamp(variance / max_variance, 0, 1)

Higher score → less stable neck → potential drowsiness or distraction.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.math_utils import compute_neck_vector, angular_deviation, clamp
from utils.smoothing import SlidingWindowSmoother


@dataclass
class NeckFeatures:
    """Per-frame neck stability snapshot."""
    timestamp: float = 0.0
    neck_vector: Optional[np.ndarray] = None
    deviation_from_baseline: float = 0.0   # degrees
    stability_score: float = 0.0           # [0, 1]


class NeckFeatureExtractor:
    """
    Neck vector stability tracker.

    Parameters
    ----------
    chin_index : int
        MediaPipe landmark index for chin.
    forehead_index : int
        MediaPipe landmark index for forehead.
    window_size : int
        Sliding window (frames) for variance calculation.
    smoothing_window : int
        Moving average window for the stability score.
    max_variance : float
        Variance that maps to score 1.0 (fully unstable).
    """

    def __init__(
        self,
        chin_index: int = 152,
        forehead_index: int = 10,
        window_size: int = 30,
        smoothing_window: int = 10,
        max_variance: float = 25.0,
    ) -> None:
        self._chin_idx = chin_index
        self._forehead_idx = forehead_index
        self._deviations: deque[float] = deque(maxlen=window_size)
        self._smoother = SlidingWindowSmoother(smoothing_window)
        self._max_variance = max_variance

        # Baseline
        self._neck_0: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def set_baseline(self, neck_0: np.ndarray) -> None:
        """Store calibration baseline Neck_0."""
        self._neck_0 = neck_0.copy()

    # ──────────────────────────────────────────────────────────────────
    # Per-frame update
    # ──────────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> NeckFeatures:
        now = time.monotonic()

        chin = landmarks[self._chin_idx]
        forehead = landmarks[self._forehead_idx]
        neck_vec = compute_neck_vector(chin, forehead)

        # Angular deviation from baseline
        dev = 0.0
        if self._neck_0 is not None:
            dev = angular_deviation(neck_vec, self._neck_0)
        self._deviations.append(dev)

        # Variance of deviations in the window
        if len(self._deviations) >= 2:
            var = float(np.var(list(self._deviations)))
        else:
            var = 0.0

        raw_score = clamp(var / self._max_variance, 0.0, 1.0)
        stability_score = self._smoother.update(raw_score)

        return NeckFeatures(
            timestamp=now,
            neck_vector=neck_vec,
            deviation_from_baseline=dev,
            stability_score=stability_score,
        )

    def compute_raw_neck_vector(self, landmarks: np.ndarray) -> np.ndarray:
        """Raw neck vector (for calibration)."""
        return compute_neck_vector(
            landmarks[self._chin_idx],
            landmarks[self._forehead_idx],
        )

    def reset(self) -> None:
        self._deviations.clear()
        self._smoother.reset()
