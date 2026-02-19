"""
vision/stability.py
===================
Head stability tracking via nose-tip position variance.

Tracks the displacement of a reference landmark (nose tip) over a sliding
window and computes the variance as a stability deviation score.  High
variance indicates restless or erratic head movement — a cognitive fatigue
indicator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from utils.smoothing import SlidingWindowSmoother


@dataclass
class StabilityState:
    """Snapshot of stability metrics for a single frame."""
    position_x: float = 0.0
    position_y: float = 0.0
    displacement: float = 0.0
    variance: float = 0.0
    stability_score: float = 0.0   # Normalised [0, 1]; higher → less stable


class StabilityTracker:
    """
    Tracks head stability using nose-tip position variance over a sliding window.

    The stability deviation score is computed as:
        variance = Var(displacement_window)
        stability_score = min(variance / max_variance, 1.0)

    Parameters
    ----------
    landmark_index : int
        MediaPipe landmark index to track (default: 1 = nose tip).
    window_size : int
        Number of frames in the sliding window.
    smoothing_window : int
        Moving-average window for smoothing the stability score.
    max_variance : float
        Variance value that maps to a stability score of 1.0 (fully unstable).
        Tune based on expected pixel movement range.
    """

    def __init__(
        self,
        landmark_index: int = 1,
        window_size: int = 30,
        smoothing_window: int = 10,
        max_variance: float = 200.0,
    ) -> None:
        self._lm_idx = landmark_index
        self._positions: deque[np.ndarray] = deque(maxlen=window_size)
        self._displacements: deque[float] = deque(maxlen=window_size)
        self._smoother = SlidingWindowSmoother(smoothing_window)
        self._max_variance = max_variance

        self._prev_pos: np.ndarray | None = None

        # Calibration
        self._calibration_samples: list[float] = []
        self._baseline_variance: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks: np.ndarray) -> StabilityState:
        """
        Update stability metrics from the current frame's landmarks.

        Parameters
        ----------
        landmarks : np.ndarray of shape (468, 2)
            Full set of MediaPipe facial landmarks in pixel space.

        Returns
        -------
        StabilityState
            Stability metrics for this frame.
        """
        pos = landmarks[self._lm_idx].copy()  # (x, y)
        self._positions.append(pos)

        # Compute displacement from previous position
        if self._prev_pos is not None:
            disp = float(np.linalg.norm(pos - self._prev_pos))
        else:
            disp = 0.0
        self._prev_pos = pos
        self._displacements.append(disp)

        # Variance of displacements over the window
        if len(self._displacements) >= 2:
            variance = float(np.var(list(self._displacements)))
        else:
            variance = 0.0

        # Normalise variance to [0, 1]
        max_var = self._max_variance
        if self._baseline_variance is not None:
            # Subtract baseline to measure deviation above normal movement
            excess = max(variance - self._baseline_variance, 0.0)
            raw_score = excess / max(max_var - self._baseline_variance, 1.0)
        else:
            raw_score = variance / max_var

        raw_score = min(raw_score, 1.0)
        stability_score = self._smoother.update(raw_score)

        return StabilityState(
            position_x=float(pos[0]),
            position_y=float(pos[1]),
            displacement=disp,
            variance=variance,
            stability_score=float(stability_score),
        )

    def add_calibration_sample(self, landmarks: np.ndarray) -> None:
        """Collect a displacement sample during the calibration phase."""
        pos = landmarks[self._lm_idx].copy()
        if self._prev_pos is not None:
            disp = float(np.linalg.norm(pos - self._prev_pos))
            self._calibration_samples.append(disp)
        self._prev_pos = pos

    def finalise_calibration(self) -> float | None:
        """
        Compute the baseline variance from calibration samples.

        Returns
        -------
        float or None
            Baseline variance, or None if insufficient samples.
        """
        if len(self._calibration_samples) < 2:
            return None
        self._baseline_variance = float(np.var(self._calibration_samples))
        self._calibration_samples.clear()
        return self._baseline_variance

    def reset(self) -> None:
        """Clear all tracked positions and state."""
        self._positions.clear()
        self._displacements.clear()
        self._prev_pos = None
        self._smoother.reset()
