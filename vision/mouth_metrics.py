"""
vision/mouth_metrics.py
=======================
Mouth Aspect Ratio (MAR) computation and yawn detection.

MAR = vertical_distance / horizontal_distance

A high MAR sustained over several frames indicates a yawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.math_utils import compute_mar
from utils.smoothing import SlidingWindowSmoother


@dataclass
class MouthState:
    """Snapshot of mouth metrics for a single frame."""
    mar: float = 0.0
    mar_smoothed: float = 0.0
    is_yawning: bool = False
    yawn_count: int = 0
    open_frames: int = 0


class MouthMetrics:
    """
    Tracks Mouth Aspect Ratio and detects yawning events.

    Parameters
    ----------
    vertical_indices : list[int]
        Two MediaPipe landmark indices: [top_lip, bottom_lip].
    horizontal_indices : list[int]
        Two MediaPipe landmark indices: [left_corner, right_corner].
    mar_threshold : float
        MAR above this value → mouth considered open (possible yawn).
    consec_frames_yawn : int
        Minimum consecutive open frames to confirm a yawn.
    smoothing_window : int
        Moving-average window size for MAR smoothing.
    """

    def __init__(
        self,
        vertical_indices: List[int],
        horizontal_indices: List[int],
        mar_threshold: float = 0.6,
        consec_frames_yawn: int = 20,
        smoothing_window: int = 5,
    ) -> None:
        self._vert_idx  = vertical_indices
        self._horiz_idx = horizontal_indices
        self._mar_thresh = mar_threshold
        self._yawn_frames = consec_frames_yawn

        self._smoother = SlidingWindowSmoother(smoothing_window)

        self._open_frame_count: int = 0
        self._yawn_count: int = 0
        self._was_open: bool = False

        # Calibration
        self._calibration_samples: List[float] = []
        self._baseline_mar: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks: np.ndarray) -> MouthState:
        """
        Compute mouth metrics from the current frame's landmarks.

        Parameters
        ----------
        landmarks : np.ndarray of shape (468, 2)
            Full set of MediaPipe facial landmarks in pixel space.

        Returns
        -------
        MouthState
            All mouth-related metrics for this frame.
        """
        # Build the 4-point array expected by compute_mar:
        # [top_lip, bottom_lip, left_corner, right_corner]
        top_lip      = landmarks[self._vert_idx[0]]
        bottom_lip   = landmarks[self._vert_idx[1]]
        left_corner  = landmarks[self._horiz_idx[0]]
        right_corner = landmarks[self._horiz_idx[1]]

        mouth_pts = np.array([top_lip, bottom_lip, left_corner, right_corner])
        mar = compute_mar(mouth_pts)
        mar_smoothed = self._smoother.update(mar)

        mouth_open = mar_smoothed > self._mar_thresh

        if mouth_open:
            self._open_frame_count += 1
        else:
            # Transition: open → closed → yawn registered
            if self._open_frame_count >= self._yawn_frames:
                self._yawn_count += 1
            self._open_frame_count = 0

        is_yawning = self._open_frame_count >= self._yawn_frames

        return MouthState(
            mar=mar,
            mar_smoothed=mar_smoothed,
            is_yawning=is_yawning,
            yawn_count=self._yawn_count,
            open_frames=self._open_frame_count,
        )

    def add_calibration_sample(self, landmarks: np.ndarray) -> None:
        """Collect a MAR sample during the calibration phase."""
        top_lip      = landmarks[self._vert_idx[0]]
        bottom_lip   = landmarks[self._vert_idx[1]]
        left_corner  = landmarks[self._horiz_idx[0]]
        right_corner = landmarks[self._horiz_idx[1]]
        mouth_pts = np.array([top_lip, bottom_lip, left_corner, right_corner])
        self._calibration_samples.append(compute_mar(mouth_pts))

    def finalise_calibration(self) -> float | None:
        """
        Compute and store the baseline MAR from calibration samples.

        Returns
        -------
        float or None
        """
        if not self._calibration_samples:
            return None
        self._baseline_mar = float(np.mean(self._calibration_samples))
        # Threshold = baseline + 50% of baseline (adaptive)
        self._mar_thresh = max(self._baseline_mar * 1.5, self._mar_thresh)
        self._calibration_samples.clear()
        return self._baseline_mar

    @property
    def yawn_count(self) -> int:
        """Total number of yawns detected since initialisation."""
        return self._yawn_count

    def reset(self) -> None:
        """Reset all counters and state."""
        self._open_frame_count = 0
        self._yawn_count = 0
        self._was_open = False
        self._smoother.reset()
