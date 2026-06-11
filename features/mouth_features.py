"""
features/mouth_features.py
==========================
Mouth Opening Index (MOI) computation and yawn detection.

MOI is the MAR normalised to the calibration baseline:
    MOI = MAR_t / MAR_0

A sustained high MOI indicates yawning — a clear physical fatigue
indicator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.math_utils import compute_mar
from utils.smoothing import SlidingWindowSmoother


@dataclass
class MouthFeatures:
    """Per-frame mouth feature snapshot."""
    timestamp: float = 0.0
    mar: float = 0.0
    mar_smoothed: float = 0.0
    mouth_opening_index: float = 0.0   # MAR_t / MAR_0
    is_yawning: bool = False
    yawn_count: int = 0
    open_frames: int = 0


class MouthFeatureExtractor:
    """
    Tracks Mouth Aspect Ratio and detects yawning.

    Parameters
    ----------
    vertical_indices : list[int]
        [top_lip, bottom_lip] landmark indices.
    horizontal_indices : list[int]
        [left_corner, right_corner] landmark indices.
    yawn_ratio : float
        MOI above this multiple of baseline → yawn.
    consec_frames_yawn : int
        Min consecutive open frames to confirm a yawn.
    smoothing_window : int
        Moving-average window for MAR smoothing.
    """

    def __init__(
        self,
        vertical_indices: list[int],
        horizontal_indices: list[int],
        yawn_ratio: float = 1.5,
        consec_frames_yawn: int = 20,
        smoothing_window: int = 5,
    ) -> None:
        self._vert_idx = vertical_indices
        self._horiz_idx = horizontal_indices
        self._yawn_ratio = yawn_ratio
        self._yawn_frames = consec_frames_yawn
        self._smoother = SlidingWindowSmoother(smoothing_window)

        self._mar_0: Optional[float] = None
        self._mar_threshold: float = 0.6  # fallback

        self._open_count: int = 0
        self._yawn_count: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def set_baseline(self, mar_0: float) -> None:
        """Set the personal MAR baseline.  Yawn threshold = MAR_0 × yawn_ratio."""
        self._mar_0 = mar_0
        self._mar_threshold = mar_0 * self._yawn_ratio

    # ──────────────────────────────────────────────────────────────────
    # Per-frame update
    # ──────────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> MouthFeatures:
        now = time.monotonic()

        top    = landmarks[self._vert_idx[0]]
        bottom = landmarks[self._vert_idx[1]]
        left   = landmarks[self._horiz_idx[0]]
        right  = landmarks[self._horiz_idx[1]]

        mouth_pts = np.array([top, bottom, left, right])
        mar = compute_mar(mouth_pts)
        mar_smoothed = self._smoother.update(mar)

        # Mouth Opening Index: normalised to baseline
        moi = mar_smoothed / self._mar_0 if self._mar_0 else mar_smoothed

        mouth_open = mar_smoothed > self._mar_threshold

        if mouth_open:
            self._open_count += 1
        else:
            if self._open_count >= self._yawn_frames:
                self._yawn_count += 1
            self._open_count = 0

        is_yawning = self._open_count >= self._yawn_frames

        return MouthFeatures(
            timestamp=now,
            mar=mar,
            mar_smoothed=mar_smoothed,
            mouth_opening_index=moi,
            is_yawning=is_yawning,
            yawn_count=self._yawn_count,
            open_frames=self._open_count,
        )

    def compute_raw_mar(self, landmarks: np.ndarray) -> float:
        """Raw MAR without state update (for calibration)."""
        top    = landmarks[self._vert_idx[0]]
        bottom = landmarks[self._vert_idx[1]]
        left   = landmarks[self._horiz_idx[0]]
        right  = landmarks[self._horiz_idx[1]]
        return compute_mar(np.array([top, bottom, left, right]))

    def reset(self) -> None:
        self._open_count = 0
        self._yawn_count = 0
        self._smoother.reset()
