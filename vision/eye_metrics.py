"""
vision/eye_metrics.py
=====================
Eye Aspect Ratio (EAR) computation, blink detection, and blink-rate tracking.

EAR formula (Soukupová & Čech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

where p1..p6 are the six eye landmarks ordered as:
    p1 = outer corner
    p2 = upper-outer lid
    p3 = upper-inner lid
    p4 = inner corner
    p5 = lower-inner lid
    p6 = lower-outer lid
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np

from utils.math_utils import compute_ear
from utils.smoothing import SlidingWindowSmoother


@dataclass
class EyeState:
    """Snapshot of eye metrics for a single frame."""
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    ear_smoothed: float = 0.0
    blink_count: int = 0
    blinks_per_minute: float = 0.0
    eyes_closed: bool = False
    closed_frames: int = 0
    drowsy: bool = False


class EyeMetrics:
    """
    Tracks eye aspect ratio, blink events, and blink frequency.

    Parameters
    ----------
    left_eye_indices : list[int]
        Six MediaPipe landmark indices for the left eye.
    right_eye_indices : list[int]
        Six MediaPipe landmark indices for the right eye.
    ear_threshold : float
        EAR below this value → eye considered closed.
    consec_frames_blink : int
        Minimum consecutive closed frames to register a blink.
    consec_frames_drowsy : int
        Consecutive closed frames that trigger a drowsiness flag.
    smoothing_window : int
        Moving-average window size for EAR smoothing.
    blink_rate_window_sec : float
        Time window (seconds) for computing blinks-per-minute.
    """

    def __init__(
        self,
        left_eye_indices: List[int],
        right_eye_indices: List[int],
        ear_threshold: float = 0.21,
        consec_frames_blink: int = 3,
        consec_frames_drowsy: int = 48,
        smoothing_window: int = 5,
        blink_rate_window_sec: float = 60.0,
    ) -> None:
        self._left_idx  = left_eye_indices
        self._right_idx = right_eye_indices
        self._ear_thresh = ear_threshold
        self._blink_frames = consec_frames_blink
        self._drowsy_frames = consec_frames_drowsy

        self._smoother = SlidingWindowSmoother(smoothing_window)

        # Blink tracking
        self._closed_frame_count: int = 0
        self._blink_count: int = 0
        self._was_closed: bool = False

        # Blink timestamps for rate calculation
        self._blink_timestamps: deque[float] = deque()
        self._blink_rate_window = blink_rate_window_sec

        # Calibration baseline
        self._baseline_ear: float | None = None
        self._calibration_samples: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks: np.ndarray) -> EyeState:
        """
        Compute eye metrics from the current frame's landmarks.

        Parameters
        ----------
        landmarks : np.ndarray of shape (468, 2)
            Full set of MediaPipe facial landmarks in pixel space.

        Returns
        -------
        EyeState
            All eye-related metrics for this frame.
        """
        # Extract 6-point arrays for each eye
        left_pts  = landmarks[self._left_idx]   # shape (6, 2)
        right_pts = landmarks[self._right_idx]  # shape (6, 2)

        ear_left  = compute_ear(left_pts)
        ear_right = compute_ear(right_pts)
        ear_avg   = (ear_left + ear_right) / 2.0

        ear_smoothed = self._smoother.update(ear_avg)

        # Determine closure using smoothed EAR
        eyes_closed = ear_smoothed < self._ear_thresh

        # Update consecutive closed-frame counter
        if eyes_closed:
            self._closed_frame_count += 1
        else:
            # Transition: closed → open → blink registered
            if self._closed_frame_count >= self._blink_frames:
                self._register_blink()
            self._closed_frame_count = 0

        drowsy = self._closed_frame_count >= self._drowsy_frames

        state = EyeState(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            ear_smoothed=ear_smoothed,
            blink_count=self._blink_count,
            blinks_per_minute=self._compute_blink_rate(),
            eyes_closed=eyes_closed,
            closed_frames=self._closed_frame_count,
            drowsy=drowsy,
        )
        return state

    def add_calibration_sample(self, landmarks: np.ndarray) -> None:
        """
        Collect an EAR sample during the calibration phase.

        Parameters
        ----------
        landmarks : np.ndarray of shape (468, 2)
        """
        left_pts  = landmarks[self._left_idx]
        right_pts = landmarks[self._right_idx]
        ear = (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0
        self._calibration_samples.append(ear)

    def finalise_calibration(self) -> float | None:
        """
        Compute and store the baseline EAR from calibration samples.

        Returns
        -------
        float or None
            Baseline EAR, or None if no samples were collected.
        """
        if not self._calibration_samples:
            return None
        self._baseline_ear = float(np.mean(self._calibration_samples))
        # Adjust threshold relative to the personal baseline
        self._ear_thresh = self._baseline_ear * 0.75
        self._calibration_samples.clear()
        return self._baseline_ear

    @property
    def baseline_ear(self) -> float | None:
        """Personal EAR baseline established during calibration."""
        return self._baseline_ear

    def reset(self) -> None:
        """Reset all counters and state."""
        self._closed_frame_count = 0
        self._blink_count = 0
        self._was_closed = False
        self._blink_timestamps.clear()
        self._smoother.reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_blink(self) -> None:
        """Record a blink event with the current timestamp."""
        self._blink_count += 1
        self._blink_timestamps.append(time.monotonic())

    def _compute_blink_rate(self) -> float:
        """
        Compute blinks per minute using a sliding time window.

        Returns
        -------
        float
            Blinks per minute.
        """
        now = time.monotonic()
        cutoff = now - self._blink_rate_window

        # Remove timestamps outside the window
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        count = len(self._blink_timestamps)
        # Scale to per-minute
        return count * (60.0 / self._blink_rate_window)
