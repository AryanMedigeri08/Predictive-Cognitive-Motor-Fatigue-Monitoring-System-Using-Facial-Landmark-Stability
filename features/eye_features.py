"""
features/eye_features.py
========================
Eye-related feature extraction:
  - EAR (Eye Aspect Ratio) — relative to calibration baseline
  - PERCLOS (Percentage of Eye Closure over a sliding window)
  - Blink detection, blink duration, blink variance

Key design: **NO hardcoded EAR thresholds.**  All closure detection is
relative to the personalised baseline EAR_0:
    EAR_relative = EAR_t / EAR_0
    Closed if EAR_t < EAR_0 * EAR_CLOSURE_RATIO

Mathematical reference — PERCLOS (Dinges & Grace, 1998):
    PERCLOS = (frames where EAR < threshold) / (total frames in window)
    Typically measured over a 60-second sliding window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.math_utils import compute_ear
from utils.smoothing import SlidingWindowSmoother


@dataclass
class EyeFeatures:
    """Per-frame eye feature snapshot."""
    timestamp: float = 0.0

    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    ear_smoothed: float = 0.0
    ear_relative: float = 1.0       # EAR_t / EAR_0

    perclos: float = 0.0            # [0, 1] — proportion of eye closure
    blink_count: int = 0
    blinks_per_minute: float = 0.0
    blink_duration_ms: float = 0.0  # most recent blink duration
    blink_variance: float = 0.0     # std of recent blink durations

    eyes_closed: bool = False
    closed_frames: int = 0


class EyeFeatureExtractor:
    """
    Extracts all eye-related features using personalised baseline.

    Parameters
    ----------
    left_eye_indices, right_eye_indices : list[int]
        6-point landmark indices for each eye.
    ear_closure_ratio : float
        Fraction of EAR_0 below which the eye is considered closed.
    consec_frames_blink : int
        Minimum consecutive closed frames to register a blink.
    perclos_window_sec : float
        PERCLOS sliding window duration in seconds.
    smoothing_window : int
        Moving-average window for EAR smoothing.
    """

    def __init__(
        self,
        left_eye_indices: list[int],
        right_eye_indices: list[int],
        ear_closure_ratio: float = 0.75,
        consec_frames_blink: int = 3,
        perclos_window_sec: float = 60.0,
        smoothing_window: int = 5,
    ) -> None:
        self._left_idx = left_eye_indices
        self._right_idx = right_eye_indices
        self._closure_ratio = ear_closure_ratio
        self._blink_frames = consec_frames_blink
        self._perclos_window = perclos_window_sec
        self._smoother = SlidingWindowSmoother(smoothing_window)

        # Baseline (set during calibration)
        self._ear_0: Optional[float] = None
        self._ear_threshold: float = 0.21   # fallback if not calibrated

        # Blink tracking
        self._closed_count: int = 0
        self._blink_count: int = 0
        self._close_start: Optional[float] = None    # when current closure began

        # Blink duration history (for variance calculation)
        self._blink_durations: deque[float] = deque(maxlen=50)
        self._blink_timestamps: deque[float] = deque()

        # PERCLOS history:  stores (timestamp, is_closed) tuples
        self._perclos_history: deque[tuple[float, bool]] = deque()

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def set_baseline(self, ear_0: float) -> None:
        """
        Set the personalised EAR baseline from calibration.

        The closure threshold becomes:
            EAR_threshold = EAR_0 × closure_ratio
        """
        self._ear_0 = ear_0
        self._ear_threshold = ear_0 * self._closure_ratio

    # ──────────────────────────────────────────────────────────────────
    # Per-frame update
    # ──────────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> EyeFeatures:
        """
        Extract all eye features from the current frame's landmarks.

        Parameters
        ----------
        landmarks : ndarray (N, 2)  — pixel-space landmarks

        Returns
        -------
        EyeFeatures
        """
        now = time.monotonic()

        left_pts  = landmarks[self._left_idx]
        right_pts = landmarks[self._right_idx]

        ear_left  = compute_ear(left_pts)
        ear_right = compute_ear(right_pts)
        ear_avg   = (ear_left + ear_right) / 2.0
        ear_smoothed = self._smoother.update(ear_avg)

        # ── Relative EAR ──
        ear_relative = ear_smoothed / self._ear_0 if self._ear_0 else 1.0

        # ── Closure detection (relative to personal baseline) ──
        eyes_closed = ear_smoothed < self._ear_threshold

        # ── Blink detection + duration tracking ──
        blink_dur_ms = 0.0
        if eyes_closed:
            self._closed_count += 1
            if self._close_start is None:
                self._close_start = now
        else:
            if self._closed_count >= self._blink_frames:
                # Blink completed
                self._blink_count += 1
                self._blink_timestamps.append(now)

                if self._close_start is not None:
                    dur = (now - self._close_start) * 1000.0  # ms
                    self._blink_durations.append(dur)
                    blink_dur_ms = dur

            self._closed_count = 0
            self._close_start = None

        # ── PERCLOS ──
        # Percentage of Eye CLOSure over sliding window
        # PERCLOS = Σ(closed frames) / Σ(total frames)  in window
        self._perclos_history.append((now, eyes_closed))
        cutoff = now - self._perclos_window
        while self._perclos_history and self._perclos_history[0][0] < cutoff:
            self._perclos_history.popleft()

        total = len(self._perclos_history)
        closed = sum(1 for _, c in self._perclos_history if c)
        perclos = closed / total if total > 0 else 0.0

        # ── Blink rate (blinks per minute) ──
        blink_cutoff = now - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < blink_cutoff:
            self._blink_timestamps.popleft()
        bpm = len(self._blink_timestamps)  # count in last 60s

        # ── Blink variance (std of recent durations) ──
        blink_var = float(np.std(list(self._blink_durations))) if len(self._blink_durations) >= 2 else 0.0

        return EyeFeatures(
            timestamp=now,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            ear_smoothed=ear_smoothed,
            ear_relative=ear_relative,
            perclos=perclos,
            blink_count=self._blink_count,
            blinks_per_minute=float(bpm),
            blink_duration_ms=blink_dur_ms,
            blink_variance=blink_var,
            eyes_closed=eyes_closed,
            closed_frames=self._closed_count,
        )

    def compute_raw_ear(self, landmarks: np.ndarray) -> float:
        """Compute raw average EAR without updating state (for calibration)."""
        left_pts  = landmarks[self._left_idx]
        right_pts = landmarks[self._right_idx]
        return (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0

    def reset(self) -> None:
        self._closed_count = 0
        self._blink_count = 0
        self._close_start = None
        self._blink_timestamps.clear()
        self._blink_durations.clear()
        self._perclos_history.clear()
        self._smoother.reset()
