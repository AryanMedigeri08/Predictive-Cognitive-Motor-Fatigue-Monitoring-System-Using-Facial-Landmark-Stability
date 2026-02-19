"""
calibration/calibration_engine.py
=================================
90-second personalised calibration engine.

During calibration the driver maintains:
  - Neutral posture
  - Forward gaze
  - Natural blink pattern

The engine accumulates per-frame samples and computes baseline values:
  EAR_0   — baseline Eye Aspect Ratio
  MAR_0   — baseline Mouth Aspect Ratio
  BF_0    — baseline Blink Frequency (blinks / min)
  H_0     — baseline Head Pose (pitch, yaw, roll)
  G_0     — baseline Gaze Vector
  Neck_0  — baseline Neck Vector

Baselines are stored in a JSON profile so that re-calibration is only
needed after significant changes (different driver, new glasses, etc.).

Outlier rejection:
    Samples are trimmed to the 10th–90th percentile before averaging to
    reject transient artefacts (coughs, involuntary movements).

Morphology-adaptive:
    Because thresholds are derived from the driver's own baseline, the
    system naturally adapts to small eye structures (e.g. Northeast
    Indian phenotypes).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CalibrationProfile:
    """Stored calibration baselines."""
    ear_0: float = 0.0
    mar_0: float = 0.0
    bf_0: float = 0.0           # blinks per minute
    pitch_0: float = 0.0
    yaw_0: float = 0.0
    roll_0: float = 0.0
    gaze_0: list[float] = field(default_factory=lambda: [0.0, 0.0])
    neck_0: list[float] = field(default_factory=lambda: [0.0, 0.0])
    timestamp: str = ""
    duration_sec: float = 0.0
    sample_count: int = 0


class CalibrationEngine:
    """
    Orchestrates the 90-second calibration phase.

    Usage
    -----
    1. Create instance
    2. Call ``add_sample()`` every frame during calibration
    3. When ``is_complete`` → call ``finalise()``
    4. Access ``profile`` for baselines
    5. Optionally ``save_profile()`` to JSON

    Parameters
    ----------
    duration_sec : float
        Calibration session length.
    json_path : str or Path
        Where to save/load the calibration profile.
    """

    def __init__(
        self,
        duration_sec: float = 90.0,
        json_path: str | Path = "calibration_profile.json",
    ) -> None:
        self._duration = duration_sec
        self._json_path = Path(json_path)
        self._start_time: Optional[float] = None

        # Sample accumulators
        self._ear_samples: list[float] = []
        self._mar_samples: list[float] = []
        self._pitch_samples: list[float] = []
        self._yaw_samples: list[float] = []
        self._roll_samples: list[float] = []
        self._gaze_samples: list[np.ndarray] = []
        self._neck_samples: list[np.ndarray] = []
        self._blink_count: int = 0

        self._profile: Optional[CalibrationProfile] = None
        self._complete = False

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the calibration timer."""
        self._start_time = time.monotonic()

    @property
    def is_complete(self) -> bool:
        if self._start_time is None:
            return False
        return (time.monotonic() - self._start_time) >= self._duration

    @property
    def remaining_sec(self) -> float:
        if self._start_time is None:
            return self._duration
        return max(0.0, self._duration - (time.monotonic() - self._start_time))

    @property
    def progress(self) -> float:
        """[0, 1] calibration progress."""
        return 1.0 - (self.remaining_sec / self._duration)

    @property
    def profile(self) -> Optional[CalibrationProfile]:
        return self._profile

    def add_sample(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        gaze_vec: np.ndarray,
        neck_vec: np.ndarray,
    ) -> None:
        """Add a per-frame sample during the calibration window."""
        self._ear_samples.append(ear)
        self._mar_samples.append(mar)
        self._pitch_samples.append(pitch)
        self._yaw_samples.append(yaw)
        self._roll_samples.append(roll)
        self._gaze_samples.append(gaze_vec.copy())
        self._neck_samples.append(neck_vec.copy())

    def register_blink(self) -> None:
        """Called by eye-feature extractor when a blink is detected during calibration."""
        self._blink_count += 1

    def finalise(self) -> CalibrationProfile:
        """
        Compute baselines from collected samples.

        Outlier rejection: trim to [10th, 90th] percentile.

        Returns
        -------
        CalibrationProfile
        """
        if self._start_time is None:
            raise RuntimeError("Calibration was never started.")

        elapsed = time.monotonic() - self._start_time

        ear_0 = self._robust_mean(self._ear_samples)
        mar_0 = self._robust_mean(self._mar_samples)
        pitch_0 = self._robust_mean(self._pitch_samples)
        yaw_0 = self._robust_mean(self._yaw_samples)
        roll_0 = self._robust_mean(self._roll_samples)

        # Blink frequency → blinks per minute
        bf_0 = (self._blink_count / max(elapsed, 1.0)) * 60.0

        # Average gaze vector
        if self._gaze_samples:
            gaze_0 = np.mean(self._gaze_samples, axis=0)
            norm = np.linalg.norm(gaze_0)
            if norm > 1e-10:
                gaze_0 = gaze_0 / norm
        else:
            gaze_0 = np.array([0.0, 0.0])

        # Average neck vector
        if self._neck_samples:
            neck_0 = np.mean(self._neck_samples, axis=0)
            norm = np.linalg.norm(neck_0)
            if norm > 1e-10:
                neck_0 = neck_0 / norm
        else:
            neck_0 = np.array([0.0, -1.0])

        self._profile = CalibrationProfile(
            ear_0=ear_0,
            mar_0=mar_0,
            bf_0=bf_0,
            pitch_0=pitch_0,
            yaw_0=yaw_0,
            roll_0=roll_0,
            gaze_0=gaze_0.tolist(),
            neck_0=neck_0.tolist(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_sec=elapsed,
            sample_count=len(self._ear_samples),
        )
        self._complete = True
        return self._profile

    # ──────────────────────────────────────────────────────────────────
    # JSON persistence
    # ──────────────────────────────────────────────────────────────────

    def save_profile(self, path: Optional[str | Path] = None) -> Path:
        """Save calibration profile to JSON."""
        if self._profile is None:
            raise RuntimeError("No profile to save — call finalise() first.")
        out = Path(path) if path else self._json_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(self._profile), f, indent=2)
        return out

    def load_profile(self, path: Optional[str | Path] = None) -> CalibrationProfile:
        """Load a previously saved calibration profile."""
        src = Path(path) if path else self._json_path
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._profile = CalibrationProfile(**data)
        self._complete = True
        return self._profile

    def has_saved_profile(self, path: Optional[str | Path] = None) -> bool:
        """Check if a saved profile exists on disk."""
        src = Path(path) if path else self._json_path
        return src.exists()

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _robust_mean(samples: list[float]) -> float:
        """
        Mean after trimming to [10th, 90th] percentile.

        Removes outliers caused by momentary artefacts during calibration
        (coughs, involuntary movements, tracking glitches).
        """
        if not samples:
            return 0.0
        arr = np.array(samples)
        lo, hi = np.percentile(arr, [10, 90])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(arr))
