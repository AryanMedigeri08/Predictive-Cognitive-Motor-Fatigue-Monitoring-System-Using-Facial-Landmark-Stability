"""
recovery/recovery_tracker.py
=============================
Fatigue Recovery Dynamics.

When the driver returns to an alert posture (eyes open, head upright,
steady gaze), the fatigue index should **gradually decay** rather than
drop abruptly — modelling the physiological reality that recovery
takes time.

Decay model (exponential):
    FI_t = FI_prev · exp(−k · Δt)

where:
    k   = RECOVERY_DECAY_K   (decay rate constant, 1/s)
    Δt  = time since last frame  (seconds)

Recovery eligibility (all must be true):
    1. EAR_relative > RECOVERY_EAR_RELATIVE_THRESH   (eyes open)
    2. Head deviation < RECOVERY_HEAD_DEV_THRESH      (head upright)
    3. |BPM − BF_0| / BF_0 < RECOVERY_BLINK_TOLERANCE  (normal blink)

If any condition fails, decay does NOT apply and FI follows the raw
model output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from utils.math_utils import clamp


@dataclass
class RecoveryState:
    """Per-frame recovery assessment."""
    is_recovering: bool = False
    fi_decayed: float = 0.0
    seconds_in_recovery: float = 0.0


class RecoveryTracker:
    """
    Tracks recovery eligibility and applies exponential decay to FI.

    Parameters
    ----------
    decay_k : float
        Exponential decay constant (1/s).
    ear_rel_thresh : float
        EAR_relative above this → eyes open.
    head_dev_thresh : float
        Head deviation below this → upright.
    blink_tolerance : float
        Normalised blink-rate deviation tolerance.
    """

    def __init__(
        self,
        decay_k: float = 0.05,
        ear_rel_thresh: float = 0.95,
        head_dev_thresh: float = 0.15,
        blink_tolerance: float = 0.3,
    ) -> None:
        self._k = decay_k
        self._ear_thresh = ear_rel_thresh
        self._head_thresh = head_dev_thresh
        self._blink_tol = blink_tolerance

        self._recovery_start: float | None = None
        self._last_time: float | None = None

    def update(
        self,
        fi_smooth: float,
        ear_relative: float,
        head_deviation: float,
        bpm: float,
        bf_0: float,
    ) -> RecoveryState:
        """
        Evaluate recovery eligibility and apply decay if warranted.

        Parameters
        ----------
        fi_smooth : float
            Current smoothed Fatigue Index.
        ear_relative : float
            EAR_t / EAR_0.
        head_deviation : float
            [0, 1] head deviation score.
        bpm : float
            Current blinks per minute.
        bf_0 : float
            Baseline blinks per minute from calibration.

        Returns
        -------
        RecoveryState
        """
        now = time.monotonic()

        # ── Check recovery eligibility ──
        eyes_ok = ear_relative > self._ear_thresh
        head_ok = head_deviation < self._head_thresh

        if bf_0 > 0:
            blink_ok = abs(bpm - bf_0) / bf_0 < self._blink_tol
        else:
            blink_ok = True

        eligible = eyes_ok and head_ok and blink_ok

        if not eligible:
            self._recovery_start = None
            self._last_time = now
            return RecoveryState(
                is_recovering=False,
                fi_decayed=fi_smooth,
                seconds_in_recovery=0.0,
            )

        # ── Apply decay ──
        if self._recovery_start is None:
            self._recovery_start = now

        dt = (now - self._last_time) if self._last_time else 0.0
        self._last_time = now

        # FI_t = FI_prev · exp(−k · Δt)
        fi_decayed = fi_smooth * math.exp(-self._k * dt)
        fi_decayed = clamp(fi_decayed, 0.0, 1.0)

        seconds_recovering = now - self._recovery_start

        return RecoveryState(
            is_recovering=True,
            fi_decayed=fi_decayed,
            seconds_in_recovery=seconds_recovering,
        )

    def reset(self) -> None:
        self._recovery_start = None
        self._last_time = None
