"""
models/fatigue_model.py
=======================
Dual-Axis Fatigue Model with configurable weights and EMA smoothing.

Equations
---------
Physical Fatigue (PF):
    PF = w1·PERCLOS + w2·BlinkDuration + w3·HeadDeviation + w4·MouthIndex

Cognitive Fatigue (CF):
    CF = v1·GazeDrift + v2·BlinkVariance + v3·MicroSaccadeVariance

Fatigue Index (FI):
    FI = α·PF + β·CF

EMA Smoothing:
    FI_smooth = λ·FI + (1−λ)·FI_prev

    where λ = EMA_LAMBDA ∈ (0, 1).  Lower λ → smoother but more lagging.

Classification:
    FI_smooth < T1          → Normal
    T1 ≤ FI_smooth < T2     → Moderate
    FI_smooth ≥ T2          → Severe
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from utils.math_utils import clamp
from utils.smoothing import ExponentialMovingAverage


# ──────────────────────────────────────────────────────────────────────
# Fatigue Level Classification
# ──────────────────────────────────────────────────────────────────────

class FatigueLevel(Enum):
    NORMAL   = auto()
    MODERATE = auto()
    SEVERE   = auto()

    def label(self) -> str:
        return self.name.capitalize()

    def color_bgr(self) -> tuple[int, int, int]:
        return {
            FatigueLevel.NORMAL:   (0, 220, 0),
            FatigueLevel.MODERATE: (0, 200, 255),
            FatigueLevel.SEVERE:   (0, 0, 255),
        }[self]


# ──────────────────────────────────────────────────────────────────────
# Feature Input + Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FatigueInput:
    """All normalised [0, 1] feature scores fed into the model."""
    perclos: float = 0.0
    blink_duration_norm: float = 0.0    # normalised blink duration
    head_deviation: float = 0.0
    mouth_opening_index: float = 0.0
    gaze_drift: float = 0.0
    blink_variance_norm: float = 0.0
    microsaccade_variance: float = 0.0


@dataclass
class FatigueResult:
    """Output of the dual-axis fatigue model."""
    physical_fatigue: float = 0.0       # PF [0, 1]
    cognitive_fatigue: float = 0.0      # CF [0, 1]
    fatigue_index: float = 0.0          # FI raw [0, 1]
    fatigue_index_smooth: float = 0.0   # FI smoothed [0, 1]
    level: FatigueLevel = FatigueLevel.NORMAL

    # Contribution breakdown (for logging / debugging)
    pf_perclos: float = 0.0
    pf_blink_dur: float = 0.0
    pf_head_dev: float = 0.0
    pf_mouth: float = 0.0
    cf_gaze: float = 0.0
    cf_blink_var: float = 0.0
    cf_saccade: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class DualAxisFatigueModel:
    """
    Computes the Fatigue Index (FI) from normalised feature scores.

    Parameters
    ----------
    w1..w4 : float — Physical Fatigue weights
    v1..v3 : float — Cognitive Fatigue weights
    alpha, beta : float — PF/CF blending weights
    ema_lambda : float — EMA smoothing factor
    threshold_moderate, threshold_severe : float — classification thresholds
    """

    def __init__(
        self,
        w1: float = 0.35, w2: float = 0.20,
        w3: float = 0.25, w4: float = 0.20,
        v1: float = 0.40, v2: float = 0.30, v3: float = 0.30,
        alpha: float = 0.6, beta: float = 0.4,
        ema_lambda: float = 0.15,
        threshold_moderate: float = 0.35,
        threshold_severe: float = 0.60,
    ) -> None:
        # Normalise PF weights so they sum to 1
        pf_total = w1 + w2 + w3 + w4
        self._w1 = w1 / pf_total
        self._w2 = w2 / pf_total
        self._w3 = w3 / pf_total
        self._w4 = w4 / pf_total

        # Normalise CF weights
        cf_total = v1 + v2 + v3
        self._v1 = v1 / cf_total
        self._v2 = v2 / cf_total
        self._v3 = v3 / cf_total

        # Normalise alpha/beta
        ab_total = alpha + beta
        self._alpha = alpha / ab_total
        self._beta = beta / ab_total

        self._t1 = threshold_moderate
        self._t2 = threshold_severe

        self._ema = ExponentialMovingAverage(lam=ema_lambda)

    def compute(self, inp: FatigueInput) -> FatigueResult:
        """
        Compute FI from the input features.

        All inputs must be pre-normalised to [0, 1].
        """
        # ── Physical Fatigue ──
        # PF = w1·PERCLOS + w2·BlinkDuration + w3·HeadDev + w4·MOI
        pf_p = self._w1 * clamp(inp.perclos, 0, 1)
        pf_b = self._w2 * clamp(inp.blink_duration_norm, 0, 1)
        pf_h = self._w3 * clamp(inp.head_deviation, 0, 1)
        pf_m = self._w4 * clamp(inp.mouth_opening_index, 0, 1)
        pf = pf_p + pf_b + pf_h + pf_m

        # ── Cognitive Fatigue ──
        # CF = v1·GazeDrift + v2·BlinkVar + v3·MicroSaccade
        cf_g = self._v1 * clamp(inp.gaze_drift, 0, 1)
        cf_bv = self._v2 * clamp(inp.blink_variance_norm, 0, 1)
        cf_s = self._v3 * clamp(inp.microsaccade_variance, 0, 1)
        cf = cf_g + cf_bv + cf_s

        # ── Fatigue Index ──
        # FI = α·PF + β·CF
        fi = self._alpha * pf + self._beta * cf
        fi = clamp(fi, 0.0, 1.0)

        # ── EMA Smoothing ──
        # FI_smooth = λ·FI + (1−λ)·FI_prev
        fi_smooth = self._ema.update(fi)

        # ── Classification ──
        level = self._classify(fi_smooth)

        return FatigueResult(
            physical_fatigue=pf,
            cognitive_fatigue=cf,
            fatigue_index=fi,
            fatigue_index_smooth=fi_smooth,
            level=level,
            pf_perclos=pf_p, pf_blink_dur=pf_b,
            pf_head_dev=pf_h, pf_mouth=pf_m,
            cf_gaze=cf_g, cf_blink_var=cf_bv, cf_saccade=cf_s,
        )

    def override_fi_smooth(self, value: float) -> None:
        """
        Used by the recovery module to nudge FI_smooth downward
        without bypassing EMA state tracking.
        """
        self._ema._value = value

    def _classify(self, fi: float) -> FatigueLevel:
        if fi >= self._t2:
            return FatigueLevel.SEVERE
        if fi >= self._t1:
            return FatigueLevel.MODERATE
        return FatigueLevel.NORMAL

    def reset(self) -> None:
        self._ema.reset()
