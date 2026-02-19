"""
model/fatigue_model.py
======================
Dual-Axis Fatigue Model.

Axis 1 — Physiological Fatigue:
    Derived from EAR (eye closure), blink frequency, and MAR (yawning).

Axis 2 — Cognitive Fatigue:
    Derived from head pose deviation and head stability variance.

Each axis produces a normalised [0, 1] score.  The scores are passed to
the FatigueScoreFusion module for weighted combination into the FSI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.eye_metrics import EyeState
from vision.mouth_metrics import MouthState
from vision.head_pose import HeadPoseState
from vision.stability import StabilityState
from utils.math_utils import clamp


@dataclass
class FatigueAxes:
    """Intermediate dual-axis fatigue scores before fusion."""
    # Axis 1 — Physiological
    ear_score: float = 0.0          # [0, 1] — higher = more fatigued
    blink_score: float = 0.0        # [0, 1] — abnormal blink rate
    mar_score: float = 0.0          # [0, 1] — yawning intensity

    # Axis 2 — Cognitive
    head_deviation_score: float = 0.0   # [0, 1] — head pose deviation
    stability_score: float = 0.0        # [0, 1] — head movement variance

    # Composite axis scores
    physiological_score: float = 0.0
    cognitive_score: float = 0.0


class DualAxisFatigueModel:
    """
    Computes physiological and cognitive fatigue scores from sensor states.

    Parameters
    ----------
    ear_threshold : float
        EAR value considered fully closed (score = 1.0).
    ear_open_value : float
        EAR value considered fully open (score = 0.0).
    mar_threshold : float
        MAR value considered a full yawn (score = 1.0).
    blink_rate_normal_min : float
        Lower bound of normal blink rate (blinks/min).
    blink_rate_normal_max : float
        Upper bound of normal blink rate (blinks/min).
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        ear_open_value: float = 0.35,
        mar_threshold: float = 0.6,
        blink_rate_normal_min: float = 10.0,
        blink_rate_normal_max: float = 20.0,
    ) -> None:
        self._ear_thresh = ear_threshold
        self._ear_open   = ear_open_value
        self._mar_thresh = mar_threshold
        self._blink_min  = blink_rate_normal_min
        self._blink_max  = blink_rate_normal_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        eye_state: EyeState,
        mouth_state: MouthState,
        head_state: HeadPoseState,
        stability_state: StabilityState,
    ) -> FatigueAxes:
        """
        Compute dual-axis fatigue scores from the current sensor states.

        Parameters
        ----------
        eye_state : EyeState
        mouth_state : MouthState
        head_state : HeadPoseState
        stability_state : StabilityState

        Returns
        -------
        FatigueAxes
            Normalised scores for each fatigue dimension.
        """
        # --- Axis 1: Physiological ---

        # EAR score: 0 when eyes fully open, 1 when fully closed
        ear_range = max(self._ear_open - self._ear_thresh, 1e-6)
        ear_score = clamp(
            (self._ear_open - eye_state.ear_smoothed) / ear_range, 0.0, 1.0
        )

        # Blink score: 0 when blink rate is in normal range, 1 at extremes
        bpm = eye_state.blinks_per_minute
        if self._blink_min <= bpm <= self._blink_max:
            blink_score = 0.0
        elif bpm < self._blink_min:
            # Too few blinks → drowsiness
            blink_score = clamp(
                (self._blink_min - bpm) / max(self._blink_min, 1.0), 0.0, 1.0
            )
        else:
            # Too many blinks → possible irritation / early fatigue
            blink_score = clamp(
                (bpm - self._blink_max) / max(self._blink_max, 1.0), 0.0, 1.0
            )

        # MAR score: 0 when mouth closed, 1 when fully yawning
        mar_score = clamp(mouth_state.mar_smoothed / max(self._mar_thresh, 1e-6), 0.0, 1.0)

        # Physiological composite (equal weighting within axis)
        physiological = (ear_score + blink_score + mar_score) / 3.0

        # --- Axis 2: Cognitive ---
        head_deviation_score = clamp(head_state.deviation_score, 0.0, 1.0)
        stability_score      = clamp(stability_state.stability_score, 0.0, 1.0)

        cognitive = (head_deviation_score + stability_score) / 2.0

        return FatigueAxes(
            ear_score=ear_score,
            blink_score=blink_score,
            mar_score=mar_score,
            head_deviation_score=head_deviation_score,
            stability_score=stability_score,
            physiological_score=physiological,
            cognitive_score=cognitive,
        )
