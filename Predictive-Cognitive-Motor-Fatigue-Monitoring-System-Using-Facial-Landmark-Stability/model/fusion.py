"""
model/fusion.py
===============
Fatigue Score Fusion module.

Combines individual fatigue dimension scores into a single
Fatigue Severity Index (FSI) using configurable weights:

    FSI = w1*EAR_score + w2*MAR_score + w3*HeadDeviation + w4*StabilityScore

Classifies FSI into three severity levels:
    Normal   : FSI < T1
    Moderate : T1 ≤ FSI < T2
    Severe   : FSI ≥ T2
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from model.fatigue_model import FatigueAxes
from utils.smoothing import SlidingWindowSmoother


class FatigueLevel(Enum):
    """Fatigue severity classification."""
    NORMAL   = auto()
    MODERATE = auto()
    SEVERE   = auto()

    def label(self) -> str:
        """Human-readable label."""
        return self.name.capitalize()

    def color_bgr(self) -> tuple[int, int, int]:
        """BGR colour for on-screen display."""
        return {
            FatigueLevel.NORMAL:   (0, 220, 0),
            FatigueLevel.MODERATE: (0, 200, 255),
            FatigueLevel.SEVERE:   (0, 0, 255),
        }[self]


@dataclass
class FusionResult:
    """Output of the fusion step."""
    fsi: float = 0.0                          # Raw FSI [0, 1]
    fsi_smoothed: float = 0.0                 # Smoothed FSI [0, 1]
    level: FatigueLevel = FatigueLevel.NORMAL
    ear_contribution: float = 0.0
    mar_contribution: float = 0.0
    head_contribution: float = 0.0
    stability_contribution: float = 0.0


class FatigueScoreFusion:
    """
    Weighted fusion of fatigue dimension scores into the FSI.

    Parameters
    ----------
    w_ear : float
        Weight for EAR score.
    w_mar : float
        Weight for MAR score.
    w_head : float
        Weight for head deviation score.
    w_stability : float
        Weight for stability score.
    threshold_moderate : float
        T1: FSI ≥ this → Moderate.
    threshold_severe : float
        T2: FSI ≥ this → Severe.
    smoothing_window : int
        Moving-average window for FSI smoothing.

    Notes
    -----
    Weights are automatically normalised so they sum to 1.0.
    """

    def __init__(
        self,
        w_ear: float = 0.35,
        w_mar: float = 0.20,
        w_head: float = 0.25,
        w_stability: float = 0.20,
        threshold_moderate: float = 0.35,
        threshold_severe: float = 0.60,
        smoothing_window: int = 15,
    ) -> None:
        # Normalise weights
        total = w_ear + w_mar + w_head + w_stability
        self._w_ear       = w_ear       / total
        self._w_mar       = w_mar       / total
        self._w_head      = w_head      / total
        self._w_stability = w_stability / total

        self._t1 = threshold_moderate
        self._t2 = threshold_severe

        self._smoother = SlidingWindowSmoother(smoothing_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, axes: FatigueAxes) -> FusionResult:
        """
        Compute the FSI and classify fatigue severity.

        Parameters
        ----------
        axes : FatigueAxes
            Output from :class:`DualAxisFatigueModel`.

        Returns
        -------
        FusionResult
            FSI, smoothed FSI, and fatigue level.
        """
        # Weighted contributions
        c_ear       = self._w_ear       * axes.ear_score
        c_mar       = self._w_mar       * axes.mar_score
        c_head      = self._w_head      * axes.head_deviation_score
        c_stability = self._w_stability * axes.stability_score

        fsi = c_ear + c_mar + c_head + c_stability
        fsi_smoothed = self._smoother.update(fsi)

        level = self._classify(fsi_smoothed)

        return FusionResult(
            fsi=fsi,
            fsi_smoothed=fsi_smoothed,
            level=level,
            ear_contribution=c_ear,
            mar_contribution=c_mar,
            head_contribution=c_head,
            stability_contribution=c_stability,
        )

    def reset(self) -> None:
        """Reset the FSI smoothing buffer."""
        self._smoother.reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, fsi: float) -> FatigueLevel:
        """
        Map FSI to a FatigueLevel.

        Parameters
        ----------
        fsi : float
            Smoothed FSI in [0, 1].

        Returns
        -------
        FatigueLevel
        """
        if fsi >= self._t2:
            return FatigueLevel.SEVERE
        if fsi >= self._t1:
            return FatigueLevel.MODERATE
        return FatigueLevel.NORMAL
