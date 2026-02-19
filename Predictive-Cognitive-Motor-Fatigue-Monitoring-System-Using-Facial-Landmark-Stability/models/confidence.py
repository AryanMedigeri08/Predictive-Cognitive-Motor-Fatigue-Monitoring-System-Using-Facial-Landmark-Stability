"""
models/confidence.py
====================
Detection Confidence Score.

Confidence is a composite metric that quantifies how reliable the
current frame's landmark detection is, considering:

  1. **Landmark confidence** — mean visibility / presence from MediaPipe
  2. **Lighting score**      — derived from mean luminance
  3. **Blur score**          — Laplacian variance (sharpness)

Confidence = w_lm · LM_conf  +  w_light · Light_score  +  w_blur · Blur_score

When confidence drops below a threshold:
  - Display "Low Detection Confidence" on-screen
  - Reduce fatigue weight influence (avoid false alarms)
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.math_utils import clamp


@dataclass
class ConfidenceResult:
    """Per-frame confidence assessment."""
    landmark_confidence: float = 1.0
    lighting_score: float = 1.0
    blur_score: float = 1.0           # normalised [0, 1]
    composite: float = 1.0            # weighted combination
    is_low: bool = False              # True if below threshold


class ConfidenceScorer:
    """
    Computes composite detection confidence.

    Parameters
    ----------
    w_landmark : float
    w_lighting : float
    w_blur : float
    low_threshold : float
        Composite confidence below this → "low confidence" mode.
    blur_laplacian_threshold : float
        Laplacian variance below this → frame is blurry (blur_score → 0).
    """

    def __init__(
        self,
        w_landmark: float = 0.50,
        w_lighting: float = 0.30,
        w_blur: float = 0.20,
        low_threshold: float = 0.5,
        blur_laplacian_threshold: float = 100.0,
    ) -> None:
        total = w_landmark + w_lighting + w_blur
        self._w_lm = w_landmark / total
        self._w_light = w_lighting / total
        self._w_blur = w_blur / total
        self._low_thresh = low_threshold
        self._blur_thresh = blur_laplacian_threshold

    def compute(
        self,
        landmark_confidence: float,
        lighting_score: float,
        blur_laplacian_var: float,
    ) -> ConfidenceResult:
        """
        Compute the composite confidence score.

        Parameters
        ----------
        landmark_confidence : float [0, 1]
            Mean landmark visibility from MediaPipe.
        lighting_score : float [0, 1]
            From preprocessing — sigmoid of luminance.
        blur_laplacian_var : float
            Raw Laplacian variance (higher → sharper).
        """
        # Normalise blur:  map [0, 2×threshold] → [0, 1]
        # blur_norm = clamp(laplacian_var / (2 × threshold), 0, 1)
        blur_norm = clamp(
            blur_laplacian_var / (2.0 * self._blur_thresh), 0.0, 1.0
        )

        lm = clamp(landmark_confidence, 0.0, 1.0)
        lt = clamp(lighting_score, 0.0, 1.0)

        # Composite = w_lm · LM + w_light · Light + w_blur · Blur
        composite = self._w_lm * lm + self._w_light * lt + self._w_blur * blur_norm
        composite = clamp(composite, 0.0, 1.0)

        return ConfidenceResult(
            landmark_confidence=lm,
            lighting_score=lt,
            blur_score=blur_norm,
            composite=composite,
            is_low=(composite < self._low_thresh),
        )
