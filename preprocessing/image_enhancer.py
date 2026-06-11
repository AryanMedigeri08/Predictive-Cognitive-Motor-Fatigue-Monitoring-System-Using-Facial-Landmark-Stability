"""
preprocessing/image_enhancer.py
===============================
Real-time image enhancement pipeline for robust face detection under
adverse conditions:

  1. **Night-mode auto-detection** — mean luminance check
  2. **CLAHE**  — Contrast-Limited Adaptive Histogram Equalisation
  3. **Gamma correction** — non-linear brightness boost for darks
  4. **Glare / reflection suppression** — reduces specular highlights
     that interfere with eye-region landmark detection (glasses users)

All transformations operate in-place on uint8 BGR frames for minimal
memory allocation overhead at ≥ 20 FPS.

Mathematical references
-----------------------
CLAHE (Pizer et al., 1987):
    Limits the contrast amplification in each tile by clipping the
    histogram at a configurable clip-limit before redistribution.

Gamma correction:
    I_out = 255 · (I_in / 255)^(1/γ)       γ > 1 → brighten darks

Laplacian variance (blur metric):
    σ² = Var(∇²I)       higher → sharper image
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhancementInfo:
    """Metadata returned alongside the enhanced frame."""
    is_night_mode: bool = False
    mean_luminance: float = 128.0
    blur_score: float = 0.0        # Laplacian variance (higher → sharper)
    lighting_score: float = 1.0    # normalised [0, 1], 1 = well-lit


class ImageEnhancer:
    """
    Stateless image enhancement pipeline.

    Parameters
    ----------
    luminance_threshold : float
        Mean brightness below this → night mode activates.
    clahe_clip_limit : float
        CLAHE clip limit (higher → more contrast).
    clahe_tile_size : tuple[int, int]
        Tile grid size for CLAHE.
    gamma : float
        Gamma correction exponent (> 1 brightens darks).
    glare_suppression : bool
        Whether to apply specular-highlight suppression.
    blur_threshold : float
        Laplacian variance below this → frame considered blurry.
    """

    def __init__(
        self,
        luminance_threshold: float = 60.0,
        clahe_clip_limit: float = 3.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        gamma: float = 1.5,
        glare_suppression: bool = True,
        blur_threshold: float = 100.0,
    ) -> None:
        self._lum_thresh = luminance_threshold
        self._gamma = gamma
        self._glare = glare_suppression
        self._blur_thresh = blur_threshold

        # Pre-build CLAHE object (re-usable across frames)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )

        # Pre-compute gamma look-up table (LUT) for O(1) per-pixel transform
        # I_out = 255 · (I_in / 255)^(1/γ)
        inv_gamma = 1.0 / max(gamma, 0.01)
        self._gamma_lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def enhance(self, frame: np.ndarray) -> tuple[np.ndarray, EnhancementInfo]:
        """
        Run the full enhancement pipeline on *frame* (BGR, uint8).

        Returns
        -------
        enhanced : np.ndarray   — enhanced BGR frame
        info     : EnhancementInfo
        """
        info = EnhancementInfo()

        # ── Compute luminance ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info.mean_luminance = float(gray.mean())

        # ── Compute blur metric (Laplacian variance) ──
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        info.blur_score = float(laplacian.var())

        # ── Lighting score  [0, 1]  (1 = ideal lighting) ──
        # Maps luminance from [0, 255] into a soft sigmoid centred at 100
        # score = 1 / (1 + exp(-0.05 * (L - 100)))
        info.lighting_score = float(
            1.0 / (1.0 + np.exp(-0.05 * (info.mean_luminance - 100.0)))
        )

        # ── Night-mode detection ──
        info.is_night_mode = info.mean_luminance < self._lum_thresh

        enhanced = frame.copy()

        if info.is_night_mode:
            enhanced = self._apply_clahe(enhanced)
            enhanced = self._apply_gamma(enhanced)

        if self._glare:
            enhanced = self._suppress_glare(enhanced)

        return enhanced, info

    # ──────────────────────────────────────────────────────────────────
    # Public helpers (can be called individually for testing)
    # ──────────────────────────────────────────────────────────────────

    def detect_low_light(self, frame: np.ndarray) -> bool:
        """True if mean luminance is below the night-mode threshold."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self._lum_thresh

    def compute_blur_metric(self, frame: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ──────────────────────────────────────────────────────────────────
    # Private transforms
    # ──────────────────────────────────────────────────────────────────

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance channel (LAB colour space) to boost
        contrast without colour distortion.

        Steps:
          1. BGR → LAB
          2. Apply CLAHE on L channel only
          3. LAB → BGR
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction using a pre-computed LUT.

        I_out = 255 · (I_in / 255)^(1/γ)

        LUT lookup is O(1) per pixel — no per-pixel math at runtime.
        """
        return cv2.LUT(frame, self._gamma_lut)

    def _suppress_glare(self, frame: np.ndarray) -> np.ndarray:
        """
        Reduce specular highlights (glasses glare / reflections).

        Approach:
          1. Convert to HSV
          2. Detect high-brightness, low-saturation pixels (specular)
          3. In-paint those regions with surrounding pixel interpolation

        This is lightweight enough for real-time use (glare regions are
        typically < 1% of pixels).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Specular highlights: very high value  AND  very low saturation
        # These are the bright white reflections on glasses lenses
        glare_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 220], dtype=np.uint8),    # low: any hue, low sat, high val
            np.array([180, 50, 255], dtype=np.uint8),  # high
        )

        # Only process if glare pixels detected (avoid unnecessary in-paint)
        if cv2.countNonZero(glare_mask) > 0:
            # Dilate mask slightly to cover glare edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
            # In-paint with Telea method (fast)
            frame = cv2.inpaint(frame, glare_mask, 3, cv2.INPAINT_TELEA)

        return frame
