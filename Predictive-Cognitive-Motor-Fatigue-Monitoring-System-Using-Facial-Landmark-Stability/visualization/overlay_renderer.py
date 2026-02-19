"""
visualization/overlay_renderer.py
==================================
Real-time overlay renderer for the Driver Fatigue Monitoring System.

Renders:
  - Eye contours (left / right)
  - Iris centres
  - Mouth contour
  - Nose tip + chin markers
  - Neck vector line
  - Head pose axes (from HeadFeatureExtractor.draw_axes)
  - Semi-transparent metric dashboard panel
  - Calibration progress screen
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from features.eye_features import EyeFeatures
from features.mouth_features import MouthFeatures
from features.head_features import HeadFeatures
from features.gaze_features import GazeFeatures
from features.neck_features import NeckFeatures
from models.fatigue_model import FatigueResult, FatigueLevel
from models.confidence import ConfidenceResult
from recovery.recovery_tracker import RecoveryState
import config as cfg


class OverlayRenderer:
    """Draws all visual overlays onto the BGR camera frame."""

    # ──────────────────────────────────────────────────────────────────
    # Landmark overlays
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_eye_contours(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = cfg.COLOR_CYAN,
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw left and right eye contours."""
        for contour_idx in [cfg.LEFT_EYE_CONTOUR, cfg.RIGHT_EYE_CONTOUR]:
            pts = landmarks[contour_idx].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True,
                         color=color, thickness=thickness)
        return frame

    @staticmethod
    def draw_iris_centres(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = cfg.COLOR_MAGENTA,
        radius: int = 3,
    ) -> np.ndarray:
        """Draw iris centre dots."""
        for idx in [cfg.LEFT_IRIS_INDICES[0], cfg.RIGHT_IRIS_INDICES[0]]:
            if idx < len(landmarks):
                pt = tuple(landmarks[idx].astype(int))
                cv2.circle(frame, pt, radius, color, -1)
        return frame

    @staticmethod
    def draw_mouth_contour(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = cfg.COLOR_CYAN,
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw mouth contour."""
        pts = landmarks[cfg.MOUTH_CONTOUR].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True,
                     color=color, thickness=thickness)
        return frame

    @staticmethod
    def draw_nose_chin(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = cfg.COLOR_GREEN,
    ) -> np.ndarray:
        """Draw nose tip and chin markers."""
        nose = tuple(landmarks[cfg.NOSE_TIP_INDEX].astype(int))
        chin = tuple(landmarks[cfg.NECK_CHIN_INDEX].astype(int))
        cv2.circle(frame, nose, 4, color, -1)
        cv2.circle(frame, chin, 4, cfg.COLOR_YELLOW, -1)
        return frame

    @staticmethod
    def draw_neck_vector(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = cfg.COLOR_YELLOW,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw neck vector (chin → forehead)."""
        chin = tuple(landmarks[cfg.NECK_CHIN_INDEX].astype(int))
        forehead = tuple(landmarks[cfg.NECK_FOREHEAD_INDEX].astype(int))
        cv2.arrowedLine(frame, chin, forehead, color, thickness, tipLength=0.2)
        return frame

    # ──────────────────────────────────────────────────────────────────
    # Dashboard
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_metrics_panel(
        frame: np.ndarray,
        eye: EyeFeatures,
        mouth: MouthFeatures,
        head: HeadFeatures,
        gaze: GazeFeatures,
        neck: NeckFeatures,
        fatigue: FatigueResult,
        confidence: ConfidenceResult,
        recovery: RecoveryState,
        fps: float,
    ) -> np.ndarray:
        """
        Semi-transparent dashboard panel in the top-right corner.
        """
        h, w = frame.shape[:2]
        panel_w, panel_h = 340, 360
        x1, y1 = w - panel_w - 10, 10
        x2, y2 = x1 + panel_w, y1 + panel_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
        frame = cv2.addWeighted(overlay, cfg.OVERLAY_ALPHA, frame,
                                1.0 - cfg.OVERLAY_ALPHA, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        color = cfg.COLOR_WHITE
        lh = 22  # line height
        x, y = x1 + 10, y1 + 20

        lines = [
            f"FPS: {fps:.1f}",
            f"EAR: {eye.ear_smoothed:.3f} (rel: {eye.ear_relative:.2f})",
            f"PERCLOS: {eye.perclos:.2%}",
            f"Blinks/min: {eye.blinks_per_minute:.0f}",
            f"Blink Dur: {eye.blink_duration_ms:.0f} ms",
            f"Blink Var: {eye.blink_variance:.1f}",
            f"MAR: {mouth.mar_smoothed:.3f} (MOI: {mouth.mouth_opening_index:.2f})",
            f"Yawns: {mouth.yawn_count}",
            f"Head Dev: {head.deviation_score:.2f}",
            f"Pitch/Yaw/Roll: {head.pitch:.1f}/{head.yaw:.1f}/{head.roll:.1f}",
            f"Gaze Drift: {gaze.gaze_drift:.2f}",
            f"Saccade Var: {gaze.microsaccade_variance:.2f}",
            f"Neck Stab: {neck.stability_score:.2f}",
            f"Confidence: {confidence.composite:.2f}",
            f"PF: {fatigue.physical_fatigue:.2f}  CF: {fatigue.cognitive_fatigue:.2f}",
            f"FI: {fatigue.fatigue_index_smooth:.3f} [{fatigue.level.label()}]",
        ]

        if recovery.is_recovering:
            lines.append(f"Recovery: {recovery.seconds_in_recovery:.1f}s")

        for line in lines:
            cv2.putText(frame, line, (x, y), font, scale, color, 1, cv2.LINE_AA)
            y += lh

        # Fatigue bar
        bar_x = x
        bar_y = y + 5
        bar_w = panel_w - 20
        bar_h = 12
        fi_pct = min(fatigue.fatigue_index_smooth, 1.0)

        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_color = fatigue.level.color_bgr()
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + int(bar_w * fi_pct), bar_y + bar_h),
                     fill_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h), cfg.COLOR_WHITE, 1)

        return frame

    # ──────────────────────────────────────────────────────────────────
    # Calibration Screen
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_calibration_screen(
        frame: np.ndarray,
        progress: float,
        remaining_sec: float,
    ) -> np.ndarray:
        """Draw calibration progress overlay."""
        h, w = frame.shape[:2]

        # Dim the background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        title = "CALIBRATION IN PROGRESS"
        (tw, _), _ = cv2.getTextSize(title, font, 1.0, 2)
        cv2.putText(frame, title, ((w - tw) // 2, h // 2 - 60),
                   font, 1.0, cfg.COLOR_CYAN, 2, cv2.LINE_AA)

        # Instructions
        instructions = [
            "Look straight ahead with a neutral expression",
            "Blink naturally",
            "Keep posture relaxed and steady",
        ]
        for i, inst in enumerate(instructions):
            (iw, _), _ = cv2.getTextSize(inst, font, 0.5, 1)
            cv2.putText(frame, inst, ((w - iw) // 2, h // 2 - 10 + i * 25),
                       font, 0.5, cfg.COLOR_WHITE, 1, cv2.LINE_AA)

        # Progress bar
        bar_w = 400
        bar_h = 20
        bx = (w - bar_w) // 2
        by = h // 2 + 60
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h),
                     (80, 80, 80), -1)
        fill = int(bar_w * min(progress, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h),
                     cfg.COLOR_GREEN, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h),
                     cfg.COLOR_WHITE, 1)

        # Countdown
        countdown = f"{remaining_sec:.0f}s remaining"
        (cw, _), _ = cv2.getTextSize(countdown, font, 0.7, 2)
        cv2.putText(frame, countdown, ((w - cw) // 2, by + 55),
                   font, 0.7, cfg.COLOR_YELLOW, 2, cv2.LINE_AA)

        # Percentage
        pct = f"{progress * 100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct, font, 0.5, 1)
        cv2.putText(frame, pct, ((w - pw) // 2, by + bar_h - 3),
                   font, 0.5, cfg.COLOR_WHITE, 1, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────────
    # Low-confidence banner
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_low_confidence_banner(
        frame: np.ndarray,
        confidence: ConfidenceResult,
    ) -> np.ndarray:
        """Show a banner when detection confidence is low."""
        if not confidence.is_low:
            return frame

        h, w = frame.shape[:2]
        msg = f"Low Detection Confidence ({confidence.composite:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(msg, font, 0.6, 2)
        tx = (w - tw) // 2
        ty = h - 30
        cv2.rectangle(frame, (tx - 10, ty - th - 5),
                     (tx + tw + 10, ty + 5), (0, 0, 100), cv2.FILLED)
        cv2.putText(frame, msg, (tx, ty), font, 0.6,
                   cfg.COLOR_ORANGE, 2, cv2.LINE_AA)
        return frame

    # ──────────────────────────────────────────────────────────────────
    # Full render pass
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def render_all(
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        eye: EyeFeatures,
        mouth: MouthFeatures,
        head: HeadFeatures,
        gaze: GazeFeatures,
        neck: NeckFeatures,
        fatigue: FatigueResult,
        confidence: ConfidenceResult,
        recovery: RecoveryState,
        fps: float,
    ) -> np.ndarray:
        """Run all overlays in order."""
        if landmarks is not None and len(landmarks) > 0:
            frame = OverlayRenderer.draw_eye_contours(frame, landmarks)
            frame = OverlayRenderer.draw_iris_centres(frame, landmarks)
            frame = OverlayRenderer.draw_mouth_contour(frame, landmarks)
            frame = OverlayRenderer.draw_nose_chin(frame, landmarks)
            frame = OverlayRenderer.draw_neck_vector(frame, landmarks)

        frame = OverlayRenderer.draw_metrics_panel(
            frame, eye, mouth, head, gaze, neck,
            fatigue, confidence, recovery, fps,
        )
        frame = OverlayRenderer.draw_low_confidence_banner(frame, confidence)
        return frame
