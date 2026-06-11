"""
alerts/alert_manager.py
=======================
Alert Escalation Engine.

Three-level alert system driven by FI_smooth, confidence, and sustained
duration above threshold:

  Level 1 (Visual)    — yellow border + text after ALERT_LEVEL1_DURATION_SEC
                        of Moderate fatigue
  Level 2 (Audio)     — audible beep after ALERT_LEVEL2_DURATION_SEC
  Level 3 (Critical)  — loud alarm + GPIO trigger after ALERT_LEVEL3_DURATION_SEC
                        of Severe fatigue

Confidence-aware:
  If confidence < threshold, alert escalation is suppressed to avoid
  false positives from unreliable detection.

Cooldown:
  After an alert fires, a cooldown timer prevents repeated alerts at
  the same level.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np

from models.fatigue_model import FatigueLevel


class AlertLevel(IntEnum):
    NONE  = 0
    LEVEL1 = 1   # Visual warning
    LEVEL2 = 2   # Audio beep
    LEVEL3 = 3   # Critical alarm + GPIO


@dataclass
class AlertState:
    """Current alert status."""
    level: AlertLevel = AlertLevel.NONE
    message: str = ""
    seconds_at_level: float = 0.0
    suppressed: bool = False   # True if suppressed by low confidence


class AlertManager:
    """
    Alert escalation manager with cooldown and confidence gating.

    Parameters
    ----------
    level1_dur, level2_dur, level3_dur : float
        Seconds of sustained fatigue before each level triggers.
    cooldown_sec : float
        Minimum seconds between same-level alerts.
    confidence_threshold : float
        Alerts suppressed if confidence < this.
    gpio_enabled : bool
        If True (edge deployment), trigger GPIO pin on Level 3.
    gpio_pin : int
        GPIO pin number for physical alert device.
    """

    def __init__(
        self,
        level1_dur: float = 3.0,
        level2_dur: float = 6.0,
        level3_dur: float = 10.0,
        cooldown_sec: float = 5.0,
        confidence_threshold: float = 0.5,
        gpio_enabled: bool = False,
        gpio_pin: int = 18,
        beep_freq_l2: int = 800,
        beep_freq_l3: int = 1200,
        beep_duration_ms: int = 500,
    ) -> None:
        self._l1_dur = level1_dur
        self._l2_dur = level2_dur
        self._l3_dur = level3_dur
        self._cooldown = cooldown_sec
        self._conf_thresh = confidence_threshold
        self._gpio = gpio_enabled
        self._gpio_pin = gpio_pin
        self._beep_l2 = beep_freq_l2
        self._beep_l3 = beep_freq_l3
        self._beep_ms = beep_duration_ms

        self._fatigue_start: Optional[float] = None
        self._current_level = AlertLevel.NONE
        self._last_alert_time: dict[AlertLevel, float] = {}

    def update(
        self,
        fatigue_level: FatigueLevel,
        confidence: float,
    ) -> AlertState:
        """
        Evaluate alert escalation.

        Parameters
        ----------
        fatigue_level : FatigueLevel
            Current classified fatigue level.
        confidence : float
            Composite detection confidence [0, 1].

        Returns
        -------
        AlertState
        """
        now = time.monotonic()

        # ── Confidence gating ──
        if confidence < self._conf_thresh:
            self._fatigue_start = None
            return AlertState(
                level=AlertLevel.NONE,
                message="Low Confidence — Alerts Suppressed",
                suppressed=True,
            )

        # ── Track sustained fatigue start ──
        if fatigue_level == FatigueLevel.NORMAL:
            self._fatigue_start = None
            self._current_level = AlertLevel.NONE
            return AlertState(level=AlertLevel.NONE)

        if self._fatigue_start is None:
            self._fatigue_start = now

        elapsed = now - self._fatigue_start

        # ── Determine target alert level ──
        target = AlertLevel.NONE

        if fatigue_level == FatigueLevel.SEVERE and elapsed >= self._l3_dur:
            target = AlertLevel.LEVEL3
        elif elapsed >= self._l2_dur:
            target = AlertLevel.LEVEL2
        elif elapsed >= self._l1_dur:
            target = AlertLevel.LEVEL1

        # ── Apply cooldown ──
        if target != AlertLevel.NONE:
            last = self._last_alert_time.get(target)
            if last and (now - last) < self._cooldown:
                # Still in cooldown — stay at current level
                return AlertState(
                    level=self._current_level,
                    message=self._message_for(self._current_level),
                    seconds_at_level=elapsed,
                )

        # ── Escalate ──
        if target > self._current_level:
            self._current_level = target
            self._last_alert_time[target] = now
            self._trigger_alert(target)

        return AlertState(
            level=self._current_level,
            message=self._message_for(self._current_level),
            seconds_at_level=elapsed,
        )

    # ──────────────────────────────────────────────────────────────────
    # Alert triggers
    # ──────────────────────────────────────────────────────────────────

    def _trigger_alert(self, level: AlertLevel) -> None:
        """Fire the appropriate alert mechanism."""
        if level == AlertLevel.LEVEL2:
            self._play_beep(self._beep_l2, self._beep_ms)
        elif level == AlertLevel.LEVEL3:
            self._play_beep(self._beep_l3, self._beep_ms)
            if self._gpio:
                self._trigger_gpio()

    @staticmethod
    def _play_beep(freq: int, duration_ms: int) -> None:
        """Cross-platform beep."""
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.Beep(freq, duration_ms)
            elif system == "Linux":
                subprocess.Popen(
                    ["beep", "-f", str(freq), "-l", str(duration_ms)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "Darwin":
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Ping.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass  # non-critical — log in production

    def _trigger_gpio(self) -> None:
        """GPIO trigger placeholder for edge deployment."""
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._gpio_pin, GPIO.OUT)
            GPIO.output(self._gpio_pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(self._gpio_pin, GPIO.LOW)
        except ImportError:
            pass  # not on Raspberry Pi

    # ──────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_alert_overlay(
        frame: np.ndarray,
        alert: AlertState,
    ) -> np.ndarray:
        """
        Draw alert visual feedback onto the frame.

        Level 1: yellow border
        Level 2: orange border + warning text
        Level 3: red border + DANGER flash overlay
        """
        h, w = frame.shape[:2]

        if alert.level == AlertLevel.NONE:
            return frame

        colors = {
            AlertLevel.LEVEL1: (0, 220, 220),    # yellow
            AlertLevel.LEVEL2: (0, 140, 255),    # orange
            AlertLevel.LEVEL3: (0, 0, 255),      # red
        }
        color = colors.get(alert.level, (255, 255, 255))
        thickness = 4 if alert.level < AlertLevel.LEVEL3 else 8

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

        msg = alert.message or f"ALERT Level {alert.level}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        (tw, th_text), _ = cv2.getTextSize(msg, font, scale, 2)
        tx = (w - tw) // 2
        ty = 50

        # Background rectangle for readability
        cv2.rectangle(frame, (tx - 10, ty - th_text - 10),
                       (tx + tw + 10, ty + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, msg, (tx, ty), font, scale, color, 2, cv2.LINE_AA)

        # Level 3: flash overlay every ~500ms
        if alert.level == AlertLevel.LEVEL3:
            if int(time.monotonic() * 2) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), cv2.FILLED)
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        return frame

    @staticmethod
    def _message_for(level: AlertLevel) -> str:
        return {
            AlertLevel.NONE:   "",
            AlertLevel.LEVEL1: "Drowsiness Detected — Stay Alert!",
            AlertLevel.LEVEL2: "WARNING: Fatigue Increasing",
            AlertLevel.LEVEL3: "DANGER: Severe Fatigue — Pull Over!",
        }.get(level, "")

    def reset(self) -> None:
        self._fatigue_start = None
        self._current_level = AlertLevel.NONE
        self._last_alert_time.clear()
