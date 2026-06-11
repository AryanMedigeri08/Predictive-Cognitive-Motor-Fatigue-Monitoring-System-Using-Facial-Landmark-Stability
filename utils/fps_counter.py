"""
utils/fps_counter.py
====================
Lightweight rolling-window FPS counter.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """
    Measures frames-per-second via a fixed-size timestamp deque.

    FPS = (N - 1) / (t_N - t_1)   where N = window size.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._stamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Record a frame timestamp and return current FPS."""
        self._stamps.append(time.monotonic())
        if len(self._stamps) < 2:
            return 0.0
        dt = self._stamps[-1] - self._stamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._stamps) - 1) / dt

    def fps(self) -> float:
        """Current FPS without recording a new tick."""
        if len(self._stamps) < 2:
            return 0.0
        dt = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / dt if dt > 0 else 0.0
