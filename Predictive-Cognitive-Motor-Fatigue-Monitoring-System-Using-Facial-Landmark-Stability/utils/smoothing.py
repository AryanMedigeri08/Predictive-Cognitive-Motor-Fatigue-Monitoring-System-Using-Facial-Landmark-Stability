"""
utils/smoothing.py
==================
Smoothing utilities for temporal signal stabilisation.

Provides three smoothers:
  1. SlidingWindowSmoother  — deque-based moving average
  2. KalmanFilter1D         — lightweight 1-D Kalman filter (head-pose)
  3. ExponentialMovingAverage — EMA for fatigue index smoothing
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Sliding Window Smoother
# ──────────────────────────────────────────────────────────────────────

class SlidingWindowSmoother:
    """
    Simple moving average over a fixed-size sliding window.

    Mean = (1/N) Σᵢ xᵢ   where N = min(window_size, samples_received)
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Add *value* and return the current mean."""
        self._window.append(value)
        return float(np.mean(self._window))

    def get(self) -> float:
        """Current mean without adding a new sample."""
        return float(np.mean(self._window)) if self._window else 0.0

    def reset(self) -> None:
        self._window.clear()

    @property
    def full(self) -> bool:
        return len(self._window) == self._window.maxlen

    @property
    def count(self) -> int:
        return len(self._window)

    @property
    def values(self) -> list[float]:
        """Return a copy of all stored samples."""
        return list(self._window)


# ──────────────────────────────────────────────────────────────────────
# 2. 1-D Kalman Filter
# ──────────────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Scalar Kalman filter for noisy 1-D signals.

    State model (constant-velocity, zero control input):
        x_k = x_{k-1} + w_k          w_k ~ N(0, Q)
        z_k = x_k     + v_k          v_k ~ N(0, R)

    Update equations:
        Predict:  P⁻ = P + Q
        Gain:     K  = P⁻ / (P⁻ + R)
        Update:   x  = x + K (z - x)
                  P  = (1 - K) P⁻
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_error: float = 1.0,
    ) -> None:
        self._Q = process_noise
        self._R = measurement_noise
        self._P = initial_error
        self._x: Optional[float] = None

    def update(self, measurement: float) -> float:
        """Process a new measurement and return the filtered estimate."""
        if self._x is None:
            self._x = measurement
            return measurement

        P_pred = self._P + self._Q
        K = P_pred / (P_pred + self._R)
        self._x = self._x + K * (measurement - self._x)
        self._P = (1.0 - K) * P_pred
        return float(self._x)

    def reset(self) -> None:
        self._x = None
        self._P = 1.0


# ──────────────────────────────────────────────────────────────────────
# 3. Exponential Moving Average
# ──────────────────────────────────────────────────────────────────────

class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) smoother.

    EMA_t = λ · x_t  +  (1 − λ) · EMA_{t−1}

    Properties:
      - Higher λ → more weight on recent values (responsive but noisy)
      - Lower  λ → smoother but more lagging
      - For fatigue index,  λ = 0.15 is a good default.
    """

    def __init__(self, lam: float = 0.15) -> None:
        self._lam = lam
        self._value: Optional[float] = None

    def update(self, x: float) -> float:
        """Feed new observation *x* and return the smoothed value."""
        if self._value is None:
            self._value = x
        else:
            self._value = self._lam * x + (1.0 - self._lam) * self._value
        return self._value

    @property
    def value(self) -> float:
        """Current EMA value."""
        return self._value if self._value is not None else 0.0

    def reset(self) -> None:
        self._value = None
