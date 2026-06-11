"""
utils/math_utils.py
===================
Mathematical utility functions used across the fatigue monitoring system.
All operations use NumPy for vectorised, high-performance computation.

Key equations documented inline.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Distance helpers
# ──────────────────────────────────────────────────────────────────────

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Euclidean distance  ||p1 - p2||₂  between two points.

    Works for 2-D or 3-D coordinates.
    """
    return float(np.linalg.norm(
        np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    ))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return the unit vector  v / ||v||.  Returns zeros if ||v|| ≈ 0."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else np.zeros_like(v)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────────────────────────────
# Eye Aspect Ratio  (Soukupová & Čech, 2016)
# ──────────────────────────────────────────────────────────────────────

def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Eye Aspect Ratio (EAR).

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

    Landmark order:
        p1 = outer corner      p4 = inner corner
        p2 = upper-outer lid   p5 = lower-inner lid
        p3 = upper-inner lid   p6 = lower-outer lid

    Parameters
    ----------
    eye_landmarks : ndarray (6, 2)

    Returns
    -------
    float   EAR ∈ [0, ~0.5].  Near 0 → closed.
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    A = euclidean_distance(p2, p6)   # vertical 1
    B = euclidean_distance(p3, p5)   # vertical 2
    C = euclidean_distance(p1, p4)   # horizontal

    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ──────────────────────────────────────────────────────────────────────
# Mouth Aspect Ratio
# ──────────────────────────────────────────────────────────────────────

def compute_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio.

    MAR = vertical_distance / horizontal_distance

    Parameters
    ----------
    mouth_landmarks : ndarray (4, 2)
        [top_lip, bottom_lip, left_corner, right_corner]

    Returns
    -------
    float   MAR.  High values → mouth wide open (yawn).
    """
    top, bottom, left, right = mouth_landmarks
    vertical   = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


# ──────────────────────────────────────────────────────────────────────
# Head Pose — Euler Angle Decomposition
# ──────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """
    Convert 3×3 rotation matrix → (pitch, yaw, roll) in **degrees**.

    Uses ZYX Tait-Bryan convention:
        pitch = arctan2(-R[2,0], √(R[0,0]² + R[1,0]²))
        yaw   = arctan2(R[1,0], R[0,0])
        roll  = arctan2(R[2,1], R[2,2])
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0

    return float(pitch), float(yaw), float(roll)


# ──────────────────────────────────────────────────────────────────────
# Gaze Vector  (from iris landmarks)
# ──────────────────────────────────────────────────────────────────────

def compute_gaze_vector(
    iris_center: np.ndarray,
    eye_inner: np.ndarray,
    eye_outer: np.ndarray,
) -> np.ndarray:
    """
    Approximate 2-D gaze direction as the normalised vector from the
    eye centroid (midpoint of inner & outer corners) to the iris center.

    gaze = normalize(iris_center - eye_midpoint)

    Parameters
    ----------
    iris_center : ndarray (2,)   — center of iris (MediaPipe landmark 468 or 473)
    eye_inner   : ndarray (2,)   — inner eye corner
    eye_outer   : ndarray (2,)   — outer eye corner

    Returns
    -------
    ndarray (2,)   unit gaze vector in pixel space
    """
    eye_mid = (eye_inner + eye_outer) / 2.0
    direction = np.asarray(iris_center, dtype=np.float64) - np.asarray(eye_mid, dtype=np.float64)
    return normalize_vector(direction)


# ──────────────────────────────────────────────────────────────────────
# Neck Vector
# ──────────────────────────────────────────────────────────────────────

def compute_neck_vector(
    chin: np.ndarray,
    forehead: np.ndarray,
) -> np.ndarray:
    """
    Neck orientation approximated as the normalised vector from chin
    to forehead:   neck_vec = normalize(forehead - chin)

    Parameters
    ----------
    chin     : ndarray (2,)  — landmark 152
    forehead : ndarray (2,)  — landmark 10
    """
    return normalize_vector(
        np.asarray(forehead, dtype=np.float64) - np.asarray(chin, dtype=np.float64)
    )


# ──────────────────────────────────────────────────────────────────────
# Angular deviation between two unit vectors
# ──────────────────────────────────────────────────────────────────────

def angular_deviation(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle (degrees) between two vectors.

    θ = arccos(clamp(v1 · v2, -1, 1))

    Returns 0.0 if either vector is zero-length.
    """
    v1n = normalize_vector(v1)
    v2n = normalize_vector(v2)
    dot = float(np.dot(v1n, v2n))
    dot = clamp(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


# ──────────────────────────────────────────────────────────────────────
# EMA helper (single-step)
# ──────────────────────────────────────────────────────────────────────

def ema_update(prev: float, current: float, lam: float) -> float:
    """
    Single-step Exponential Moving Average.

    EMA_t = λ · x_t + (1 − λ) · EMA_{t-1}

    Parameters
    ----------
    prev    : previous EMA value
    current : new observation
    lam     : smoothing factor λ ∈ (0, 1).  Higher → more responsive.
    """
    return lam * current + (1.0 - lam) * prev
