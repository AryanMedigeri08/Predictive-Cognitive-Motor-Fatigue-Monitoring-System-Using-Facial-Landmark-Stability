"""
vision/head_pose.py
===================
Head pose estimation using OpenCV's solvePnP.

Maps 6 facial landmarks to a generic 3-D face model and solves for the
rotation and translation vectors.  The rotation vector is converted to
Euler angles (pitch, yaw, roll) via Rodrigues decomposition.

Kalman filters are applied to each Euler angle for temporal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from utils.math_utils import rotation_matrix_to_euler_angles
from utils.smoothing import KalmanFilter1D


@dataclass
class HeadPoseState:
    """Snapshot of head pose for a single frame."""
    pitch: float = 0.0   # Nodding forward/backward  (degrees)
    yaw: float   = 0.0   # Turning left/right         (degrees)
    roll: float  = 0.0   # Tilting left/right         (degrees)
    pitch_raw: float = 0.0
    yaw_raw: float   = 0.0
    roll_raw: float  = 0.0
    deviation_score: float = 0.0   # Normalised [0, 1] head deviation


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) from 6 facial landmarks.

    Uses a generic 3-D face model (millimetre scale) and OpenCV's
    ``solvePnP`` with the EPNP algorithm.

    Parameters
    ----------
    model_points_3d : list of (float, float, float)
        3-D coordinates of the generic face model points.
    landmark_indices : list[int]
        MediaPipe landmark indices corresponding to each model point.
    pitch_threshold : float
        Pitch deviation (°) above which a head-down alert is raised.
    yaw_threshold : float
        Yaw deviation (°) above which a head-turn alert is raised.
    roll_threshold : float
        Roll deviation (°) above which a head-tilt alert is raised.
    kalman_process_noise : float
        Kalman filter process noise Q.
    kalman_measurement_noise : float
        Kalman filter measurement noise R.
    """

    def __init__(
        self,
        model_points_3d: List[Tuple[float, float, float]],
        landmark_indices: List[int],
        pitch_threshold: float = 15.0,
        yaw_threshold: float   = 20.0,
        roll_threshold: float  = 20.0,
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
    ) -> None:
        self._model_pts = np.array(model_points_3d, dtype=np.float64)
        self._lm_indices = landmark_indices
        self._pitch_thresh = pitch_threshold
        self._yaw_thresh   = yaw_threshold
        self._roll_thresh  = roll_threshold

        # Kalman filters for each Euler angle
        kw = dict(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
        self._kf_pitch = KalmanFilter1D(**kw)
        self._kf_yaw   = KalmanFilter1D(**kw)
        self._kf_roll  = KalmanFilter1D(**kw)

        # Camera intrinsics (estimated; updated on first frame)
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Calibration baselines
        self._baseline_pitch: float | None = None
        self._baseline_yaw: float | None   = None
        self._baseline_roll: float | None  = None
        self._calibration_samples: List[Tuple[float, float, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, landmarks: np.ndarray, frame_shape: Tuple[int, int]
    ) -> HeadPoseState:
        """
        Estimate head pose from landmarks.

        Parameters
        ----------
        landmarks : np.ndarray of shape (468, 2)
            Full set of MediaPipe facial landmarks in pixel space.
        frame_shape : (height, width)
            Shape of the source frame (used to build camera matrix).

        Returns
        -------
        HeadPoseState
            Euler angles and deviation score.
        """
        h, w = frame_shape

        # Build/update camera intrinsic matrix (pinhole approximation)
        if self._camera_matrix is None:
            focal = w  # Approximate focal length = image width
            cx, cy = w / 2.0, h / 2.0
            self._camera_matrix = np.array(
                [[focal, 0, cx],
                 [0, focal, cy],
                 [0,     0,  1]],
                dtype=np.float64,
            )

        # Extract the 6 image points
        image_pts = np.array(
            [landmarks[i] for i in self._lm_indices], dtype=np.float64
        )  # shape (6, 2)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self._model_pts,
            image_pts,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success:
            return HeadPoseState()

        # Convert rotation vector → rotation matrix via Rodrigues
        R, _ = cv2.Rodrigues(rvec)

        pitch_raw, yaw_raw, roll_raw = rotation_matrix_to_euler_angles(R)

        # Apply Kalman smoothing
        pitch = self._kf_pitch.update(pitch_raw)
        yaw   = self._kf_yaw.update(yaw_raw)
        roll  = self._kf_roll.update(roll_raw)

        # Compute deviation score (normalised [0, 1])
        deviation_score = self._compute_deviation_score(pitch, yaw, roll)

        return HeadPoseState(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            pitch_raw=pitch_raw,
            yaw_raw=yaw_raw,
            roll_raw=roll_raw,
            deviation_score=deviation_score,
        )

    def add_calibration_sample(
        self, landmarks: np.ndarray, frame_shape: Tuple[int, int]
    ) -> None:
        """Collect a head-pose sample during the calibration phase."""
        state = self.update(landmarks, frame_shape)
        self._calibration_samples.append((state.pitch, state.yaw, state.roll))

    def finalise_calibration(self) -> None:
        """Compute baseline Euler angles from calibration samples."""
        if not self._calibration_samples:
            return
        arr = np.array(self._calibration_samples)
        self._baseline_pitch = float(np.mean(arr[:, 0]))
        self._baseline_yaw   = float(np.mean(arr[:, 1]))
        self._baseline_roll  = float(np.mean(arr[:, 2]))
        self._calibration_samples.clear()

    def draw_axes(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
        axis_length: float = 100.0,
    ) -> np.ndarray:
        """
        Draw 3-D coordinate axes on the nose tip to visualise head pose.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to annotate.
        landmarks : np.ndarray of shape (468, 2)
        frame_shape : (height, width)
        axis_length : float
            Length of each axis in the 3-D model coordinate system.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        h, w = frame_shape
        if self._camera_matrix is None:
            return frame

        image_pts = np.array(
            [landmarks[i] for i in self._lm_indices], dtype=np.float64
        )
        success, rvec, tvec = cv2.solvePnP(
            self._model_pts, image_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return frame

        # Project axis endpoints
        axis_pts = np.float32([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ])
        nose_tip_3d = np.float32([[0, 0, 0]])

        proj_nose, _ = cv2.projectPoints(
            nose_tip_3d, rvec, tvec, self._camera_matrix, self._dist_coeffs
        )
        proj_axes, _ = cv2.projectPoints(
            axis_pts, rvec, tvec, self._camera_matrix, self._dist_coeffs
        )

        origin = tuple(proj_nose[0].ravel().astype(int))
        x_end  = tuple(proj_axes[0].ravel().astype(int))
        y_end  = tuple(proj_axes[1].ravel().astype(int))
        z_end  = tuple(proj_axes[2].ravel().astype(int))

        cv2.line(frame, origin, x_end, (0, 0, 255),   2)  # X → Red
        cv2.line(frame, origin, y_end, (0, 255, 0),   2)  # Y → Green
        cv2.line(frame, origin, z_end, (255, 0, 0),   2)  # Z → Blue

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_deviation_score(
        self, pitch: float, yaw: float, roll: float
    ) -> float:
        """
        Compute a normalised [0, 1] deviation score from the neutral pose.

        Subtracts the calibration baseline (if available) and normalises
        each angle by its threshold before combining.

        Returns
        -------
        float
            Deviation score in [0, 1].
        """
        bp = self._baseline_pitch or 0.0
        by = self._baseline_yaw   or 0.0
        br = self._baseline_roll  or 0.0

        dp = abs(pitch - bp) / max(self._pitch_thresh, 1.0)
        dy = abs(yaw   - by) / max(self._yaw_thresh,   1.0)
        dr = abs(roll  - br) / max(self._roll_thresh,  1.0)

        # Take the max deviation across axes, clamped to [0, 1]
        score = min(max(dp, dy, dr), 1.0)
        return float(score)
