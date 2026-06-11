"""
main.py
=======
Main orchestrator for the Driver Fatigue & Cognitive Degradation
Monitoring System.

Pipeline (per frame):
  1. Capture  →  camera grab (threaded for non-blocking I/O)
  2. Preprocess → CLAHE / gamma / glare suppression
  3. Landmarks → MediaPipe Face Mesh (478 points + iris)
  4. Features  → EAR, PERCLOS, MAR, HeadPose, Gaze, Neck
  5. Confidence → composite detection confidence
  6. Glasses Fallback → reweight features when eye region unreliable
  7. Fatigue Model → PF + CF → FI (EMA-smoothed)
  8. Recovery  → exponential decay when driver recovers
  9. Alert     → 3-level escalation with confidence gating
  10. Render   → landmark overlays + dashboard + alert banners
  11. Log      → CSV row (rate-limited)

Performance target: ≥ 20 FPS on commodity hardware.

Usage:
    python main.py
    python main.py --calibrate          force re-calibration
    python main.py --profile path.json  load specific profile
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# ── Local imports ──
import config as cfg

from preprocessing.image_enhancer import ImageEnhancer
from landmarks.face_mesh import FaceMeshDetector
from features.eye_features import EyeFeatureExtractor, EyeFeatures
from features.mouth_features import MouthFeatureExtractor, MouthFeatures
from features.head_features import HeadFeatureExtractor, HeadFeatures
from features.gaze_features import GazeFeatureExtractor, GazeFeatures
from features.neck_features import NeckFeatureExtractor, NeckFeatures
from calibration.calibration_engine import CalibrationEngine, CalibrationProfile
from models.fatigue_model import (
    DualAxisFatigueModel, FatigueInput, FatigueResult, FatigueLevel,
)
from models.confidence import ConfidenceScorer, ConfidenceResult
from recovery.recovery_tracker import RecoveryTracker, RecoveryState
from alerts.alert_manager import AlertManager, AlertState
from visualization.overlay_renderer import OverlayRenderer
from logging_system.csv_logger import CSVLogger
from utils.fps_counter import FPSCounter


# ══════════════════════════════════════════════════════════════════════
# Threaded Camera Capture
# ══════════════════════════════════════════════════════════════════════

class CameraCapture:
    """
    Non-blocking camera reader using a background thread.

    The main loop always gets the **latest** frame instantly, decoupled
    from the camera's I/O latency.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraCapture":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════
# Blink Duration Normalisation
# ══════════════════════════════════════════════════════════════════════

def normalise_blink_duration(dur_ms: float, max_ms: float = 500.0) -> float:
    """Map blink duration [0, max_ms] ms → [0, 1]."""
    return min(dur_ms / max_ms, 1.0) if max_ms > 0 else 0.0


def normalise_blink_variance(var: float, max_var: float = 200.0) -> float:
    """Map blink variance [0, max_var] → [0, 1]."""
    return min(var / max_var, 1.0) if max_var > 0 else 0.0


def normalise_mouth_index(moi: float) -> float:
    """
    Mouth Opening Index from calibration ratio.
    MOI = MAR / MAR_0.  Normal ≈ 1.0;  yawn ≈ 2.0+
    Map [1, 3] → [0, 1].
    """
    return min(max((moi - 1.0) / 2.0, 0.0), 1.0)


# ══════════════════════════════════════════════════════════════════════
# Main Loop
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Driver Fatigue Monitor")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force re-calibration even if profile exists.")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to a calibration profile JSON.")
    args = parser.parse_args()

    # ── Initialise components ──
    camera = CameraCapture(
        cfg.CAMERA_INDEX, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT
    ).start()

    enhancer = ImageEnhancer(
        luminance_threshold=cfg.NIGHT_MODE_LUMINANCE_THRESHOLD,
        clahe_clip_limit=cfg.CLAHE_CLIP_LIMIT,
        clahe_tile_size=cfg.CLAHE_TILE_GRID_SIZE,
        gamma=cfg.GAMMA_CORRECTION_VALUE,
        glare_suppression=cfg.GLARE_SUPPRESSION_ENABLED,
        blur_threshold=cfg.BLUR_LAPLACIAN_THRESHOLD,
    )

    detector = FaceMeshDetector(
        max_faces=cfg.FACE_MESH_MAX_FACES,
        refine_landmarks=cfg.FACE_MESH_REFINE_LANDMARKS,
        min_detection_confidence=cfg.FACE_MESH_MIN_DETECTION_CONF,
        min_tracking_confidence=cfg.FACE_MESH_MIN_TRACKING_CONF,
    )

    eye_ext = EyeFeatureExtractor(
        left_eye_indices=cfg.LEFT_EYE_INDICES,
        right_eye_indices=cfg.RIGHT_EYE_INDICES,
        ear_closure_ratio=cfg.EAR_CLOSURE_RATIO,
        consec_frames_blink=cfg.EAR_CONSEC_FRAMES_BLINK,
        perclos_window_sec=cfg.PERCLOS_WINDOW_SEC,
        smoothing_window=cfg.SMOOTHING_WINDOW_EAR,
    )

    mouth_ext = MouthFeatureExtractor(
        vertical_indices=cfg.MOUTH_VERTICAL_INDICES,
        horizontal_indices=cfg.MOUTH_HORIZONTAL_INDICES,
        yawn_ratio=cfg.MAR_YAWN_RATIO,
        consec_frames_yawn=cfg.MAR_CONSEC_FRAMES_YAWN,
        smoothing_window=cfg.SMOOTHING_WINDOW_MAR,
    )

    head_ext = HeadFeatureExtractor(
        model_points_3d=cfg.HEAD_POSE_3D_MODEL_POINTS,
        landmark_indices=cfg.HEAD_POSE_LANDMARK_INDICES,
        pitch_threshold=cfg.HEAD_PITCH_THRESHOLD,
        yaw_threshold=cfg.HEAD_YAW_THRESHOLD,
        roll_threshold=cfg.HEAD_ROLL_THRESHOLD,
        kalman_process_noise=cfg.KALMAN_PROCESS_NOISE,
        kalman_measurement_noise=cfg.KALMAN_MEASUREMENT_NOISE,
    )

    gaze_ext = GazeFeatureExtractor(
        left_iris_indices=cfg.LEFT_IRIS_INDICES,
        right_iris_indices=cfg.RIGHT_IRIS_INDICES,
        left_eye_inner=263,
        left_eye_outer=362,
        right_eye_inner=133,
        right_eye_outer=33,
        drift_window_sec=cfg.GAZE_DRIFT_WINDOW_SEC,
        microsaccade_window_sec=cfg.MICROSACCADE_WINDOW_SEC,
    )

    neck_ext = NeckFeatureExtractor(
        chin_index=cfg.NECK_CHIN_INDEX,
        forehead_index=cfg.NECK_FOREHEAD_INDEX,
        window_size=cfg.NECK_STABILITY_WINDOW,
        smoothing_window=cfg.SMOOTHING_WINDOW_STABILITY,
    )

    calib_engine = CalibrationEngine(
        duration_sec=cfg.CALIBRATION_DURATION_SEC,
        json_path=args.profile or cfg.CALIBRATION_JSON_PATH,
    )

    fatigue_model = DualAxisFatigueModel(
        w1=cfg.PF_W1_PERCLOS, w2=cfg.PF_W2_BLINK_DUR,
        w3=cfg.PF_W3_HEAD_DEV, w4=cfg.PF_W4_MOUTH_INDEX,
        v1=cfg.CF_V1_GAZE_DRIFT, v2=cfg.CF_V2_BLINK_VAR,
        v3=cfg.CF_V3_MICROSACCADE,
        alpha=cfg.FI_ALPHA, beta=cfg.FI_BETA,
        ema_lambda=cfg.EMA_LAMBDA,
        threshold_moderate=cfg.FI_THRESHOLD_MODERATE,
        threshold_severe=cfg.FI_THRESHOLD_SEVERE,
    )

    confidence_scorer = ConfidenceScorer(
        w_landmark=cfg.CONFIDENCE_W_LANDMARK,
        w_lighting=cfg.CONFIDENCE_W_LIGHTING,
        w_blur=cfg.CONFIDENCE_W_BLUR,
        low_threshold=cfg.CONFIDENCE_LOW_THRESH,
        blur_laplacian_threshold=cfg.BLUR_LAPLACIAN_THRESHOLD,
    )

    recovery_tracker = RecoveryTracker(
        decay_k=cfg.RECOVERY_DECAY_K,
        ear_rel_thresh=cfg.RECOVERY_EAR_RELATIVE_THRESH,
        head_dev_thresh=cfg.RECOVERY_HEAD_DEV_THRESH,
        blink_tolerance=cfg.RECOVERY_BLINK_TOLERANCE,
    )

    alert_mgr = AlertManager(
        level1_dur=cfg.ALERT_LEVEL1_DURATION_SEC,
        level2_dur=cfg.ALERT_LEVEL2_DURATION_SEC,
        level3_dur=cfg.ALERT_LEVEL3_DURATION_SEC,
        cooldown_sec=cfg.ALERT_COOLDOWN_SEC,
        confidence_threshold=cfg.CONFIDENCE_LOW_THRESH,
        gpio_enabled=cfg.GPIO_ALERT_ENABLED,
        gpio_pin=cfg.GPIO_PIN_ALERT,
        beep_freq_l2=cfg.BEEP_FREQ_LEVEL2,
        beep_freq_l3=cfg.BEEP_FREQ_LEVEL3,
        beep_duration_ms=cfg.BEEP_DURATION_MS,
    )

    csv_logger = CSVLogger(
        log_dir=cfg.CSV_LOG_DIR,
        interval=cfg.CSV_LOG_INTERVAL,
        log_level=cfg.LOG_LEVEL,
    )
    logger = csv_logger.logger

    fps_counter = FPSCounter(cfg.FPS_WINDOW_SIZE)
    renderer = OverlayRenderer()

    # ── State flags ──
    calibrating = True
    profile: Optional[CalibrationProfile] = None

    # ── Check for existing calibration profile ──
    if not args.calibrate and calib_engine.has_saved_profile():
        try:
            profile = calib_engine.load_profile()
            _apply_calibration(
                profile, eye_ext, mouth_ext, head_ext, gaze_ext, neck_ext,
            )
            calibrating = False
            logger.info("Loaded existing calibration profile.")
        except Exception as e:
            logger.warning(f"Failed to load profile: {e}. Re-calibrating.")

    if calibrating:
        calib_engine.start()
        logger.info(f"Starting {cfg.CALIBRATION_DURATION_SEC}s calibration...")

    # ══════════════════════════════════════════════════════════════════
    # Frame loop
    # ══════════════════════════════════════════════════════════════════

    prev_blink_count = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            fps = fps_counter.tick()
            h, w = frame.shape[:2]

            # ── 1. Preprocess ──
            enhanced, enhance_info = enhancer.enhance(frame)

            # ── 2. Landmarks ──
            lm_result = detector.process(enhanced)

            if lm_result.landmarks_px is None:
                # No face detected — show blank frame
                if calibrating:
                    frame = renderer.draw_calibration_screen(
                        frame, calib_engine.progress, calib_engine.remaining_sec,
                    )
                cv2.imshow(cfg.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            landmarks = lm_result.landmarks_px

            # ── 3. Extract features ──
            eye_feat = eye_ext.update(landmarks)
            mouth_feat = mouth_ext.update(landmarks)
            head_feat = head_ext.update(landmarks, (h, w))
            gaze_feat = gaze_ext.update(landmarks)
            neck_feat = neck_ext.update(landmarks)

            # ── 4. Calibration phase ──
            if calibrating:
                # Accumulate calibration samples
                raw_ear = eye_ext.compute_raw_ear(landmarks)
                raw_mar = mouth_ext.compute_raw_mar(landmarks)
                raw_pitch, raw_yaw, raw_roll = head_ext.compute_raw_euler(
                    landmarks, (h, w))
                raw_gaze = gaze_ext.compute_raw_gaze(landmarks)
                raw_neck = neck_ext.compute_raw_neck_vector(landmarks)

                calib_engine.add_sample(
                    ear=raw_ear, mar=raw_mar,
                    pitch=raw_pitch, yaw=raw_yaw, roll=raw_roll,
                    gaze_vec=raw_gaze, neck_vec=raw_neck,
                )

                # Track blinks during calibration
                if eye_feat.blink_count > prev_blink_count:
                    calib_engine.register_blink()
                    prev_blink_count = eye_feat.blink_count

                # Draw calibration overlay
                frame = renderer.draw_calibration_screen(
                    frame, calib_engine.progress, calib_engine.remaining_sec,
                )

                # Draw landmarks even during calibration so user can see
                frame = renderer.draw_eye_contours(frame, landmarks)
                frame = renderer.draw_iris_centres(frame, landmarks)
                frame = renderer.draw_mouth_contour(frame, landmarks)

                cv2.imshow(cfg.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if calib_engine.is_complete:
                    profile = calib_engine.finalise()
                    calib_engine.save_profile()
                    _apply_calibration(
                        profile, eye_ext, mouth_ext, head_ext,
                        gaze_ext, neck_ext,
                    )
                    calibrating = False
                    # Reset feature extractors for clean start
                    eye_ext.reset()
                    mouth_ext.reset()
                    gaze_ext.reset()
                    neck_ext.reset()
                    head_ext.reset()
                    prev_blink_count = 0
                    logger.info(
                        f"Calibration complete. EAR_0={profile.ear_0:.3f}, "
                        f"MAR_0={profile.mar_0:.3f}, BF_0={profile.bf_0:.1f}"
                    )

                continue

            # ── 5. Confidence ──
            # Glasses fallback: check eye-region confidence
            eye_region_conf = detector.get_region_confidence(
                lm_result.landmarks_3d,
                cfg.LEFT_EYE_INDICES + cfg.RIGHT_EYE_INDICES,
            )
            effective_lm_conf = lm_result.landmark_confidence

            confidence = confidence_scorer.compute(
                landmark_confidence=effective_lm_conf,
                lighting_score=enhance_info.lighting_score,
                blur_laplacian_var=enhance_info.blur_score,
            )

            # ── 6. Glasses fallback ──
            # When eye region is unreliable, reduce eye-feature
            # contribution and rely more on head + mouth + blink rate
            glasses_fallback = eye_region_conf < cfg.GLASSES_CONFIDENCE_THRESHOLD
            eye_weight = 1.0
            if glasses_fallback:
                eye_weight = max(eye_region_conf / cfg.GLASSES_CONFIDENCE_THRESHOLD, 0.1)

            # ── 7. Fatigue Model ──
            fatigue_input = FatigueInput(
                perclos=eye_feat.perclos * eye_weight,
                blink_duration_norm=normalise_blink_duration(
                    eye_feat.blink_duration_ms) * eye_weight,
                head_deviation=head_feat.deviation_score,
                mouth_opening_index=normalise_mouth_index(
                    mouth_feat.mouth_opening_index),
                gaze_drift=gaze_feat.gaze_drift * eye_weight,
                blink_variance_norm=normalise_blink_variance(
                    eye_feat.blink_variance),
                microsaccade_variance=gaze_feat.microsaccade_variance * eye_weight,
            )

            fatigue_result = fatigue_model.compute(fatigue_input)

            # ── 8. Recovery ──
            bf_0 = profile.bf_0 if profile else 15.0
            recovery_state = recovery_tracker.update(
                fi_smooth=fatigue_result.fatigue_index_smooth,
                ear_relative=eye_feat.ear_relative,
                head_deviation=head_feat.deviation_score,
                bpm=eye_feat.blinks_per_minute,
                bf_0=bf_0,
            )

            if recovery_state.is_recovering:
                fatigue_model.override_fi_smooth(recovery_state.fi_decayed)
                fatigue_result.fatigue_index_smooth = recovery_state.fi_decayed
                # Re-classify after recovery adjustment
                if recovery_state.fi_decayed < cfg.FI_THRESHOLD_MODERATE:
                    fatigue_result.level = FatigueLevel.NORMAL
                elif recovery_state.fi_decayed < cfg.FI_THRESHOLD_SEVERE:
                    fatigue_result.level = FatigueLevel.MODERATE
                else:
                    fatigue_result.level = FatigueLevel.SEVERE

            # ── 9. Alert ──
            alert_state = alert_mgr.update(
                fatigue_level=fatigue_result.level,
                confidence=confidence.composite,
            )

            # ── 10. Render ──
            display = enhanced.copy()

            # Head pose axes
            display = head_ext.draw_axes(display, head_feat)

            # Full overlays
            display = renderer.render_all(
                frame=display,
                landmarks=landmarks,
                eye=eye_feat,
                mouth=mouth_feat,
                head=head_feat,
                gaze=gaze_feat,
                neck=neck_feat,
                fatigue=fatigue_result,
                confidence=confidence,
                recovery=recovery_state,
                fps=fps,
            )

            # Alert overlay
            display = AlertManager.draw_alert_overlay(display, alert_state)

            # Glasses fallback indicator
            if glasses_fallback:
                cv2.putText(
                    display,
                    "Glasses Mode: Head+Mouth Fallback",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    cfg.COLOR_ORANGE, 1, cv2.LINE_AA,
                )

            # Night mode indicator
            if enhance_info.is_night_mode:
                cv2.putText(
                    display, "Night Mode Active",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    cfg.COLOR_CYAN, 1, cv2.LINE_AA,
                )

            cv2.imshow(cfg.WINDOW_NAME, display)

            # ── 11. Log ──
            csv_logger.log(
                eye=eye_feat,
                mouth=mouth_feat,
                head=head_feat,
                gaze=gaze_feat,
                neck=neck_feat,
                fatigue=fatigue_result,
                confidence=confidence,
                alert=alert_state,
                recovery=recovery_state,
            )

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                break
            elif key == ord("r"):
                logger.info("Manual reset — restarting calibration.")
                calibrating = True
                calib_engine = CalibrationEngine(
                    duration_sec=cfg.CALIBRATION_DURATION_SEC,
                    json_path=cfg.CALIBRATION_JSON_PATH,
                )
                calib_engine.start()
                fatigue_model.reset()
                recovery_tracker.reset()
                alert_mgr.reset()
                prev_blink_count = 0

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        camera.stop()
        detector.close()
        csv_logger.close()
        cv2.destroyAllWindows()
        print("\n[INFO] System shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════
# Apply Calibration Profile
# ══════════════════════════════════════════════════════════════════════

def _apply_calibration(
    profile: CalibrationProfile,
    eye_ext: EyeFeatureExtractor,
    mouth_ext: MouthFeatureExtractor,
    head_ext: HeadFeatureExtractor,
    gaze_ext: GazeFeatureExtractor,
    neck_ext: NeckFeatureExtractor,
) -> None:
    """Push calibration baselines into all feature extractors."""
    eye_ext.set_baseline(profile.ear_0)
    mouth_ext.set_baseline(profile.mar_0)
    head_ext.set_baseline(profile.pitch_0, profile.yaw_0, profile.roll_0)
    gaze_ext.set_baseline(np.array(profile.gaze_0))
    neck_ext.set_baseline(np.array(profile.neck_0))


if __name__ == "__main__":
    main()
