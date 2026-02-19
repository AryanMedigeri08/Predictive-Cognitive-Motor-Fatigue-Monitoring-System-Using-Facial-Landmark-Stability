"""
config.py
=========
Central configuration for the Driver Fatigue & Cognitive Degradation
Monitoring System.  Every tunable parameter lives here so that no
business-logic file contains magic numbers.

Sections
--------
1.  Camera / Display
2.  Calibration
3.  MediaPipe Face Mesh
4.  Landmark indices
5.  Preprocessing / Night-mode / Glasses
6.  Feature extraction — EAR, PERCLOS, MAR, Blink, Gaze, Neck
7.  Dual-Axis Fatigue Model weights
8.  EMA / Smoothing
9.  Recovery dynamics
10. Confidence scoring
11. Alert system
12. Logging
13. Performance
14. Overlay colours
"""

from __future__ import annotations
import os

# ──────────────────────────────────────────────────────────────────────
# 1. Camera / Display
# ──────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
WINDOW_NAME   = "Driver Fatigue & Cognitive Degradation Monitor"

# ──────────────────────────────────────────────────────────────────────
# 2. Calibration
# ──────────────────────────────────────────────────────────────────────
CALIBRATION_DURATION_SEC = 90          # 90-second calibration session
CALIBRATION_JSON_PATH    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration_profile.json"
)

# ──────────────────────────────────────────────────────────────────────
# 3. MediaPipe Face Mesh
# ──────────────────────────────────────────────────────────────────────
FACE_MESH_MAX_FACES          = 1
FACE_MESH_REFINE_LANDMARKS   = True    # enables iris landmarks (478 pts)
FACE_MESH_MIN_DETECTION_CONF = 0.5
FACE_MESH_MIN_TRACKING_CONF  = 0.5

# ──────────────────────────────────────────────────────────────────────
# 4. Landmark Indices  (MediaPipe 478-point model w/ iris)
# ──────────────────────────────────────────────────────────────────────
# Eye: 6-point EAR order — p1(outer) p2(upper-outer) p3(upper-inner)
#                            p4(inner) p5(lower-inner) p6(lower-outer)
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

# Full eye contour (16 pts each) for drawing
LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33,  7,   163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

# Iris landmarks (mediapipe refine_landmarks=True provides these)
LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472]    # center + 4 boundary
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Mouth
MOUTH_VERTICAL_INDICES   = [13, 14]       # top lip, bottom lip
MOUTH_HORIZONTAL_INDICES = [61, 291]      # left corner, right corner
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Head pose — 6-point PnP model
HEAD_POSE_3D_MODEL_POINTS = [
    (0.0,     0.0,     0.0),      # Nose tip          — lm 1
    (0.0,   -330.0,  -65.0),      # Chin               — lm 152
    (-225.0,  170.0, -135.0),     # Left eye corner    — lm 263
    (225.0,   170.0, -135.0),     # Right eye corner   — lm 33
    (-150.0, -150.0, -125.0),     # Left mouth corner  — lm 287
    (150.0,  -150.0, -125.0),     # Right mouth corner — lm 57
]
HEAD_POSE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# Neck vector approximation: chin (152) to forehead (10)
NECK_CHIN_INDEX     = 152
NECK_FOREHEAD_INDEX = 10

# Nose tip for stability tracking
NOSE_TIP_INDEX = 1

# ──────────────────────────────────────────────────────────────────────
# 5. Preprocessing / Night-mode / Glasses
# ──────────────────────────────────────────────────────────────────────
# Night-mode auto-detection: if mean luminance < threshold → apply CLAHE
NIGHT_MODE_LUMINANCE_THRESHOLD = 60    # [0-255] scale
CLAHE_CLIP_LIMIT               = 3.0
CLAHE_TILE_GRID_SIZE           = (8, 8)
GAMMA_CORRECTION_VALUE         = 1.5   # gamma > 1 brightens darks
IR_SUPPORT_MODE                = False # flag for IR camera compatibility

# Glasses / glare
GLARE_SUPPRESSION_ENABLED = True
GLASSES_CONFIDENCE_THRESHOLD = 0.4     # if eye-region confidence < this
                                        # → fallback to head+blink+mouth

# ──────────────────────────────────────────────────────────────────────
# 6. Feature Extraction Thresholds
# ──────────────────────────────────────────────────────────────────────
# EAR — NOTE: NO hardcoded EAR threshold.
# Closure is determined as  EAR_t < EAR_0 * EAR_CLOSURE_RATIO
EAR_CLOSURE_RATIO       = 0.75     # 75% of baseline → closed
EAR_CONSEC_FRAMES_BLINK = 3
BLINK_RATE_NORMAL_MIN   = 10       # blinks / min
BLINK_RATE_NORMAL_MAX   = 20

# PERCLOS  (% eye closure over a sliding window)
PERCLOS_WINDOW_SEC      = 60.0     # sliding window length in seconds

# MAR — relative to baseline
MAR_YAWN_RATIO          = 1.5      # MAR_t > MAR_0 * 1.5 → yawn
MAR_CONSEC_FRAMES_YAWN  = 20

# Gaze drift
GAZE_DRIFT_WINDOW_SEC   = 5.0      # sliding window for drift calculation

# Micro-saccade
MICROSACCADE_WINDOW_SEC = 2.0      # sliding window for variance

# Neck stability
NECK_STABILITY_WINDOW   = 30       # frames

# Head pose deviation thresholds (degrees) — used for normalisation
HEAD_PITCH_THRESHOLD = 15.0
HEAD_YAW_THRESHOLD   = 20.0
HEAD_ROLL_THRESHOLD  = 20.0

# ──────────────────────────────────────────────────────────────────────
# 7. Dual-Axis Fatigue Model Weights
# ──────────────────────────────────────────────────────────────────────
# Physical Fatigue:  PF = w1*PERCLOS + w2*BlinkDuration + w3*HeadDev + w4*MOI
PF_W1_PERCLOS       = 0.35
PF_W2_BLINK_DUR     = 0.20
PF_W3_HEAD_DEV      = 0.25
PF_W4_MOUTH_INDEX   = 0.20

# Cognitive Fatigue: CF = v1*GazeDrift + v2*BlinkVar + v3*MicroSaccadeVar
CF_V1_GAZE_DRIFT    = 0.40
CF_V2_BLINK_VAR     = 0.30
CF_V3_MICROSACCADE  = 0.30

# Fatigue Index:  FI = alpha * PF + beta * CF
FI_ALPHA = 0.6
FI_BETA  = 0.4

# Classification thresholds on FI_smooth
FI_THRESHOLD_MODERATE = 0.35
FI_THRESHOLD_SEVERE   = 0.60

# ──────────────────────────────────────────────────────────────────────
# 8. EMA / Smoothing
# ──────────────────────────────────────────────────────────────────────
# Exponential Moving Average for FI:
#   FI_smooth = EMA_LAMBDA * FI + (1 - EMA_LAMBDA) * FI_prev
EMA_LAMBDA = 0.15

# Sliding-window sizes (frames)
SMOOTHING_WINDOW_EAR       = 5
SMOOTHING_WINDOW_MAR       = 5
SMOOTHING_WINDOW_HEAD      = 7
SMOOTHING_WINDOW_STABILITY = 10

# Kalman filter (head pose)
KALMAN_PROCESS_NOISE     = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2
KALMAN_INITIAL_ERROR     = 1.0

# ──────────────────────────────────────────────────────────────────────
# 9. Recovery Dynamics
# ──────────────────────────────────────────────────────────────────────
# Fatigue decay:  FI_t = FI_prev * exp(-RECOVERY_DECAY_K * dt)
RECOVERY_DECAY_K              = 0.05   # decay rate constant (1/s)
RECOVERY_EAR_RELATIVE_THRESH  = 0.95   # EAR_rel > this → recovery eligible
RECOVERY_HEAD_DEV_THRESH      = 0.15   # head deviation < this → recovery
RECOVERY_BLINK_TOLERANCE      = 0.3    # |BPM - BF_0| / BF_0 < this

# ──────────────────────────────────────────────────────────────────────
# 10. Confidence Scoring
# ──────────────────────────────────────────────────────────────────────
# Confidence = w_lm * LM_conf + w_light * Light_score + w_blur * Blur_score
CONFIDENCE_W_LANDMARK  = 0.50
CONFIDENCE_W_LIGHTING  = 0.30
CONFIDENCE_W_BLUR      = 0.20
CONFIDENCE_LOW_THRESH  = 0.5    # below this → "Low Detection Confidence"

# Blur detection
BLUR_LAPLACIAN_THRESHOLD = 100.0   # Laplacian variance < this → blurry

# ──────────────────────────────────────────────────────────────────────
# 11. Alert System
# ──────────────────────────────────────────────────────────────────────
ALERT_LEVEL1_DURATION_SEC = 3.0    # seconds of Moderate before Level-1
ALERT_LEVEL2_DURATION_SEC = 6.0    # seconds before Level-2
ALERT_LEVEL3_DURATION_SEC = 10.0   # seconds of Severe before Level-3
ALERT_COOLDOWN_SEC        = 5.0    # min seconds between same-level alerts

# Audio
BEEP_FREQ_LEVEL2 = 800
BEEP_FREQ_LEVEL3 = 1200
BEEP_DURATION_MS = 500

# GPIO placeholder (for edge deployment)
GPIO_ALERT_ENABLED = False
GPIO_PIN_ALERT     = 18

# ──────────────────────────────────────────────────────────────────────
# 12. Logging
# ──────────────────────────────────────────────────────────────────────
CSV_LOG_ENABLED   = True
CSV_LOG_DIR       = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs"
)
CSV_LOG_INTERVAL  = 1.0   # seconds between writes
LOG_LEVEL         = "INFO"

# ──────────────────────────────────────────────────────────────────────
# 13. Performance
# ──────────────────────────────────────────────────────────────────────
TARGET_FPS              = 20
FPS_WINDOW_SIZE         = 30
CAPTURE_THREAD_ENABLED  = True     # threaded capture pipeline

# Stability tracking
STABILITY_WINDOW_SIZE   = 30
STABILITY_LANDMARK_IDX  = 1        # nose-tip

# ──────────────────────────────────────────────────────────────────────
# 14. Overlay Colours (BGR)
# ──────────────────────────────────────────────────────────────────────
COLOR_GREEN   = (0,   220,   0)
COLOR_YELLOW  = (0,   220, 220)
COLOR_ORANGE  = (0,   140, 255)
COLOR_RED     = (0,     0, 255)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,     0,   0)
COLOR_CYAN    = (255, 220,   0)
COLOR_MAGENTA = (255,   0, 200)

OVERLAY_ALPHA = 0.4
