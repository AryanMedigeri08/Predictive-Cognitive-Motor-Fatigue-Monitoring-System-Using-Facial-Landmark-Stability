# Predictive-Cognitive-Motor-Fatigue-Monitoring-System-Using-Facial-Landmark-Stability

# Driver Fatigue & Cognitive Degradation Monitoring System

A production-ready, real-time driver fatigue monitoring system built with Python, OpenCV, and MediaPipe. Features personalised calibration, dual-axis fatigue modelling, recovery dynamics, and edge-deployable architecture.

## Features

| Feature | Description |
|---------|-------------|
| **90s Personalised Calibration** | Establishes baselines (EAR₀, MAR₀, BF₀, H₀, G₀, Neck₀) adaptive to any morphology |
| **Dual-Axis Fatigue Model** | Physical Fatigue (PF) + Cognitive Fatigue (CF) → Fatigue Index (FI) |
| **PERCLOS** | 60-second sliding-window percentage of eye closure |
| **Gaze Drift & Micro-Saccade** | Iris-based gaze tracking for cognitive load detection |
| **Recovery Dynamics** | Exponential decay: FI_t = FI_prev · exp(−k · Δt) |
| **Glasses/Glare Handling** | Glare suppression + confidence-gated fallback to head/mouth |
| **Night Mode** | Auto-activated CLAHE + gamma correction in low-light |
| **3-Level Alert Escalation** | Visual → Audio → Critical alarm with confidence gating |
| **Confidence Scoring** | Composite metric (landmark + lighting + blur quality) |
| **CSV Logging** | Daily-rotating CSV with all 25 features per row |
| **Threaded Pipeline** | Non-blocking camera capture for ≥20 FPS |

## Architecture

```
driver_fatigue_system/
├── config.py                    # All tunable parameters
├── main.py                      # Orchestrator (11-stage pipeline)
├── requirements.txt
├── test_smoke.py                # 30+ unit tests
│
├── calibration/
│   └── calibration_engine.py    # 90s calibration + JSON profiles
│
├── preprocessing/
│   └── image_enhancer.py        # CLAHE, gamma, glare suppression
│
├── landmarks/
│   └── face_mesh.py             # MediaPipe Face Mesh wrapper
│
├── features/
│   ├── eye_features.py          # EAR, PERCLOS, blink dur/var
│   ├── mouth_features.py        # MAR, MOI, yawn detection
│   ├── head_features.py         # PnP head pose + deviation
│   ├── gaze_features.py         # Gaze drift, micro-saccade var
│   └── neck_features.py         # Neck vector stability
│
├── models/
│   ├── fatigue_model.py         # Dual-axis PF + CF → FI
│   └── confidence.py            # Composite confidence scorer
│
├── recovery/
│   └── recovery_tracker.py      # Exponential decay recovery
│
├── alerts/
│   └── alert_manager.py         # 3-level escalation + GPIO
│
├── visualization/
│   └── overlay_renderer.py      # Landmarks, dashboard, alerts
│
├── logging_system/
│   └── csv_logger.py            # Daily CSV + Python logging
│
└── utils/
    ├── math_utils.py            # EAR, MAR, gaze, neck, EMA
    ├── smoothing.py             # SlidingWindow, Kalman, EMA
    └── fps_counter.py           # Rolling FPS counter
```

## Installation

```bash
# Python 3.10+ required
cd driver_fatigue_system
pip install -r requirements.txt
```

## Usage

```bash
# First run — performs 90s calibration
python main.py

# Force re-calibration
python main.py --calibrate

# Load specific profile
python main.py --profile my_profile.json
```

### Controls
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset and re-calibrate |

## Calibration

The system requires a **90-second calibration session** before monitoring begins:

1. **Position**: Sit naturally, facing the camera
2. **Gaze**: Look straight ahead
3. **Expression**: Keep a neutral expression
4. **Blink**: Blink naturally

The calibration establishes personalised baselines:
- **EAR₀** — baseline Eye Aspect Ratio
- **MAR₀** — baseline Mouth Aspect Ratio
- **BF₀** — baseline Blink Frequency (bpm)
- **H₀** — baseline Head Pose (pitch, yaw, roll)
- **G₀** — baseline Gaze Vector
- **Neck₀** — baseline Neck Vector

Baselines are saved to `calibration_profile.json` and auto-loaded on subsequent runs.

## Mathematical Model

### Dual-Axis Fatigue

```
Physical Fatigue:
  PF = w₁·PERCLOS + w₂·BlinkDuration + w₃·HeadDeviation + w₄·MouthIndex

Cognitive Fatigue:
  CF = v₁·GazeDrift + v₂·BlinkVariance + v₃·MicroSaccadeVariance

Fatigue Index:
  FI = α·PF + β·CF

EMA Smoothing:
  FI_smooth = λ·FI + (1−λ)·FI_prev
```

### Recovery Dynamics

```
When all recovery conditions are met:
  FI_t = FI_prev · exp(−k · Δt)

Conditions:
  - EAR_relative > 0.95  (eyes open)
  - Head deviation < 0.15 (upright)
  - |BPM − BF₀| / BF₀ < 0.3 (normal blinking)
```

### Confidence Score

```
Confidence = w_lm · LandmarkConf + w_light · LightScore + w_blur · BlurScore
```

## Testing

```bash
python test_smoke.py
```

Covers: math utils, smoothing, fatigue model, confidence, recovery, calibration serialisation, preprocessing, alerts, and normalisation.

## Deployment Notes

### Jetson Nano / Raspberry Pi

1. Install OpenCV with CUDA (Jetson) or headless (Pi):
   ```bash
   pip install opencv-python-headless
   ```
2. Enable GPIO alerts in `config.py`:
   ```python
   GPIO_ALERT_ENABLED = True
   GPIO_PIN_ALERT = 18
   ```
3. Reduce resolution for performance:
   ```python
   FRAME_WIDTH = 640
   FRAME_HEIGHT = 480
   ```
4. Use IR camera by setting `IR_SUPPORT_MODE = True`

### Performance Tuning

- `CAPTURE_THREAD_ENABLED = True` — threaded capture (default)
- `FACE_MESH_REFINE_LANDMARKS = False` — disable iris (faster, no gaze)
- Reduce `SMOOTHING_WINDOW_*` values for lower latency
- Increase `CSV_LOG_INTERVAL` to reduce I/O overhead

## License

This project is for educational and research purposes.
