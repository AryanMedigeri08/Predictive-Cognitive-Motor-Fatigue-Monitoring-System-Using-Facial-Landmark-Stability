# Predictive Cognitive-Motor Fatigue Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **Real-time driver fatigue & cognitive degradation monitoring system** using facial landmark stability analysis, dual-axis fatigue modeling, and personalized calibration.

A production-ready, edge-deployable solution for detecting driver fatigue through eye closure, gaze drift, head pose deviation, and mouth activity with adaptive recovery dynamics.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Personalized Calibration** | Adaptive 90s baseline establishment (EAR₀, MAR₀, BF₀, H₀, G₀, Neck₀) |
| **Dual-Axis Fatigue Model** | Physical Fatigue (PF) + Cognitive Fatigue (CF) → Composite Fatigue Index |
| **PERCLOS Metric** | 60-second sliding-window eye closure percentage analysis |
| **Gaze & Micro-Saccade Detection** | Iris-based gaze tracking for cognitive load assessment |
| **Recovery Dynamics** | Exponential decay model: FI_t = FI_prev · exp(−k · Δt) |
| **Night Mode** | Auto-activated CLAHE + gamma correction in low-light conditions |
| **Multi-Level Alerts** | Visual → Audio → Critical alarm with confidence gating |
| **Confidence Scoring** | Composite metric (landmarks + lighting + image quality) |
| **CSV Logging** | Daily-rotating logs with all 25 features per frame |
| **Optimized Pipeline** | Threaded architecture achieving ≥20 FPS real-time performance |
| **Robust Detection** | Glasses/glare handling with intelligent fallback mechanisms |

---

## Tech Stack

### Core Technologies
- **Language**: Python 3.10+
- **Computer Vision**: OpenCV 4.8+, MediaPipe 0.10+
- **Numerical Computing**: NumPy, SciPy
- **Visualization**: Matplotlib
- **Concurrency**: Threading (stdlib)
- **Data Logging**: CSV (stdlib), Python logging

### Key Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **opencv-python** | ≥4.8.0 | Image processing, camera capture, visualization |
| **mediapipe** | ≥0.10.0 | Facial landmarks, face mesh detection |
| **numpy** | ≥1.24.0 | Numerical operations, array handling |
| **scipy** | ≥1.10.0 | Signal filtering, smoothing algorithms |
| **matplotlib** | ≥3.7.0 | Data visualization, plotting |

---

## Project Structure

```
Predictive-Cognitive-Motor-Fatigue-Monitoring-System/
├── config.py                           # Configuration & tunable parameters
├── main.py                             # Main orchestrator (11-stage pipeline)
├── requirements.txt                    # Python dependencies
├── test_smoke.py                       # Unit tests & validation
├── calibration_profile.json            # Stored calibration profiles
├── fatigue_log.csv                     # Fatigue monitoring logs
│
├── calibration/                        # Calibration engine
│   ├── __init__.py
│   └── calibration_engine.py           # 90s personalized calibration
│
├── preprocessing/                      # Image preprocessing
│   ├── __init__.py
│   └── image_enhancer.py               # CLAHE, gamma, glare handling
│
├── landmarks/                          # Facial landmark detection
│   ├── __init__.py
│   └── face_mesh.py                    # MediaPipe Face Mesh wrapper
│
├── features/                           # Feature extraction modules
│   ├── __init__.py
│   ├── eye_features.py                 # EAR, PERCLOS, blink dynamics
│   ├── mouth_features.py               # MAR, MOI, yawn detection
│   ├── head_features.py                # Pose + deviation tracking
│   ├── gaze_features.py                # Gaze drift, micro-saccades
│   └── neck_features.py                # Neck stability metrics
│
├── models/                             # Fatigue modeling
│   ├── __init__.py
│   ├── fatigue_model.py                # Dual-axis PF + CF → FI
│   └── confidence.py                   # Composite confidence scorer
│
├── recovery/                           # Recovery tracking
│   ├── __init__.py
│   └── recovery_tracker.py             # Exponential decay dynamics
│
├── alerts/                             # Alert management
│   ├── __init__.py
│   └── alert_manager.py                # 3-level escalation system
│
├── visualization/                      # Real-time visualization
│   ├── __init__.py
│   └── overlay_renderer.py             # Landmarks, dashboard, alerts
│
├── logging_system/                     # Data logging
│   ├── __init__.py
│   └── csv_logger.py                   # Daily CSV rotation
│
├── utils/                              # Utility functions
│   ├── __init__.py
│   ├── math_utils.py                   # Mathematical operations
│   ├── smoothing.py                    # Filtering algorithms
│   └── fps_counter.py                  # Performance monitoring
│
├── vision/                             # Computer vision modules
│   ├── __init__.py
│   ├── eye_metrics.py
│   ├── face_mesh.py
│   ├── head_pose.py
│   ├── mouth_metrics.py
│   └── stability.py
│
└── logs/                               # Log storage
    └── fatigue_log_*.csv               # Daily log files
```

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Webcam or video input device
- ~500 MB disk space for dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Predictive-Cognitive-Motor-Fatigue-Monitoring-System.git
cd Predictive-Cognitive-Motor-Fatigue-Monitoring-System

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# First run — performs 90-second calibration
python main.py

# Force re-calibration
python main.py --calibrate

# Load specific calibration profile
python main.py --profile my_profile.json

# Run tests
python test_smoke.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset & recalibrate |
| `s` | Toggle statistics display |
| `a` | Test audio alert |

---

## Configuration

Edit `config.py` to customize:

```python
# Fatigue thresholds
FATIGUE_THRESHOLD_WARNING = 0.6
FATIGUE_THRESHOLD_CRITICAL = 0.8

# Alert settings
ALERT_AUDIO_ENABLED = True
ALERT_VISUAL_ENABLED = True

# Calibration
CALIBRATION_DURATION_SECONDS = 90

# Recovery parameters
RECOVERY_HALF_LIFE = 30  # seconds

# CSV logging
LOG_FEATURES = True
LOG_ROTATION_HOUR = 0  # Daily rotation at midnight
```

---

## System Architecture

### 11-Stage Pipeline

1. **Frame Capture** — Camera input (threaded)
2. **Preprocessing** — CLAHE, gamma correction, glare suppression
3. **Face Detection** — MediaPipe Face Mesh
4. **Feature Extraction** — Eye, mouth, head, gaze metrics
5. **Normalization** — Baseline-relative feature scaling
6. **Confidence Scoring** — Composite quality metric
7. **Fatigue Calculation** — Dual-axis PF + CF fusion
8. **Recovery Application** — Exponential decay
9. **Alert Logic** — 3-level escalation
10. **Logging** — CSV + system logs
11. **Visualization** — Real-time overlay rendering

### Calibration Process

The **90-second calibration** establishes personal baselines:
- **EAR₀** (Eye Aspect Ratio) — neutral eye open state
- **MAR₀** (Mouth Aspect Ratio) — resting mouth position
- **BF₀** (Baseline Frequency) — normal blink rate
- **H₀** (Head Pose) — neutral head orientation
- **G₀** (Gaze Point) — forward gaze baseline
- **Neck₀** (Neck Stability) — head movement baseline

---

## Output & Logging

### CSV Log Format

Each row contains:
- **Timestamp** — Frame capture time
- **Facial Features** — EAR, MAR, blink metrics, gaze coordinates
- **Head Pose** — Pitch, yaw, roll angles
- **Fatigue Metrics** — PF, CF, composite FI
- **Alerts** — Alert level & type
- **Confidence Scores** — Overall system confidence

### Daily Log Rotation

Logs are automatically rotated daily:
```
logs/
├── fatigue_log_2026-02-19.csv
├── fatigue_log_2026-02-20.csv
└── fatigue_log_2026-02-21.csv
```

---

## Calibration Profile Format

Example `calibration_profile.json`:

```json
{
  "timestamp": "2026-02-19T10:30:00",
  "user_id": "driver_001",
  "ear_baseline": 0.385,
  "mar_baseline": 0.215,
  "blink_frequency_baseline": 0.18,
  "head_pose_baseline": [0.0, 0.0, 0.0],
  "gaze_baseline": [0.5, 0.5],
  "neck_stability_baseline": 0.92,
  "confidence_threshold": 0.75
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- **Free to use** for personal and commercial projects
- **Modify and distribute** with attribution
- **Use in proprietary software** (with license notice)
- **No liability** — provided as-is

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Discussions for feature requests

---

## Acknowledgments

- **MediaPipe** — Advanced facial landmark detection
- **OpenCV** — Computer vision processing
- **NumPy/SciPy** — Numerical computing
- Research in driver fatigue detection and human factors

---

## ⚠️ Disclaimer

This system is designed as a supplementary monitoring tool and should **not be the sole basis** for safety-critical decisions. Always follow local regulations and driver safety guidelines.

---

**Last Updated:** February 2026  
**Version:** 1.0.0

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
>>>>>>> 265ab35 (Initial Commit)
