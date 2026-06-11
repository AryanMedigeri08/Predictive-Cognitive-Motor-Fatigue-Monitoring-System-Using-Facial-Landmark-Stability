"""
test_smoke.py
=============
Comprehensive smoke-test suite for the Driver Fatigue Monitoring System.

Tests core mathematical utilities, feature extractors, model logic,
calibration, recovery, and confidence scoring — all without requiring
a webcam or MediaPipe runtime.

Run:
    python test_smoke.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────
# 1. Math Utilities
# ──────────────────────────────────────────────────────────────────────

class TestMathUtils(unittest.TestCase):
    """Test utils/math_utils.py."""

    def test_euclidean_distance_2d(self):
        from utils.math_utils import euclidean_distance
        d = euclidean_distance(np.array([0, 0]), np.array([3, 4]))
        self.assertAlmostEqual(d, 5.0, places=6)

    def test_euclidean_distance_3d(self):
        from utils.math_utils import euclidean_distance
        d = euclidean_distance(np.array([1, 2, 3]), np.array([4, 6, 3]))
        self.assertAlmostEqual(d, 5.0, places=6)

    def test_normalize_vector(self):
        from utils.math_utils import normalize_vector
        v = normalize_vector(np.array([3, 4]))
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=6)

    def test_normalize_zero_vector(self):
        from utils.math_utils import normalize_vector
        v = normalize_vector(np.array([0, 0]))
        self.assertEqual(np.linalg.norm(v), 0.0)

    def test_clamp(self):
        from utils.math_utils import clamp
        self.assertEqual(clamp(1.5, 0.0, 1.0), 1.0)
        self.assertEqual(clamp(-0.5, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)

    def test_compute_ear_open(self):
        """Open eye should have EAR ~ 0.3."""
        from utils.math_utils import compute_ear
        # Construct a roughly open eye
        eye = np.array([
            [0, 0],     # p1 outer
            [1, 1],     # p2 upper-outer
            [2, 1],     # p3 upper-inner
            [3, 0],     # p4 inner
            [2, -1],    # p5 lower-inner
            [1, -1],    # p6 lower-outer
        ], dtype=np.float64)
        ear = compute_ear(eye)
        self.assertGreater(ear, 0.2)

    def test_compute_ear_closed(self):
        """Closed eye: all points on the horizontal → EAR ≈ 0."""
        from utils.math_utils import compute_ear
        eye = np.array([
            [0, 0], [1, 0], [2, 0],
            [3, 0], [2, 0], [1, 0],
        ], dtype=np.float64)
        ear = compute_ear(eye)
        self.assertAlmostEqual(ear, 0.0, places=4)

    def test_compute_mar(self):
        from utils.math_utils import compute_mar
        pts = np.array([
            [50, 40],    # top lip
            [50, 60],    # bottom lip
            [30, 50],    # left corner
            [70, 50],    # right corner
        ], dtype=np.float64)
        mar = compute_mar(pts)
        self.assertAlmostEqual(mar, 20.0 / 40.0, places=4)

    def test_ema_update(self):
        from utils.math_utils import ema_update
        prev = 0.0
        val = ema_update(prev, 1.0, 0.5)
        self.assertAlmostEqual(val, 0.5)
        val2 = ema_update(val, 1.0, 0.5)
        self.assertAlmostEqual(val2, 0.75)

    def test_compute_gaze_vector(self):
        from utils.math_utils import compute_gaze_vector
        iris = np.array([52, 50])
        inner = np.array([40, 50])
        outer = np.array([60, 50])
        gaze = compute_gaze_vector(iris, inner, outer)
        self.assertGreater(gaze[0], 0)  # iris is right of midpoint

    def test_angular_deviation(self):
        from utils.math_utils import angular_deviation
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        angle = angular_deviation(v1, v2)
        self.assertAlmostEqual(angle, 90.0, places=1)


# ──────────────────────────────────────────────────────────────────────
# 2. Smoothing
# ──────────────────────────────────────────────────────────────────────

class TestSmoothing(unittest.TestCase):
    """Test utils/smoothing.py."""

    def test_sliding_window(self):
        from utils.smoothing import SlidingWindowSmoother
        s = SlidingWindowSmoother(3)
        self.assertAlmostEqual(s.update(1.0), 1.0)
        self.assertAlmostEqual(s.update(2.0), 1.5)
        self.assertAlmostEqual(s.update(3.0), 2.0)
        self.assertAlmostEqual(s.update(4.0), 3.0)  # window drops 1.0

    def test_kalman_convergence(self):
        from utils.smoothing import KalmanFilter1D
        kf = KalmanFilter1D()
        for _ in range(100):
            val = kf.update(5.0)
        self.assertAlmostEqual(val, 5.0, places=2)

    def test_ema_smoother(self):
        from utils.smoothing import ExponentialMovingAverage
        ema = ExponentialMovingAverage(lam=0.5)
        ema.update(1.0)
        v = ema.update(1.0)
        self.assertAlmostEqual(v, 1.0)  # two 1.0s → EMA = 1.0


# ──────────────────────────────────────────────────────────────────────
# 3. Fatigue Model
# ──────────────────────────────────────────────────────────────────────

class TestFatigueModel(unittest.TestCase):
    """Test models/fatigue_model.py."""

    def _model(self):
        from models.fatigue_model import DualAxisFatigueModel
        return DualAxisFatigueModel()

    def test_zero_input_normal(self):
        from models.fatigue_model import FatigueInput, FatigueLevel
        model = self._model()
        result = model.compute(FatigueInput())
        self.assertEqual(result.level, FatigueLevel.NORMAL)
        self.assertAlmostEqual(result.fatigue_index, 0.0, places=4)

    def test_max_input_severe(self):
        from models.fatigue_model import FatigueInput, FatigueLevel
        model = self._model()
        inp = FatigueInput(
            perclos=1.0, blink_duration_norm=1.0,
            head_deviation=1.0, mouth_opening_index=1.0,
            gaze_drift=1.0, blink_variance_norm=1.0,
            microsaccade_variance=1.0,
        )
        # Pump through EMA to let it converge
        for _ in range(100):
            result = model.compute(inp)
        self.assertEqual(result.level, FatigueLevel.SEVERE)
        self.assertGreater(result.fatigue_index_smooth, 0.9)

    def test_moderate_threshold(self):
        from models.fatigue_model import FatigueInput, FatigueLevel
        model = self._model()
        inp = FatigueInput(perclos=0.5, head_deviation=0.4)
        for _ in range(100):
            result = model.compute(inp)
        self.assertEqual(result.level, FatigueLevel.MODERATE)

    def test_pf_cf_breakdown(self):
        from models.fatigue_model import FatigueInput
        model = self._model()
        inp = FatigueInput(perclos=0.5, gaze_drift=0.8)
        result = model.compute(inp)
        self.assertGreater(result.physical_fatigue, 0)
        self.assertGreater(result.cognitive_fatigue, 0)


# ──────────────────────────────────────────────────────────────────────
# 4. Confidence Scoring
# ──────────────────────────────────────────────────────────────────────

class TestConfidence(unittest.TestCase):
    """Test models/confidence.py."""

    def test_high_confidence(self):
        from models.confidence import ConfidenceScorer
        scorer = ConfidenceScorer()
        result = scorer.compute(
            landmark_confidence=1.0,
            lighting_score=1.0,
            blur_laplacian_var=200.0,  # sharp
        )
        self.assertGreater(result.composite, 0.8)
        self.assertFalse(result.is_low)

    def test_low_confidence(self):
        from models.confidence import ConfidenceScorer
        scorer = ConfidenceScorer()
        result = scorer.compute(
            landmark_confidence=0.1,
            lighting_score=0.1,
            blur_laplacian_var=5.0,   # very blurry
        )
        self.assertLess(result.composite, 0.5)
        self.assertTrue(result.is_low)


# ──────────────────────────────────────────────────────────────────────
# 5. Recovery Dynamics
# ──────────────────────────────────────────────────────────────────────

class TestRecovery(unittest.TestCase):
    """Test recovery/recovery_tracker.py."""

    def test_decay_when_recovering(self):
        from recovery.recovery_tracker import RecoveryTracker
        tracker = RecoveryTracker(decay_k=1.0)  # fast decay for testing
        state1 = tracker.update(
            fi_smooth=0.8,
            ear_relative=1.0,
            head_deviation=0.05,
            bpm=15.0,
            bf_0=15.0,
        )
        self.assertTrue(state1.is_recovering)

        time.sleep(0.1)
        state2 = tracker.update(
            fi_smooth=0.8,
            ear_relative=1.0,
            head_deviation=0.05,
            bpm=15.0,
            bf_0=15.0,
        )
        self.assertTrue(state2.is_recovering)
        self.assertLess(state2.fi_decayed, 0.8)

    def test_no_decay_when_eyes_closed(self):
        from recovery.recovery_tracker import RecoveryTracker
        tracker = RecoveryTracker()
        state = tracker.update(
            fi_smooth=0.8,
            ear_relative=0.5,    # eyes mostly closed → NOT recovering
            head_deviation=0.05,
            bpm=15.0,
            bf_0=15.0,
        )
        self.assertFalse(state.is_recovering)
        self.assertAlmostEqual(state.fi_decayed, 0.8)


# ──────────────────────────────────────────────────────────────────────
# 6. Calibration (engine logic)
# ──────────────────────────────────────────────────────────────────────

class TestCalibration(unittest.TestCase):
    """Test calibration/calibration_engine.py."""

    def test_robust_mean(self):
        from calibration.calibration_engine import CalibrationEngine
        # Add outliers — they should be trimmed by 10/90 percentile
        samples = [0.28] * 80 + [0.01] * 10 + [0.99] * 10
        mean = CalibrationEngine._robust_mean(samples)
        self.assertAlmostEqual(mean, 0.28, places=1)

    def test_profile_serialisation(self):
        from calibration.calibration_engine import CalibrationEngine
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name

        try:
            engine = CalibrationEngine(duration_sec=0.1, json_path=tmp)
            engine.start()
            for _ in range(10):
                engine.add_sample(
                    ear=0.30, mar=0.4,
                    pitch=5.0, yaw=-2.0, roll=1.0,
                    gaze_vec=np.array([0.1, -0.9]),
                    neck_vec=np.array([0.0, -1.0]),
                )
            time.sleep(0.15)
            profile = engine.finalise()
            engine.save_profile()

            engine2 = CalibrationEngine(json_path=tmp)
            loaded = engine2.load_profile()
            self.assertAlmostEqual(loaded.ear_0, profile.ear_0, places=3)
            self.assertAlmostEqual(loaded.mar_0, profile.mar_0, places=3)
        finally:
            os.unlink(tmp)


# ──────────────────────────────────────────────────────────────────────
# 7. Normalisation Helpers
# ──────────────────────────────────────────────────────────────────────

class TestNormalisationHelpers(unittest.TestCase):
    """Test normalisation functions from main.py."""

    def test_blink_duration_norm(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from main import normalise_blink_duration
        self.assertAlmostEqual(normalise_blink_duration(250, 500), 0.5)
        self.assertAlmostEqual(normalise_blink_duration(600, 500), 1.0)

    def test_mouth_index_norm(self):
        from main import normalise_mouth_index
        self.assertAlmostEqual(normalise_mouth_index(1.0), 0.0)
        self.assertAlmostEqual(normalise_mouth_index(2.0), 0.5)
        self.assertAlmostEqual(normalise_mouth_index(3.0), 1.0)


# ──────────────────────────────────────────────────────────────────────
# 8. Preprocessing (unit-testable parts)
# ──────────────────────────────────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing/image_enhancer.py."""

    def _make_frame(self, brightness: int = 128) -> np.ndarray:
        return np.full((100, 100, 3), brightness, dtype=np.uint8)

    def test_detect_low_light(self):
        from preprocessing.image_enhancer import ImageEnhancer
        enh = ImageEnhancer(luminance_threshold=60)
        self.assertTrue(enh.detect_low_light(self._make_frame(30)))
        self.assertFalse(enh.detect_low_light(self._make_frame(128)))

    def test_enhance_returns_info(self):
        from preprocessing.image_enhancer import ImageEnhancer
        enh = ImageEnhancer()
        enhanced, info = enh.enhance(self._make_frame(128))
        self.assertEqual(enhanced.shape, (100, 100, 3))
        self.assertFalse(info.is_night_mode)
        self.assertGreater(info.lighting_score, 0)

    def test_night_mode_activated(self):
        from preprocessing.image_enhancer import ImageEnhancer
        enh = ImageEnhancer(luminance_threshold=60)
        _, info = enh.enhance(self._make_frame(30))
        self.assertTrue(info.is_night_mode)


# ──────────────────────────────────────────────────────────────────────
# 9. Alert Manager
# ──────────────────────────────────────────────────────────────────────

class TestAlertManager(unittest.TestCase):
    """Test alerts/alert_manager.py."""

    def test_no_alert_normal(self):
        from alerts.alert_manager import AlertManager, AlertLevel
        mgr = AlertManager()
        state = mgr.update(
            fatigue_level=__import__("models.fatigue_model",
                                     fromlist=["FatigueLevel"]).FatigueLevel.NORMAL,
            confidence=0.9,
        )
        self.assertEqual(state.level, AlertLevel.NONE)

    def test_suppressed_by_low_confidence(self):
        from alerts.alert_manager import AlertManager, AlertLevel
        from models.fatigue_model import FatigueLevel
        mgr = AlertManager(confidence_threshold=0.5)
        state = mgr.update(fatigue_level=FatigueLevel.SEVERE, confidence=0.3)
        self.assertTrue(state.suppressed)
        self.assertEqual(state.level, AlertLevel.NONE)


if __name__ == "__main__":
    print("=" * 60)
    print("  Driver Fatigue Monitoring System — Smoke Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
