"""
Automated Test Suite for Driver Drowsiness Detection System
-----------------------------------------------------------
Tests mathematical correctness, state machine transitions,
missing face handling, audio alert non-blocking behavior,
and model verification.
"""

import unittest
import time
import os
import numpy as np

from drowsiness_detector import (
    calculate_ear,
    calculate_euclidean_distance,
    DrowsinessDetector
)
from audio_alert import AudioAlert
from download_model import ensure_model_exists, MODEL_PATH


class TestEarMathematics(unittest.TestCase):
    """Verifies Eye Aspect Ratio (EAR) calculations."""

    def test_euclidean_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        dist = calculate_euclidean_distance(p1, p2)
        self.assertAlmostEqual(dist, 5.0, places=4)

    def test_open_eye_ear(self):
        # Synthetic open eye points: [p1, p2, p3, p4, p5, p6]
        # Horizontal width = 40 (p1 to p4)
        # Vertical 1 = 16 (p2 to p6)
        # Vertical 2 = 16 (p3 to p5)
        # EAR = (16 + 16) / (2 * 40) = 32 / 80 = 0.40
        open_eye = [
            (10, 50),  # P1: outer corner
            (25, 42),  # P2: top 1
            (35, 42),  # P3: top 2
            (50, 50),  # P4: inner corner
            (35, 58),  # P5: bottom 2
            (25, 58),  # P6: bottom 1
        ]
        ear = calculate_ear(open_eye)
        self.assertAlmostEqual(ear, 0.40, places=3)
        self.assertGreater(ear, 0.25, "Open eye EAR should be above typical 0.24 threshold")

    def test_closed_eye_ear(self):
        # Synthetic closed eye points: vertical distance is very small (2 units)
        # EAR = (2 + 2) / (2 * 40) = 4 / 80 = 0.05
        closed_eye = [
            (10, 50),
            (25, 49),
            (35, 49),
            (50, 50),
            (35, 51),
            (25, 51),
        ]
        ear = calculate_ear(closed_eye)
        self.assertAlmostEqual(ear, 0.05, places=3)
        self.assertLess(ear, 0.20, "Closed eye EAR should be below threshold")

    def test_degenerate_points(self):
        # Collapsed points (width = 0)
        zero_eye = [(10, 10)] * 6
        ear = calculate_ear(zero_eye)
        self.assertEqual(ear, 0.0, "Degenerate eye points should return 0.0 without division by zero")


class TestDrowsinessDetector(unittest.TestCase):
    """Verifies DrowsinessDetector state machine and error handling."""

    @classmethod
    def setUpClass(cls):
        cls.detector = DrowsinessDetector(
            ear_threshold=0.24,
            consecutive_frames_threshold=3,
            time_threshold_seconds=0.3
        )

    def setUp(self):
        self.detector.reset_state()

    def test_none_and_empty_frames(self):
        # Should handle None frame without crashing
        res_none = self.detector.process_frame(None)
        self.assertFalse(res_none["face_detected"])
        self.assertEqual(res_none["system_status"], "Camera Error")

        # Should handle empty ndarray without crashing
        res_empty = self.detector.process_frame(np.array([]))
        self.assertFalse(res_empty["face_detected"])

    def test_blank_black_frame_no_face(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.detector.process_frame(blank)
        self.assertFalse(res["face_detected"])
        self.assertEqual(res["system_status"], "No Face Detected")
        self.assertEqual(res["closed_frames"], 0)
        self.assertFalse(res["is_drowsy"])

    def test_state_reset(self):
        self.detector.closed_frames_count = 10
        self.detector.total_alerts = 5
        self.detector.is_drowsy = True
        self.detector.reset_state()
        self.assertEqual(self.detector.closed_frames_count, 0)
        self.assertEqual(self.detector.total_alerts, 0)
        self.assertFalse(self.detector.is_drowsy)


class TestAudioAlert(unittest.TestCase):
    """Verifies non-blocking audio alerts."""

    def test_audio_non_blocking(self):
        alert = AudioAlert(cooldown_seconds=1.0, enabled=True)
        start = time.time()
        alert.trigger(frequency=1000, duration_ms=200)
        elapsed = time.time() - start
        # The trigger call must return within 50ms (non-blocking)
        self.assertLess(elapsed, 0.05, f"Audio trigger took {elapsed:.4f}s; it should be non-blocking")

    def test_audio_cooldown(self):
        alert = AudioAlert(cooldown_seconds=2.0, enabled=True)
        alert.trigger()
        # Immediately triggering again should be blocked by cooldown
        last_alert = alert._last_alert_time
        alert.trigger()
        self.assertEqual(alert._last_alert_time, last_alert, "Second trigger should be suppressed by cooldown")

    def test_audio_disabled_flag(self):
        alert = AudioAlert(cooldown_seconds=0.1, enabled=False)
        alert.trigger()
        self.assertEqual(alert._last_alert_time, 0.0, "Disabled alert should not trigger")


class TestModelIntegrity(unittest.TestCase):
    """Verifies MediaPipe model file exists and has valid size."""

    def test_model_file_exists(self):
        model_file = ensure_model_exists()
        self.assertTrue(os.path.exists(model_file), "face_landmarker.task must exist")
        size = os.path.getsize(model_file)
        self.assertGreater(size, 1_000_000, f"Model size {size} bytes is unexpectedly small")


if __name__ == "__main__":
    unittest.main(verbosity=2)
