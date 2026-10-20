import unittest

import cv2
import numpy as np

from presence_guard.badge import analyze_badge_roi, compute_badge_roi, decide
from presence_guard.config import Config
from presence_guard.types import BadgeDecision, Box


def synthetic_frame(blue=True, brightness=45):
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    cv2.rectangle(frame, (200, 120), (440, 479), (70, 70, 75), -1)
    cv2.ellipse(frame, (320, 105), (48, 60), 0, 0, 360, (150, 180, 215), -1)
    color = (110, 45, 20) if blue else (30, 30, 200)
    cv2.rectangle(frame, (300, 215), (345, 265), color, -1)
    cv2.rectangle(frame, (312, 152), (318, 210), color, -1)
    return frame, Box(180, 40, 460, 479), Box(275, 50, 365, 160)


class BadgeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.badge.min_valid_frames = 3
        self.cfg.badge.present_frame_votes = 2

    def test_roi_is_below_chin(self):
        frame, person, face = synthetic_frame()
        roi = compute_badge_roi(frame.shape[:2], person, face, self.cfg.badge)
        self.assertIsNotNone(roi)
        self.assertLessEqual(roi.y1, face.y2)
        self.assertGreater(roi.y2, face.y2)
        self.assertGreater(roi.height, face.height)
        self.assertGreaterEqual(roi.x1, 200)
        self.assertLessEqual(roi.x2, 440)

    def test_roi_requires_face(self):
        frame, person, _ = synthetic_frame()
        self.assertIsNone(compute_badge_roi(frame.shape[:2], person, None,
                                            self.cfg.badge))

    def test_roi_rejects_tiny_face(self):
        frame, person, face = synthetic_frame()
        tiny = Box(300, 95, 315, 110)
        self.assertIsNone(compute_badge_roi(frame.shape[:2], person, tiny,
                                            self.cfg.badge))

    def test_dark_blue_detected(self):
        frame, person, face = synthetic_frame(blue=True)
        roi = compute_badge_roi(frame.shape[:2], person, face, self.cfg.badge)
        stats = analyze_badge_roi(frame, roi, face, self.cfg.badge)
        self.assertTrue(stats.valid)
        self.assertTrue(stats.present)
        self.assertGreater(stats.largest_area, self.cfg.badge.min_blue_pixels)

    def test_red_is_absent(self):
        frame, person, face = synthetic_frame(blue=False)
        roi = compute_badge_roi(frame.shape[:2], person, face, self.cfg.badge)
        stats = analyze_badge_roi(frame, roi, face, self.cfg.badge)
        self.assertTrue(stats.valid)
        self.assertFalse(stats.present)
        result = decide([stats, stats, stats], self.cfg.badge)
        self.assertEqual(result.decision, BadgeDecision.ABSENT)

    def test_too_dark_is_unknown(self):
        frame, person, face = synthetic_frame(blue=True, brightness=2)
        self.cfg.badge.min_brightness = 200.0
        roi = compute_badge_roi(frame.shape[:2], person, face, self.cfg.badge)
        stats = analyze_badge_roi(frame, roi, face, self.cfg.badge)
        self.assertFalse(stats.valid)
        self.assertEqual(stats.reason, "too_dark")
        result = decide([stats, stats, stats], self.cfg.badge)
        self.assertEqual(result.decision, BadgeDecision.UNKNOWN)

    def test_one_ambiguous_frame_is_unknown(self):
        frame_a, roi_a, face_a = self._stats_for(blue=False)
        absent = analyze_badge_roi(frame_a, roi_a, face_a, self.cfg.badge)
        frame_p, roi_p, face_p = self._stats_for(blue=True)
        present = analyze_badge_roi(frame_p, roi_p, face_p, self.cfg.badge)
        result = decide([present, absent, absent], self.cfg.badge)
        self.assertEqual(result.decision, BadgeDecision.UNKNOWN)

    def test_not_enough_valid_frames_is_unknown(self):
        result = decide([], self.cfg.badge)
        self.assertEqual(result.decision, BadgeDecision.UNKNOWN)

    def _stats_for(self, blue):
        frame, person, face = synthetic_frame(blue=blue)
        roi = compute_badge_roi(frame.shape[:2], person, face, self.cfg.badge)
        return frame, roi, face


if __name__ == "__main__":
    unittest.main()
