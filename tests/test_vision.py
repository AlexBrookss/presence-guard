import unittest

import numpy as np

from presence_guard.config import Config
from presence_guard.geometry import letterbox, unletterbox_box
from presence_guard.types import Box, Detection
from presence_guard.vision import (frame_is_usable, select_face,
                                   select_primary_person)


def face_row(x, y, w, h, score=0.9):
    row = np.zeros(15, dtype=np.float32)
    row[0:4] = (x, y, w, h)
    row[-1] = score
    return row


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.detector.person_confidence = 0.3
        self.cfg.detector.min_person_height_ratio = 0.18
        self.cfg.detector.center_zone_ratio = 0.9

    def test_prefers_central_large_person(self):
        shape = (480, 640)
        dets = [
            Detection(Box(0, 80, 40, 400), 0.9),        # off-centre
            Detection(Box(260, 120, 400, 470), 0.7),    # central
            Detection(Box(600, 300, 620, 340), 0.8),    # too small
        ]
        primary = select_primary_person(dets, shape, self.cfg)
        self.assertIsNotNone(primary)
        self.assertEqual(primary.box.x1, 260)

    def test_ignores_non_person_classes(self):
        shape = (480, 640)
        dets = [Detection(Box(200, 100, 440, 470), 0.99, class_id=56)]
        self.assertIsNone(select_primary_person(dets, shape, self.cfg))

    def test_select_face_inside_person(self):
        person = Box(180, 40, 460, 479)
        faces = np.stack([
            face_row(600, 50, 30, 30, 0.99),   # outside person
            face_row(300, 60, 80, 90, 0.95),   # good
            face_row(280, 80, 30, 30, 0.99),   # smaller
        ])
        face = select_face(faces, (480, 640), person, self.cfg,
                           self.cfg.badge.min_face_width_ratio)
        self.assertIsNotNone(face)
        self.assertGreater(face.width, 70)

    def test_select_face_rejects_low_score(self):
        faces = np.stack([face_row(300, 60, 80, 90, 0.1)])
        face = select_face(faces, (480, 640), None, self.cfg, 0.05)
        self.assertIsNone(face)

    def test_black_frame_is_rejected(self):
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertFalse(frame_is_usable(black, self.cfg))
        gray = np.full((240, 320, 3), 40, dtype=np.uint8)
        self.assertTrue(frame_is_usable(gray, self.cfg))

    def test_letterbox_roundtrip(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, pad = letterbox(image, (416, 416))
        self.assertEqual(padded.shape, (416, 416, 3))
        original = Box(100, 50, 300, 400)
        pad_x, pad_y = pad
        mapped_input = Box(
            original.x1 * scale + pad_x,
            original.y1 * scale + pad_y,
            original.x2 * scale + pad_x,
            original.y2 * scale + pad_y,
        )
        mapped = unletterbox_box(mapped_input, scale, pad, 640, 480)
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped.x1, original.x1, delta=1.5)
        self.assertAlmostEqual(mapped.y1, original.y1, delta=1.5)
        self.assertAlmostEqual(mapped.x2, original.x2, delta=1.5)
        self.assertAlmostEqual(mapped.y2, original.y2, delta=1.5)


if __name__ == "__main__":
    unittest.main()
