"""Regression tests on real camera frames from the UOS test machine.

Fixtures
--------
``badge_worn.jpg``       navy lanyard + card visible under the chin.
``badge_strap_only.jpg`` lanyard still worn, card removed.
``badge_removed.jpg``    nothing blue near the neck; a bright blue desk fan
                         and a blue object in the hand sit at the right edge
                         of the chest box.

The last two are the important ones: a naive "any blue pixel" rule fires on
the fan and on the bare lanyard.
"""

import os
import unittest

import cv2

from presence_guard.config import Config
from presence_guard.types import BadgeDecision
from presence_guard.vision import VisionPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WORN = os.path.join(DATA_DIR, "badge_worn.jpg")
STRAP_ONLY = os.path.join(DATA_DIR, "badge_strap_only.jpg")
REMOVED = os.path.join(DATA_DIR, "badge_removed.jpg")

FRAMES = (WORN, STRAP_ONLY, REMOVED)
HAVE_FIXTURES = all(os.path.exists(p) for p in FRAMES)


@unittest.skipUnless(HAVE_FIXTURES, "real-camera fixtures are missing")
class RealFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cache = {}

    def _analyse(self, path, mode):
        key = (path, mode)
        if key in self._cache:
            return self._cache[key]
        cfg = Config()
        cfg.badge.mode = mode
        vision = VisionPipeline(cfg)
        frame = cv2.imread(path)
        self.assertIsNotNone(frame, f"cannot read {path}")
        presence = vision.analyze_presence([frame], person_fallback=True)
        self.assertTrue(presence.present, "person must be detected")
        result = vision.analyze_badge([frame] * 5, presence)
        self._cache[key] = result
        return result

    def test_strap_mode_counts_a_lanyard(self):
        self.assertEqual(self._analyse(WORN, "strap").decision,
                         BadgeDecision.PRESENT)
        self.assertEqual(self._analyse(STRAP_ONLY, "strap").decision,
                         BadgeDecision.PRESENT)
        self.assertEqual(self._analyse(REMOVED, "strap").decision,
                         BadgeDecision.ABSENT)

    def test_card_mode_requires_a_card(self):
        self.assertEqual(self._analyse(WORN, "card").decision,
                         BadgeDecision.PRESENT)
        self.assertEqual(self._analyse(STRAP_ONLY, "card").decision,
                         BadgeDecision.ABSENT)
        self.assertEqual(self._analyse(REMOVED, "card").decision,
                         BadgeDecision.ABSENT)

    def test_blue_fan_never_votes(self):
        result = self._analyse(REMOVED, "strap")
        best = max(s.largest_area for s in result.stats)
        self.assertLess(best, Config().badge.min_blue_pixels)


if __name__ == "__main__":
    unittest.main()
