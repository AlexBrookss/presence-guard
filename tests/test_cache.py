import os
import tempfile
import time
import unittest
from pathlib import Path

from presence_guard.cache import cleanup_dir


class CacheCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _make(self, name: str, age_days: float = 0.0) -> Path:
        path = self.root / name
        path.write_bytes(b"x" * 100)
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
        return path

    def test_keep_newest_files(self):
        for i in range(10):
            self._make(f"check-{i}.jpg", age_days=10 - i)
        removed, freed = cleanup_dir(self.root, keep=3, max_age_days=0)
        self.assertEqual(removed, 7)
        self.assertEqual(freed, 700)
        self.assertEqual(len(list(self.root.glob("*.jpg"))), 3)

    def test_remove_old_files(self):
        self._make("old.jpg", age_days=10)
        self._make("new.jpg", age_days=0)
        removed, _ = cleanup_dir(self.root, keep=0, max_age_days=7)
        self.assertEqual(removed, 1)
        self.assertTrue((self.root / "new.jpg").exists())

    def test_ignores_non_images(self):
        (self.root / "report.json").write_text("{}")
        removed, _ = cleanup_dir(self.root, keep=0, max_age_days=0)
        self.assertEqual(removed, 0)
        self.assertTrue((self.root / "report.json").exists())

    def test_missing_dir(self):
        self.assertEqual(cleanup_dir(self.root / "nope"), (0, 0))


if __name__ == "__main__":
    unittest.main()
