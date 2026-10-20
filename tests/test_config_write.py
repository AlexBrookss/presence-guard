import tempfile
import unittest
from pathlib import Path

from presence_guard.config import set_config_value


class SetConfigValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.toml"

    def test_create_missing_file(self):
        self.assertTrue(set_config_value(self.path, "camera", "device", "/dev/video1"))
        text = self.path.read_text()
        self.assertIn('[camera]', text)
        self.assertIn('device = "/dev/video1"', text)

    def test_update_existing_key(self):
        self.path.write_text('[camera]\ndevice = "/dev/video0"\nfps = 10\n')
        self.assertTrue(set_config_value(self.path, "camera", "device", "/dev/video2"))
        self.assertIn('device = "/dev/video2"', self.path.read_text())
        self.assertIn('fps = 10', self.path.read_text())

    def test_insert_new_key_into_section(self):
        self.path.write_text('[camera]\nfps = 10\n\n[scheduler]\npresence_interval_seconds = 60\n')
        self.assertTrue(set_config_value(self.path, "camera", "device", "0"))
        text = self.path.read_text()
        self.assertIn('device = "0"', text)
        # scheduler section must remain intact and after camera's fps
        self.assertLess(text.index('device = "0"'), text.index('[scheduler]'))

    def test_bool_and_int(self):
        set_config_value(self.path, "scheduler", "lock_on_holidays", True)
        set_config_value(self.path, "scheduler", "presence_interval_seconds", 30)
        text = self.path.read_text()
        self.assertIn('lock_on_holidays = true', text)
        self.assertIn('presence_interval_seconds = 30', text)


if __name__ == "__main__":
    unittest.main()
