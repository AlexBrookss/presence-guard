import datetime as dt
import unittest

from presence_guard.config import Config
from presence_guard.schedule import BadgeSchedule, parse_window


class BadgeScheduleTests(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        cfg.badge.holiday_mode = "weekdays"
        cfg.badge.workdays = [0, 1, 2, 3, 4]
        cfg.badge.windows = ["09:30-10:00", "13:00-13:30"]
        cfg.badge.interval_seconds = 180
        self.schedule = BadgeSchedule(cfg.badge)
        self.monday = dt.datetime(2026, 10, 19)  # Monday

    def test_parse_window(self):
        window = parse_window("09:30-10:00")
        self.assertEqual(window.start, dt.time(9, 30))
        self.assertEqual(window.end, dt.time(10, 0))
        with self.assertRaises(ValueError):
            parse_window("10:00-09:30")
        with self.assertRaises(ValueError):
            parse_window("9:30")

    def test_is_active(self):
        self.assertFalse(self.schedule.is_active(self.monday.replace(hour=9, minute=29)))
        self.assertTrue(self.schedule.is_active(self.monday.replace(hour=9, minute=30)))
        self.assertTrue(self.schedule.is_active(self.monday.replace(hour=9, minute=59)))
        self.assertFalse(self.schedule.is_active(self.monday.replace(hour=10, minute=0)))
        self.assertTrue(self.schedule.is_active(self.monday.replace(hour=13, minute=0)))
        self.assertTrue(self.schedule.is_active(self.monday.replace(hour=13, minute=29)))
        self.assertFalse(self.schedule.is_active(self.monday.replace(hour=13, minute=30)))
        saturday = dt.datetime(2026, 10, 24, 9, 45)
        self.assertFalse(self.schedule.is_active(saturday))

    def test_next_slot(self):
        at_9 = self.monday.replace(hour=9, minute=0)
        self.assertEqual(self.schedule.next_slot_after(at_9),
                         self.monday.replace(hour=9, minute=30))
        self.assertEqual(self.schedule.next_slot_after(
            self.monday.replace(hour=9, minute=30)),
            self.monday.replace(hour=9, minute=33))
        self.assertEqual(self.schedule.next_slot_after(
            self.monday.replace(hour=9, minute=31)),
            self.monday.replace(hour=9, minute=33))
        # Last slot is 09:57; 09:58 rolls over to the afternoon window.
        self.assertEqual(self.schedule.next_slot_after(
            self.monday.replace(hour=9, minute=58)),
            self.monday.replace(hour=13, minute=0))
        self.assertEqual(self.schedule.next_slot_after(
            self.monday.replace(hour=13, minute=29)),
            self.monday.replace(day=20, hour=9, minute=30))

    def test_next_check_at_is_immediate_inside_window(self):
        now = self.monday.replace(hour=9, minute=31)
        self.assertEqual(self.schedule.next_check_at(now), now)
        self.assertEqual(
            self.schedule.next_check_at(self.monday.replace(hour=9, minute=0)),
            self.monday.replace(hour=9, minute=30),
        )

    def test_current_window_start(self):
        self.assertEqual(
            self.schedule.current_window_start(
                self.monday.replace(hour=9, minute=45)),
            self.monday.replace(hour=9, minute=30),
        )
        self.assertEqual(
            self.schedule.current_window_start(
                self.monday.replace(hour=13, minute=10)),
            self.monday.replace(hour=13, minute=0),
        )
        self.assertIsNone(self.schedule.current_window_start(
            self.monday.replace(hour=10, minute=0)))


class ChineseHolidayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import chinese_calendar  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("chinesecalendar not installed")

    def setUp(self):
        cfg = Config()
        cfg.badge.holiday_mode = "chinese"
        cfg.badge.workdays = [0, 1, 2, 3, 4]
        cfg.badge.windows = ["09:30-10:00"]
        cfg.badge.interval_seconds = 180
        self.schedule = BadgeSchedule(cfg.badge)

    def test_national_day_is_not_active(self):
        # 2026-10-01 is a Thursday but a public holiday.
        at = dt.datetime(2026, 10, 1, 9, 45)
        import chinese_calendar
        self.assertFalse(chinese_calendar.is_workday(at.date()))
        self.assertFalse(self.schedule.is_active(at))

    def test_next_slot_skips_holiday_week(self):
        # From Sep 30, the next slot must land after the Oct 1-7 holiday.
        nxt = self.schedule.next_slot_after(dt.datetime(2026, 9, 30, 12, 0))
        import chinese_calendar
        self.assertTrue(chinese_calendar.is_workday(nxt.date()), nxt)
        self.assertGreater(nxt, dt.datetime(2026, 10, 7, 23, 59))
        self.assertEqual(nxt.time(), dt.time(9, 30))

    def test_matches_library_for_sample_dates(self):
        import chinese_calendar
        for day in [dt.date(2026, 10, 9), dt.date(2026, 10, 10),
                    dt.date(2026, 2, 16), dt.date(2026, 2, 20),
                    dt.date(2026, 1, 1)]:
            self.assertEqual(self.schedule.is_workday(day),
                             bool(chinese_calendar.is_workday(day)), str(day))

    def test_out_of_range_falls_back_to_weekdays(self):
        # Years beyond the package data raise NotImplementedError; the
        # schedule falls back to the configured weekday set.
        self.assertTrue(self.schedule.is_workday(dt.date(2035, 6, 11)))  # Mon
        self.assertFalse(self.schedule.is_workday(dt.date(2035, 6, 16)))  # Sat


if __name__ == "__main__":
    unittest.main()
