import datetime as dt
import tempfile
import unittest

import numpy as np

from presence_guard.camera import CameraError
from presence_guard.config import Config
from presence_guard.engine import Engine
from presence_guard.state_store import StateStore
from presence_guard.types import BadgeDecision, BadgeResult, PresenceResult


def present(conf=0.8):
    return PresenceResult(present=True, confidence=conf, votes=2, valid_frames=2)


def absent():
    return PresenceResult(present=False, confidence=0.0, votes=0, valid_frames=2)


def badge(decision):
    return BadgeResult(decision=decision, valid_frames=6,
                       present_votes=6 if decision == BadgeDecision.PRESENT else 0,
                       reason="test")


class FakeCamera:
    def __init__(self):
        self.calls = 0
        self.counts = []
        self.error = None

    def capture_burst(self, count, width=None, height=None):
        self.calls += 1
        self.counts.append((count, width, height))
        if self.error:
            raise self.error
        return [np.zeros((480, 640, 3), np.uint8) for _ in range(count)]


class FakeVision:
    def __init__(self):
        self.presence = []
        self.badge = []
        self.badge_calls = 0
        self.presence_flags = []

    def analyze_presence(self, frames, person_fallback=True):
        if not self.presence:
            raise AssertionError("unexpected presence inference")
        self.presence_flags.append(person_fallback)
        return self.presence.pop(0)

    def analyze_badge(self, frames, presence):
        self.badge_calls += 1
        if not self.badge:
            raise AssertionError("unexpected badge inference")
        return self.badge.pop(0)


class FakeIdle:
    def __init__(self, value=30.0):
        self.value = value

    def idle_seconds(self):
        return self.value


class FakeActions:
    def __init__(self):
        self.dry_run = False
        self.lock_enabled = True
        self.locked = False
        self.lock_calls = 0
        self.would_locks = 0
        self.lock_result = True
        self.notifications = []

    def is_locked(self):
        return self.locked

    def request_lock(self):
        if self.dry_run or not self.lock_enabled:
            self.would_locks += 1
            return True
        self.lock_calls += 1
        return self.lock_result

    def notify(self, summary, body, replaces_id=0, timeout_ms=8000):
        if self.dry_run:
            self.would_locks += 0
            return 0
        self.notifications.append((summary, body, replaces_id))
        return len(self.notifications)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = [0.0]
        self.base_epoch = dt.datetime(2026, 10, 19, 9, 0).timestamp()

    def make_engine(self, badge_enabled=False, base=None, startup_delay=5,
                    idle_value=30.0, idle_gate=10):
        if base is None:
            base = dt.datetime(2026, 10, 19, 9, 0)
        self.base_epoch = base.timestamp()
        cfg = Config()
        cfg.scheduler.presence_interval_seconds = 60
        cfg.scheduler.absence_confirm_seconds = 10
        cfg.scheduler.absence_recheck_seconds = 10
        cfg.scheduler.startup_delay_seconds = startup_delay
        cfg.scheduler.camera_error_retry_seconds = 300
        cfg.scheduler.locked_poll_seconds = 60
        cfg.scheduler.idle_gate_seconds = idle_gate
        cfg.camera.device = "0"
        cfg.general.state_dir = self.tmp.name
        cfg.badge.enabled = badge_enabled
        cfg.badge.workdays = [0, 1, 2, 3, 4]
        cfg.badge.windows = ["09:30-10:00", "13:00-13:30"]
        cfg.badge.interval_seconds = 180
        cfg.badge.notify_cooldown_seconds = 600
        cfg.badge.max_reminders_per_window = 3
        self.camera = FakeCamera()
        self.vision = FakeVision()
        self.actions = FakeActions()
        self.idle = FakeIdle(idle_value)
        self.state = StateStore(self.tmp.name)
        engine = Engine(cfg, self.camera, self.vision, self.actions, self.state,
                        clock=lambda: self.clock[0],
                        wall_clock=lambda: self.base_epoch + self.clock[0],
                        idle_monitor=self.idle)
        return engine

    def tick(self, engine, when, presence_result):
        self.clock[0] = when
        self.vision.presence.append(presence_result)
        return engine.tick(when)

    def badge_tick(self, engine, when, presence_result, decision):
        self.clock[0] = when
        self.vision.presence.append(presence_result)
        self.vision.badge.append(badge(decision))
        return engine.tick(when)

    # ----------------------------------------------------------- presence
    def test_absence_locks_after_10s_confirmation(self):
        engine = self.make_engine()
        self.tick(engine, 5, present())
        self.assertEqual(engine.next_presence_at, 65)

        self.tick(engine, 65, absent())
        self.assertEqual(engine.next_presence_at, 75)
        self.assertEqual(self.actions.lock_calls, 0)

        self.tick(engine, 75, absent())
        self.assertEqual(self.actions.lock_calls, 1)
        self.assertEqual(engine.next_presence_at, 135)
        self.assertIsNone(engine.absence_started_at)

    def test_presence_cancels_absence(self):
        engine = self.make_engine()
        self.tick(engine, 5, present())
        self.tick(engine, 65, absent())
        self.tick(engine, 75, present())
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertEqual(engine.next_presence_at, 135)

    def test_not_due_does_nothing(self):
        engine = self.make_engine()
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 1)
        self.clock[0] = 64
        self.assertIsNone(engine.tick(64))
        self.assertEqual(self.camera.calls, 1)

    def test_camera_error_never_locks(self):
        engine = self.make_engine()
        self.camera.error = CameraError("busy")
        self.tick(engine, 5, present())
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertEqual(engine.next_presence_at, 305)
        self.assertIsNone(engine.absence_started_at)

    def test_insufficient_valid_frames_never_locks(self):
        engine = self.make_engine()
        bad = PresenceResult(present=False, votes=0, valid_frames=0,
                             error="model failed")
        self.tick(engine, 5, bad)
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertTrue(engine.camera_unavailable)
        self.assertEqual(engine.next_presence_at, 65)

    def test_locked_screen_does_not_open_camera(self):
        engine = self.make_engine()
        self.actions.locked = True
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 0)
        self.assertEqual(engine.next_presence_at, 65)

        self.actions.locked = False
        self.tick(engine, 65, present())
        self.assertEqual(self.camera.calls, 1)
        self.assertEqual(engine.next_presence_at, 125)

    def test_dry_run_does_not_lock(self):
        engine = self.make_engine()
        self.actions.dry_run = True
        self.tick(engine, 5, present())
        self.tick(engine, 65, absent())
        self.tick(engine, 75, absent())
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertEqual(self.actions.would_locks, 1)
        self.assertEqual(engine.next_presence_at, 135)

    def test_person_fallback_only_on_confirmation(self):
        engine = self.make_engine()
        self.tick(engine, 5, absent())
        self.assertEqual(self.vision.presence_flags, [False])
        self.tick(engine, 15, absent())
        self.assertEqual(self.vision.presence_flags, [False, True])

    def test_active_user_skips_camera(self):
        engine = self.make_engine(idle_value=1.0)
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 0)
        self.assertEqual(engine.next_presence_at, 65)
        self.assertIsNone(engine.absence_started_at)

    def test_idle_user_scans_camera(self):
        engine = self.make_engine(idle_value=20.0)
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 1)

    def test_idle_gate_disabled_always_scans(self):
        engine = self.make_engine(idle_value=1.0, idle_gate=0)
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 1)

    def test_idle_unavailable_always_scans(self):
        engine = self.make_engine(idle_value=None)
        self.tick(engine, 5, present())
        self.assertEqual(self.camera.calls, 1)

    def test_activity_cancels_absence(self):
        engine = self.make_engine(idle_value=20.0)
        self.tick(engine, 5, present())
        self.tick(engine, 65, absent())
        self.assertEqual(self.actions.lock_calls, 0)
        self.idle.value = 1.0
        self.tick(engine, 75, present())
        self.assertEqual(self.camera.calls, 2)
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertEqual(engine.next_presence_at, 135)

    # -------------------------------------------------------------- badge
    def test_badge_is_checked_only_inside_window(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 0),
            startup_delay=0)
        self.tick(engine, 1740, present())  # 09:29
        self.assertEqual(self.vision.badge_calls, 0)
        self.assertEqual(self.actions.notifications, [])

        self.badge_tick(engine, 1800, present(), BadgeDecision.ABSENT)  # 09:30
        self.assertEqual(self.vision.badge_calls, 1)
        self.assertEqual(len(self.actions.notifications), 1)

    def test_badge_deferred_when_nobody_present(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 30),
            startup_delay=0)
        self.tick(engine, 0, absent())  # 09:30, empty chair
        self.assertEqual(self.vision.badge_calls, 0)
        self.assertEqual(self.actions.notifications, [])

        # 10 s later the person is back; absence is cancelled.  The 09:33
        # badge slot was skipped because the chair was empty, so it runs at
        # the next presence check after 09:33.
        self.tick(engine, 10, present())  # 09:30:10
        self.badge_tick(engine, 600, present(), BadgeDecision.ABSENT)  # 09:40
        self.assertEqual(self.vision.badge_calls, 1)
        self.assertEqual(len(self.actions.notifications), 1)

    def test_active_user_still_gets_badge_check(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 30),
            startup_delay=0, idle_value=1.0)
        self.badge_tick(engine, 0, present(), BadgeDecision.ABSENT)
        self.assertEqual(self.camera.calls, 1)
        self.assertEqual(len(self.actions.notifications), 1)

    def test_badge_unknown_does_not_notify(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 30),
            startup_delay=0)
        self.badge_tick(engine, 0, present(), BadgeDecision.UNKNOWN)
        self.assertEqual(len(self.actions.notifications), 0)
        self.assertEqual(self.vision.badge_calls, 1)

    def test_badge_cap_three_per_window_and_reset(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 30),
            startup_delay=0)
        # 09:30, 09:40, 09:50 -> three reminders; 09:55 -> capped.
        self.badge_tick(engine, 0, present(), BadgeDecision.ABSENT)
        self.badge_tick(engine, 600, present(), BadgeDecision.ABSENT)
        self.badge_tick(engine, 1200, present(), BadgeDecision.ABSENT)
        self.assertEqual(len(self.actions.notifications), 3)
        self.assertEqual(self.state.get_int("badge_window_count"), 3)

        report = self.badge_tick(engine, 1500, present(), BadgeDecision.ABSENT)
        self.assertEqual(report.action, "badge_reminder_cap")
        self.assertEqual(len(self.actions.notifications), 3)

        # 10:00: outside the window, no badge check.
        self.tick(engine, 1800, present())
        self.assertEqual(self.vision.badge_calls, 4)

        # 13:00: new window, counter resets, reminder allowed again.
        self.badge_tick(engine, 12600, present(), BadgeDecision.ABSENT)
        self.assertEqual(len(self.actions.notifications), 4)
        self.assertEqual(self.state.get_int("badge_window_count"), 1)
        self.assertEqual(self.state.get("badge_window_key"),
                         "2026-10-19 13:00")

    def test_manual_badge_check_runs_outside_window(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 15, 0),
            startup_delay=0)
        scheduled_before = engine.next_badge_at
        self.clock[0] = 0
        response = engine._handle_control({"cmd": "check_badge"})
        self.assertTrue(response["ok"])
        self.assertTrue(engine.force_badge)

        self.vision.presence.append(present())
        self.vision.badge.append(badge(BadgeDecision.ABSENT))
        report = engine.tick(0)
        self.assertEqual(self.vision.badge_calls, 1)
        self.assertEqual(report.action, "badge_absent_manual")
        self.assertEqual(self.actions.notifications, [])
        self.assertEqual(engine.last_badge["decision"], "absent")
        self.assertEqual(engine.next_badge_at, scheduled_before)

    def test_manual_badge_check_skips_absent_person(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 15, 0),
            startup_delay=0)
        self.clock[0] = 0
        engine._handle_control({"cmd": "check_badge"})
        self.tick(engine, 0, absent())
        self.assertEqual(self.vision.badge_calls, 0)
        self.assertFalse(engine.force_badge)

    def test_no_relock_until_presence_seen_after_lock(self):
        engine = self.make_engine()
        self.tick(engine, 5, present())
        self.tick(engine, 65, absent())
        self.tick(engine, 75, absent())
        self.assertEqual(self.actions.lock_calls, 1)
        self.assertTrue(engine.awaiting_presence)

        # Unlocked remotely, still nobody at the desk: must NOT lock again.
        self.tick(engine, 135, absent())
        self.assertEqual(self.actions.lock_calls, 1)
        self.assertEqual(engine.next_presence_at, 195)

        # The person returns: re-arm the auto-lock.
        self.tick(engine, 195, present())
        self.assertFalse(engine.awaiting_presence)
        self.tick(engine, 255, absent())
        self.tick(engine, 265, absent())
        self.assertEqual(self.actions.lock_calls, 2)

    def test_non_workday_skips_auto_lock(self):
        engine = self.make_engine(base=dt.datetime(2026, 10, 1, 9, 0),
                                  startup_delay=0)
        self.tick(engine, 0, absent())
        self.assertEqual(self.actions.lock_calls, 0)
        self.assertEqual(engine.next_presence_at, 60)
        self.assertEqual(engine.last_report.action, "lock_skipped_holiday")
        self.assertEqual(engine.last_report.state, "absent")

    def test_lock_on_holidays_can_be_enabled(self):
        engine = self.make_engine(base=dt.datetime(2026, 10, 1, 9, 0),
                                  startup_delay=0)
        engine.cfg.scheduler.lock_on_holidays = True
        self.tick(engine, 0, absent())
        self.tick(engine, 10, absent())
        self.assertEqual(self.actions.lock_calls, 1)

    def test_camera_pause_clears_when_camera_returns(self):
        engine = self.make_engine()
        self.camera.error = CameraError("busy")
        self.tick(engine, 5, present())
        self.assertTrue(engine.camera_unavailable)
        self.assertEqual(engine.last_report.state, "camera_error")

        self.camera.error = None
        engine.next_presence_at = 65
        self.tick(engine, 65, present())
        self.assertFalse(engine.camera_unavailable)

    def test_set_camera_command(self):
        engine = self.make_engine()
        self.clock[0] = 10
        engine.next_presence_at = 999
        response = engine._handle_control({
            "cmd": "set_camera", "device": "/dev/video2"})
        self.assertTrue(response["ok"])
        self.assertEqual(engine.cfg.camera.device, "/dev/video2")
        self.assertEqual(engine.next_presence_at, 10)
        self.assertFalse(engine.camera_unavailable)

    def test_retry_camera_command(self):
        engine = self.make_engine()
        self.clock[0] = 10
        engine.next_presence_at = 999
        response = engine._handle_control({"cmd": "retry_camera"})
        self.assertTrue(response["ok"])
        self.assertEqual(engine.next_presence_at, 10)

    def test_manual_badge_busy_does_not_replace_state(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 15, 0),
            startup_delay=0)
        self.tick(engine, 0, present())
        self.assertEqual(engine.last_report.state, "present")
        self.clock[0] = 1
        engine._handle_control({"cmd": "check_badge"})
        self.assertEqual(engine.busy, "badge")
        self.assertEqual(engine.status.data.get("busy"), "badge")
        self.assertEqual(engine.status.data.get("state"), "present")

        self.vision.presence.append(present())
        self.vision.badge.append(badge(BadgeDecision.PRESENT))
        engine.next_presence_at = 1
        engine.tick(1)
        self.assertIsNone(engine.busy)
        self.assertEqual(engine.last_badge["decision"], "present")

    def test_badge_present_does_not_notify(self):
        engine = self.make_engine(
            badge_enabled=True, base=dt.datetime(2026, 10, 19, 9, 30),
            startup_delay=0)
        self.badge_tick(engine, 0, present(), BadgeDecision.PRESENT)
        self.assertEqual(len(self.actions.notifications), 0)
        self.assertEqual(self.vision.badge_calls, 1)


if __name__ == "__main__":
    unittest.main()
