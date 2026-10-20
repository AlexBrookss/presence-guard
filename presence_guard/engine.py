"""Scheduling engine.

Timeline (defaults):

    every 60 s    -> cheap idle query; if the user is active, no camera at all
    every 60 s    -> otherwise one quick YuNet scan (320x240)
    no face       -> 10 s later, a NanoDet confirmation scan; lock if still absent
    badge windows -> workday 09:30-10:00 / 13:00-13:30, one check every 3 min
    screen locked -> no camera access at all, only poll the lock state

The camera is never opened when it is not needed.  A camera/model error is
fail-safe: it can never trigger the lock.

Every cycle publishes a human-readable snapshot to ``status.json`` and the
daemon listens on ``control.sock`` so the PyQt UI can request an immediate
check, pause/resume, or read the current state.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .cache import cleanup_dir
from .camera import CameraCapture, CameraError
from .config import Config, set_config_value
from .dbus_api import DesktopActions
from .idle import IdleMonitor
from .schedule import BadgeSchedule, WorkdayCalendar, describe_windows
from .state_store import StateStore
from .status import ControlServer, StatusStore
from .types import BadgeDecision, BadgeResult, CycleReport
from .vision import VisionPipeline

log = logging.getLogger(__name__)

BADGE_TEXT = {
    "present": "已佩戴工牌",
    "absent": "未佩戴工牌",
    "unknown": "无法判断",
    "not_checked": "尚未检查",
}


class Engine:
    def __init__(self, cfg: Config, camera: CameraCapture, vision: VisionPipeline,
                 actions: DesktopActions, state: StateStore,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time,
                 idle_monitor: Optional[IdleMonitor] = None,
                 status_store: Optional[StatusStore] = None) -> None:
        self.cfg = cfg
        self.calendar = WorkdayCalendar(cfg.badge)
        self.camera = camera
        self.vision = vision
        self.actions = actions
        self.state = state
        self.clock = clock
        self.wall_clock = wall_clock
        self.idle = idle_monitor or IdleMonitor(cfg.scheduler.idle_provider)
        self.status = status_store or StatusStore(cfg.general.state_dir)
        self.control: Optional[ControlServer] = None
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self._control_lock = threading.Lock()
        self._state_lock = threading.Lock()

        now = self.clock()
        self.next_presence_at = now + cfg.scheduler.startup_delay_seconds
        self.absence_started_at: Optional[float] = None
        self.absence_rechecks = 0
        self.pause_until = 0.0
        self.force_badge = False
        self.camera_unavailable = False
        self.busy: Optional[str] = None
        self._last_state_text = "启动中"
        self._last_camera = "ok"
        self._holiday_noted_date: Optional[object] = None
        # After the screen is locked (by us or by the user) the auto-lock is
        # re-armed only once a person is detected again, so unlocking does not
        # immediately trigger a second lock.
        self.awaiting_presence = False
        self.last_report: Optional[CycleReport] = None

        self.last_badge: dict[str, Any] = {
            "decision": "not_checked",
            "text": BADGE_TEXT["not_checked"],
            "reason": "",
            "checked_at": None,
            "votes": "",
        }
        saved_decision = state.get("last_badge_decision")
        saved_ts = state.get_float("last_badge_check_ts", 0.0)
        if saved_decision in BADGE_TEXT and saved_ts > 0:
            self.last_badge["decision"] = saved_decision
            self.last_badge["text"] = BADGE_TEXT[saved_decision]
            self.last_badge["checked_at"] = _dt.datetime.fromtimestamp(
                saved_ts).strftime("%m-%d %H:%M")
        self._last_state = "starting"
        self._last_present: Optional[bool] = None
        self._last_badge_decision: Optional[str] = None
        self._cap_noted_window: Optional[str] = None

        self.next_badge_at = float("inf")
        self.badge_schedule: Optional[BadgeSchedule] = None
        if cfg.badge.enabled:
            self.badge_schedule = BadgeSchedule(cfg.badge, self.calendar)
            now_dt = _dt.datetime.fromtimestamp(self.wall_clock())
            self.next_badge_at = self.badge_schedule.next_check_at(now_dt).timestamp()

        log.info(
            "engine: presence every %ss, absence confirm %ss (recheck %ss), "
            "badge %s",
            cfg.scheduler.presence_interval_seconds,
            cfg.scheduler.absence_confirm_seconds,
            cfg.scheduler.absence_recheck_seconds,
            describe_windows(cfg.badge) if cfg.badge.enabled else "disabled",
        )

    # ------------------------------------------------------------------ main
    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()

    def run_forever(self) -> None:
        log.info("presence-guard running; camera=%s", self.cfg.camera.device)
        self.status.add_event("服务启动", "info")
        removed, freed = cleanup_dir(
            self.cfg.general.debug_dir,
            self.cfg.general.debug_keep_files,
            self.cfg.general.debug_max_age_days,
        )
        if removed:
            self.status.add_event(
                f"已清理旧调试图片 {removed} 张（{freed // 1024} KB）", "info")
        self._publish("starting", "启动中", None, None)

        self.control = ControlServer(self.cfg.general.state_dir, self._handle_control)
        try:
            self.control.start()
        except Exception:
            log.exception("control socket unavailable; UI controls disabled")
            self.control = None

        try:
            while not self.stop_event.is_set():
                now = self.clock()
                wait = max(0.0, self.next_presence_at - now)
                self.wake_event.wait(wait)
                self.wake_event.clear()
                if self.stop_event.is_set():
                    break
                try:
                    self.tick()
                except Exception:
                    log.exception("cycle failed; retrying in 60 s")
                    self.absence_started_at = None
                    self.absence_rechecks = 0
                    self.next_presence_at = self.clock() + 60
                    self._publish("error", "内部错误，60 秒后重试", None, None)
        finally:
            if self.control is not None:
                self.control.stop()
            self.status.add_event("服务停止", "info")
            self._publish("stopped", "已停止", None, None)
            log.info("presence-guard stopped")

    def tick(self, now: Optional[float] = None) -> Optional[CycleReport]:
        now = self.clock() if now is None else now

        if self.pause_until > now:
            remaining = max(1, int(self.pause_until - now))
            self.next_presence_at = min(self.pause_until, now + 60)
            report = CycleReport(
                timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                state="paused",
                state_text=f"已暂停（剩余 {remaining // 60} 分 {remaining % 60} 秒）",
            )
            self._publish(report.state, report.state_text, None, report)
            return report

        if now < self.next_presence_at:
            return None

        self.busy = None
        started = time.perf_counter()
        cpu_started = time.process_time()
        report = CycleReport(
            timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        def finish(rep: CycleReport, state: str, text: str,
                   present: Optional[bool], camera: str = "ok") -> CycleReport:
            rep.elapsed_ms = (time.perf_counter() - started) * 1000.0
            rep.cpu_ms = (time.process_time() - cpu_started) * 1000.0
            rep.state = state
            rep.state_text = text
            self.last_report = rep
            self._publish(state, text, present, rep, camera=camera)
            return rep

        locked = self._safe_is_locked()
        if locked:
            report.locked = True
            self.awaiting_presence = True
            self.next_presence_at = now + self.cfg.scheduler.locked_poll_seconds
            log.info("screen is locked; camera stays off")
            return finish(report, "locked", "锁屏中", None)

        now_dt = _dt.datetime.fromtimestamp(self.wall_clock())
        force_badge = self.force_badge
        self.force_badge = False
        scheduled_badge_due = False
        if self.badge_schedule is not None:
            if self.badge_schedule.is_active(now_dt):
                scheduled_badge_due = self.wall_clock() >= self.next_badge_at
            elif self.next_badge_at <= self.wall_clock():
                # The window ended while the app was busy/locked: jump to the
                # next workday window instead of firing late.
                self.next_badge_at = self.badge_schedule.next_slot_after(
                    now_dt).timestamp()
        badge_due = scheduled_badge_due or force_badge

        presence_count = self.cfg.camera.presence_frames
        badge_count = self.cfg.camera.badge_frames if badge_due else 0

        idle_seconds = None
        if self.cfg.scheduler.idle_gate_seconds > 0:
            idle_seconds = self.idle.idle_seconds()
        if (idle_seconds is not None
                and idle_seconds < self.cfg.scheduler.idle_gate_seconds
                and not badge_due):
            # Keyboard/mouse activity is a free, reliable presence signal.
            # No camera wake-up while the user is actively working.
            self._reset_absence()
            if self.awaiting_presence:
                self.awaiting_presence = False
                self.status.add_event("检测到在座，自动锁屏重新启用", "info")
            report.action = "active"
            report.idle_seconds = idle_seconds
            self.next_presence_at = now + self.cfg.scheduler.presence_interval_seconds
            log.debug("user active (idle=%.1fs); camera skipped", idle_seconds)
            return finish(report, "active", "在工位（键鼠活跃）", True)

        try:
            frames = self.camera.capture_burst(
                presence_count + badge_count,
                width=(self.cfg.camera.badge_width if badge_due
                       else self.cfg.camera.presence_width),
                height=(self.cfg.camera.badge_height if badge_due
                        else self.cfg.camera.presence_height),
            )
        except CameraError as exc:
            report.error = str(exc)
            self._cancel_absence("camera error")
            self.camera_unavailable = True
            self.next_presence_at = now + self._camera_retry_delay()
            log.warning("camera unavailable: %s (detection paused)", exc)
            self._notify_camera_error()
            return finish(report, "camera_error",
                          "摄像头不可用，检测已暂停（恢复后自动继续）",
                          None, camera="error")
        except Exception as exc:
            report.error = f"camera: {exc}"
            self._cancel_absence("camera error")
            self.camera_unavailable = True
            self.next_presence_at = now + self._camera_retry_delay()
            log.exception("unexpected camera error; auto-lock paused")
            return finish(report, "camera_error", "摄像头异常，检测已暂停",
                          None, camera="error")

        if self.camera_unavailable:
            self.camera_unavailable = False
            self.status.add_event("摄像头已恢复，继续检测", "info")

        try:
            presence = self.vision.analyze_presence(
                frames[:presence_count],
                person_fallback=self.absence_started_at is not None,
            )
        except Exception as exc:
            report.error = f"vision: {exc}"
            self._cancel_absence("vision error")
            self.next_presence_at = now + self.cfg.scheduler.camera_error_retry_seconds
            log.exception("presence inference failed; auto-lock paused")
            return finish(report, "error", "识别异常", None, camera="error")

        report.presence = presence
        minimum_valid = max(1, math.ceil(presence_count / 2.0))
        if presence.error or presence.valid_frames < minimum_valid:
            report.error = presence.error or (
                f"only {presence.valid_frames}/{presence_count} valid frames")
            self._cancel_absence("insufficient valid frames")
            self.camera_unavailable = True
            retry = max(30, self.cfg.scheduler.camera_error_retry_seconds // 5)
            self.next_presence_at = now + retry
            log.warning("presence analysis unusable: %s (auto-lock paused)",
                        report.error)
            return finish(report, "camera_error",
                          "画面不可用（过暗/被遮挡），不会锁屏",
                          None, camera="error")

        report.idle_seconds = idle_seconds

        if presence.present:
            self._reset_absence()
            if self.awaiting_presence:
                self.awaiting_presence = False
                self.status.add_event("检测到在座，自动锁屏重新启用", "info")
            report.action = "present"

            if badge_due:
                badge_frames = frames[presence_count:]
                try:
                    badge = self.vision.analyze_badge(badge_frames, presence)
                except Exception as exc:
                    log.exception("badge inference failed")
                    badge = None
                    report.error = f"badge: {exc}"
                if badge is not None:
                    report.badge = badge
                    report.action = self._handle_badge(
                        badge, now_dt, notify=not force_badge)
                    self.state.update(
                        last_badge_check_ts=self.wall_clock(),
                        last_badge_decision=badge.decision.value,
                    )
                    self.state.save()
                    self._note_badge(badge, manual=force_badge)
                log.info("badge check%s: %s",
                         " (manual)" if force_badge else "",
                         self._badge_summary(badge))

            self.next_presence_at = now + self.cfg.scheduler.presence_interval_seconds
            log.info("presence: present conf=%.2f votes=%d/%d",
                     presence.confidence, presence.votes, presence.valid_frames)
            result = finish(report, "present", "在工位", True)
        else:
            if force_badge:
                self.status.add_event("工牌检测跳过：未检测到人", "warn")
            if self.awaiting_presence:
                # The screen was already locked once; do not lock again until
                # a person has been seen (e.g. remote unlock while away).
                report.action = "absent_no_relock"
                self.next_presence_at = (now
                                         + self.cfg.scheduler.presence_interval_seconds)
                log.info("presence: absent (lock re-armed only after presence)")
                result = finish(report, "absent",
                                "已离座（等待返回，暂不重复锁屏）", False)
            else:
                self._handle_absence(now, report, now_dt)
                log.info("presence: absent votes=%d/%d (next check in %ds)",
                         presence.votes, presence.valid_frames,
                         max(0, int(self.next_presence_at - now)))
                if report.action == "lock":
                    result = finish(report, "locked", "已离座，已锁屏", False)
                elif report.action == "lock_skipped_holiday":
                    result = finish(report, "absent",
                                    "已离座（非工作日，不自动锁屏）", False)
                else:
                    result = finish(report, "absent", "已离座", False)

        if scheduled_badge_due and self.badge_schedule is not None:
            self.next_badge_at = self.badge_schedule.next_slot_after(
                now_dt).timestamp()
            log.debug("next badge slot: %s", self.next_badge_at)

        self.last_report = result
        log.debug("cycle took %.0f ms (cpu %.0f ms)",
                  result.elapsed_ms, result.cpu_ms)
        return result

    # ------------------------------------------------------------- absence
    def _reset_absence(self) -> None:
        if self.absence_started_at is not None:
            log.info("person is back; absence confirmation cancelled")
        self.absence_started_at = None
        self.absence_rechecks = 0

    def _cancel_absence(self, reason: str) -> None:
        if self.absence_started_at is not None:
            log.info("absence confirmation cancelled: %s", reason)
        self.absence_started_at = None
        self.absence_rechecks = 0

    def _handle_absence(self, now: float, report: CycleReport,
                        now_dt: _dt.datetime) -> None:
        cfg = self.cfg.scheduler
        today = now_dt.date()
        if not self.calendar.is_workday(today) and not cfg.lock_on_holidays:
            # Non-workday: keep monitoring, but never start the 10 s
            # confirmation / lock sequence.
            report.action = "lock_skipped_holiday"
            self.absence_started_at = None
            self.absence_rechecks = 0
            self.next_presence_at = now + cfg.presence_interval_seconds
            if self._holiday_noted_date != today:
                self._holiday_noted_date = today
                self.status.add_event("非工作日，已跳过自动锁屏", "info")
            log.info("non-workday; absence confirmation skipped")
            return
        if self.absence_started_at is None:
            self.absence_started_at = now
            self.absence_rechecks = 0
            wait = max(1, min(cfg.absence_recheck_seconds,
                              cfg.absence_confirm_seconds))
            self.next_presence_at = now + wait
            report.action = "absence_start"
            log.info("person absent; confirming for %ss (next check in %ss)",
                     cfg.absence_confirm_seconds, wait)
            return

        self.absence_rechecks += 1
        elapsed = now - self.absence_started_at
        if elapsed >= cfg.absence_confirm_seconds:
            today = now_dt.date()
            workday = self.calendar.is_workday(today)
            if not workday and not cfg.lock_on_holidays:
                report.action = "lock_skipped_holiday"
                self.absence_started_at = None
                self.absence_rechecks = 0
                self.next_presence_at = now + cfg.presence_interval_seconds
                if self._holiday_noted_date != today:
                    self._holiday_noted_date = today
                    self.status.add_event("非工作日，已跳过自动锁屏", "info")
                log.info("non-workday; auto-lock skipped")
                return
            ok = self.actions.request_lock()
            if ok:
                report.action = "lock"
                self.awaiting_presence = True
                log.warning("no person for %.0fs (confirmed); screen locked",
                            elapsed)
                self.state.set("last_lock_ts", self.wall_clock())
                self.state.save()
                self.status.add_event("已离座，自动锁屏", "warn")
                self.absence_started_at = None
                self.absence_rechecks = 0
                self.next_presence_at = now + cfg.locked_poll_seconds
            else:
                report.action = "lock_failed"
                log.error("screen lock failed; retrying in 60 s")
                self.next_presence_at = now + 60
            return

        remaining = max(1.0, cfg.absence_confirm_seconds - elapsed)
        wait = max(1, min(cfg.absence_recheck_seconds, remaining))
        self.next_presence_at = now + wait
        report.action = "absence_check"
        log.info("still absent after %.0fs; locking in %.0fs",
                 elapsed, remaining)

    # --------------------------------------------------------------- badge
    def _handle_badge(self, badge: BadgeResult, now_dt: _dt.datetime,
                      notify: bool = True) -> str:
        now = self.wall_clock()
        if badge.decision == BadgeDecision.PRESENT:
            if not self.actions.dry_run:
                self.state.set("last_badge_present_ts", now)
            log.info("badge present (%s)", badge.reason)
            return "badge_present"

        if badge.decision == BadgeDecision.UNKNOWN:
            log.info("badge unknown (%s); no notification", badge.reason)
            return "badge_unknown"

        if not notify:
            log.info("badge absent (manual check); no notification")
            return "badge_absent_manual"

        # ABSENT: apply the per-window reminder cap, then the cooldown.
        window_key = None
        if self.badge_schedule is not None:
            window_start = self.badge_schedule.current_window_start(now_dt)
            if window_start is not None:
                window_key = window_start.strftime("%Y-%m-%d %H:%M")
                if self.state.get("badge_window_key") != window_key:
                    if not self.actions.dry_run:
                        self.state.update(
                            badge_window_key=window_key,
                            badge_window_count=0,
                        )
                        self.state.save()
                    else:
                        self.state.data["badge_window_key"] = window_key
                        self.state.data["badge_window_count"] = 0

        count = self.state.get_int("badge_window_count", 0)
        maximum = self.cfg.badge.max_reminders_per_window
        if window_key is not None and count >= maximum:
            log.info("badge absent, but reminder cap reached (%d/%d) for %s; "
                     "no more reminders in this window",
                     count, maximum, window_key)
            if self._cap_noted_window != window_key:
                self._cap_noted_window = window_key
                self.status.add_event(
                    f"工牌提醒已达上限（{maximum} 次），本时段不再提醒", "warn")
            return "badge_reminder_cap"

        cooldown = self.cfg.badge.notify_cooldown_seconds
        last = self.state.get_float("last_badge_notify_ts", 0.0)
        if cooldown > 0 and last > 0 and (now - last) < cooldown:
            log.info("badge absent, but notification cooldown is active "
                     "(%.0fs left)", cooldown - (now - last))
            return "badge_absent_cooldown"

        # Each reminder stays as its own entry in the notification centre, so
        # the 3 allowed reminders remain reviewable instead of overwriting
        # each other.
        notification_id = self.actions.notify(
            self.cfg.badge.notify_title,
            self.cfg.badge.notify_body,
            replaces_id=0,
            timeout_ms=self.cfg.badge.notify_timeout_ms,
        )
        if notification_id is None:
            log.warning("badge absent, but notification could not be shown")
            return "notify_failed"

        if self.actions.dry_run:
            log.info("DRY-RUN: badge reminder %d/%d for %s",
                     count + 1, maximum, window_key or "current window")
            return "notify_dry_run"

        self.state.update(
            last_badge_notify_ts=now,
            last_notification_id=int(notification_id),
            badge_window_count=count + 1,
        )
        self.state.save()
        self.status.add_event(
            f"工牌提醒已发送（{count + 1}/{maximum}）", "warn")
        log.warning("badge absent; notification shown (id=%s, %d/%d in window)",
                    notification_id, count + 1, maximum)
        return "notify"

    def _note_badge(self, badge: BadgeResult, manual: bool = False) -> None:
        stats = [s for s in badge.stats if s.valid]
        largest = max((s.largest_area for s in stats), default=0)
        self.last_badge = {
            "decision": badge.decision.value,
            "text": BADGE_TEXT.get(badge.decision.value, "无法判断"),
            "reason": badge.reason,
            "checked_at": _dt.datetime.fromtimestamp(
                self.wall_clock()).strftime("%H:%M:%S"),
            "votes": f"{badge.present_votes}/{badge.valid_frames}",
            "largest": largest,
        }
        if badge.decision.value != self._last_badge_decision:
            level = "warn" if badge.decision != BadgeDecision.PRESENT else "info"
            suffix = "，手动检查" if manual else ""
            self.status.add_event(
                f"工牌：{self.last_badge['text']}（{self.last_badge['votes']}）{suffix}",
                level)
            self._last_badge_decision = badge.decision.value

    @staticmethod
    def _badge_summary(badge: Optional[BadgeResult]) -> str:
        if badge is None:
            return "not run"
        stats = [s for s in badge.stats if s.valid]
        ratio = max((s.blue_ratio for s in stats), default=0.0)
        largest = max((s.largest_area for s in stats), default=0)
        return (f"{badge.decision.value} votes={badge.present_votes}/"
                f"{badge.valid_frames} max_ratio={ratio:.4f} "
                f"max_component={largest} ({badge.reason})")

    # -------------------------------------------------------------- status
    def _publish(self, state: str, text: str, present: Optional[bool],
                 report: Optional[CycleReport], camera: str = "ok") -> None:
        clock_now = self.clock()
        wall_now = self.wall_clock()
        remaining = max(0, int(round(self.next_presence_at - clock_now)))
        next_presence_text = _dt.datetime.fromtimestamp(
            wall_now + remaining).strftime("%H:%M:%S")

        badge_window_active = bool(
            self.badge_schedule is not None
            and self.badge_schedule.is_active(
                _dt.datetime.fromtimestamp(wall_now)))
        next_badge_text = None
        if self.badge_schedule is not None and math.isfinite(self.next_badge_at):
            next_badge_text = _dt.datetime.fromtimestamp(
                self.next_badge_at).strftime("%m-%d %H:%M")

        paused_until = None
        if self.pause_until > clock_now:
            paused_until = _dt.datetime.fromtimestamp(
                wall_now + (self.pause_until - clock_now)).strftime("%H:%M:%S")

        if state in ("present", "active"):
            present_value: Optional[bool] = True
        elif state == "absent":
            present_value = False
        else:
            present_value = present

        if present_value is True and self._last_present is not True:
            self.status.add_event("检测到在座", "info")
        elif present_value is False and self._last_present is not False:
            self.status.add_event("检测到离座", "warn")
        self._last_present = present_value

        if state != self._last_state and state in ("error", "camera_error",
                                                   "locked", "paused", "stopped"):
            self.status.add_event(text, "warn" if state != "stopped" else "info")
        self._last_state = state
        self._last_state_text = text
        self._last_camera = camera

        fields: dict[str, Any] = {
            "state": state,
            "state_text": text,
            "present": present_value,
            "badge": self.last_badge.get("decision", "not_checked"),
            "badge_text": self.last_badge.get("text", "尚未检查"),
            "badge_reason": self.last_badge.get("reason", ""),
            "badge_checked_at": self.last_badge.get("checked_at"),
            "badge_votes": self.last_badge.get("votes", ""),
            "badge_mode": self.cfg.badge.mode,
            "badge_holiday_mode": self.cfg.badge.holiday_mode,
            "badge_windows": self.cfg.badge.windows,
            "badge_window_active": badge_window_active,
            "badge_reminder_count": self.state.get_int("badge_window_count", 0),
            "badge_reminder_max": self.cfg.badge.max_reminders_per_window,
            "next_presence_at": next_presence_text,
            "next_presence_in": remaining,
            "next_badge_at": next_badge_text,
            "presence_interval": self.cfg.scheduler.presence_interval_seconds,
            "absence_confirm": self.cfg.scheduler.absence_confirm_seconds,
            "badge_interval": self.cfg.badge.interval_seconds,
            "paused_until": paused_until,
            "awaiting_presence": bool(self.awaiting_presence),
            "camera": camera,
            "camera_unavailable": bool(self.camera_unavailable),
            "camera_device": self.cfg.camera.device,
            "busy": self.busy,
            "is_workday": bool(self.calendar.is_workday(
                _dt.datetime.fromtimestamp(wall_now).date())),
            "lock_on_holidays": bool(self.cfg.scheduler.lock_on_holidays),
            "lock_allowed_today": bool(
                self.calendar.is_workday(_dt.datetime.fromtimestamp(wall_now).date())
                or self.cfg.scheduler.lock_on_holidays),
            "dry_run": bool(self.actions.dry_run),
            "idle_seconds": (round(report.idle_seconds, 1)
                             if report is not None and report.idle_seconds is not None
                             else None),
            "cpu_ms": round(report.cpu_ms, 1) if report is not None else None,
            "cycle_ms": round(report.elapsed_ms, 1) if report is not None else None,
        }
        with self._state_lock:
            self.status.publish(**fields)

    def _handle_control(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = str(request.get("cmd") or "")
        with self._control_lock:
            if cmd == "ping":
                return {"ok": True, "state": self._last_state}
            if cmd == "check_now":
                self.pause_until = 0.0
                self.next_presence_at = self.clock()
                self.wake_event.set()
                self.status.add_event("手动触发一次检测", "info")
                self.busy = "presence"
                self._publish(self._last_state, self._last_state_text,
                              self._last_present, self.last_report,
                              camera=self._last_camera)
                return {"ok": True, "state": "scheduled"}
            if cmd == "check_badge":
                self.force_badge = True
                self.pause_until = 0.0
                self.next_presence_at = self.clock()
                self.wake_event.set()
                self.status.add_event("手动触发工牌检测", "info")
                self.busy = "badge"
                self._publish(self._last_state, self._last_state_text,
                              self._last_present, self.last_report,
                              camera=self._last_camera)
                return {"ok": True, "state": "scheduled"}
            if cmd == "set_camera":
                device = str(request.get("device") or "").strip()
                if not device:
                    return {"ok": False, "error": "empty device"}
                self.cfg.camera.device = device
                config_path = (self.cfg.config_path
                               or str(Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
                                      / "presence-guard" / "config.toml"))
                set_config_value(config_path, "camera", "device", device)
                self.camera_unavailable = False
                self.awaiting_presence = False
                self._cancel_absence("camera switched")
                self.next_presence_at = self.clock()
                self.wake_event.set()
                self.status.add_event(f"摄像头切换到 {device}", "info")
                self._publish(self._last_state, self._last_state_text,
                              self._last_present, self.last_report,
                              camera=self._last_camera)
                return {"ok": True, "device": device}
            if cmd == "retry_camera":
                self.pause_until = 0.0
                self.force_badge = False
                self.next_presence_at = self.clock()
                self.wake_event.set()
                self.status.add_event("手动重试摄像头", "info")
                return {"ok": True, "state": "scheduled"}
            if cmd == "pause":
                try:
                    seconds = float(request.get("seconds") or 1800)
                except (TypeError, ValueError):
                    seconds = 1800.0
                seconds = max(10.0, min(seconds, 24 * 3600.0))
                self.pause_until = self.clock() + seconds
                self.next_presence_at = self.clock()
                self._cancel_absence("paused by user")
                self.wake_event.set()
                self.status.add_event(
                    f"已暂停 {int(seconds // 60)} 分钟", "warn")
                return {"ok": True, "seconds": int(seconds)}
            if cmd == "resume":
                self.pause_until = 0.0
                self.next_presence_at = self.clock()
                self.wake_event.set()
                self.status.add_event("已恢复检测", "info")
                return {"ok": True}
        return {"ok": False, "error": f"unknown command: {cmd}"}

    # -------------------------------------------------------------- helpers
    def _camera_retry_delay(self) -> float:
        device = self.cfg.camera.device
        if device.startswith("/") and not Path(device).exists():
            # Device vanished: poll often so it resumes when replugged.
            return 15.0
        return float(self.cfg.scheduler.camera_error_retry_seconds)

    def _safe_is_locked(self) -> Optional[bool]:
        try:
            return self.actions.is_locked()
        except Exception as exc:
            log.debug("lock-state query failed: %s", exc)
            return None

    def _notify_camera_error(self) -> None:
        if not self.cfg.general.notify_on_camera_error:
            return
        now = self.wall_clock()
        cooldown = self.cfg.scheduler.camera_error_notify_cooldown_seconds
        last = self.state.get_float("last_camera_error_notify_ts", 0.0)
        if cooldown > 0 and last > 0 and (now - last) < cooldown:
            return
        notification_id = self.actions.notify(
            "摄像头不可用",
            "presence-guard 无法打开摄像头，自动锁屏已暂停。请检查设备是否被占用。",
            replaces_id=self.state.get_int("last_notification_id", 0),
        )
        if notification_id is not None:
            self.state.set("last_camera_error_notify_ts", now)
            self.state.save()

    # ----------------------------------------------------------- one-shot
    def inspect(self, image_path: Optional[str] = None,
                force_badge: bool = False,
                debug_dir: Optional[Path] = None
                ) -> tuple[CycleReport, Optional[object]]:
        """Analyse one image or one camera burst without acting on the desktop."""

        started = time.perf_counter()
        cpu_started = time.process_time()
        report = CycleReport(
            timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        def finish(rep: CycleReport) -> CycleReport:
            rep.elapsed_ms = (time.perf_counter() - started) * 1000.0
            rep.cpu_ms = (time.process_time() - cpu_started) * 1000.0
            return rep

        frame = None
        if image_path:
            import cv2
            frame = cv2.imread(image_path)
            if frame is None:
                report.error = f"cannot read image: {image_path}"
                return finish(report), None
            frames = [frame]
        else:
            try:
                frames = self.camera.capture_burst(
                    self.cfg.camera.presence_frames + self.cfg.camera.badge_frames)
            except CameraError as exc:
                report.error = str(exc)
                return finish(report), None
            frame = frames[0]

        presence = self.vision.analyze_presence(
            frames[:self.cfg.camera.presence_frames], person_fallback=True)
        report.presence = presence
        if not presence.present:
            report.action = "absent"
        badge = None
        if presence.present and force_badge:
            badge_frames = frames[self.cfg.camera.presence_frames:] or frames
            # A single still image cannot supply the usual multi-frame vote;
            # repeat it so --image gives a meaningful decision.
            if len(badge_frames) < self.cfg.badge.min_valid_frames:
                badge_frames = badge_frames * self.cfg.badge.min_valid_frames
            badge = self.vision.analyze_badge(badge_frames, presence)
            report.badge = badge
            report.action = f"badge_{badge.decision.value}"

        # Annotate the badge frame that actually voted, not the presence frame.
        display = frame
        if (badge is not None and badge.best_index is not None
                and image_path is None
                and self.cfg.camera.presence_frames < len(frames)
                and badge.best_index < len(frames) - self.cfg.camera.presence_frames):
            display = frames[self.cfg.camera.presence_frames + badge.best_index]
        annotated = self.vision.annotate(display, presence, badge) if display is not None else None
        if debug_dir is not None and annotated is not None:
            import cv2
            debug_dir = Path(debug_dir).expanduser()
            debug_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = debug_dir / (
                _dt.datetime.now().strftime("check-%Y%m%d-%H%M%S.jpg"))
            cv2.imwrite(str(path), annotated)
            try:
                path.chmod(0o600)
            except OSError:
                pass
            log.info("debug image written to %s", path)
            cleanup_dir(
                debug_dir,
                self.cfg.general.debug_keep_files,
                self.cfg.general.debug_max_age_days,
            )
        return finish(report), annotated
