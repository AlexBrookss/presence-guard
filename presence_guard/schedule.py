"""Workday badge windows and 3-minute slot calculation.

"Holiday mode" uses the `chinese_calendar` package (public holidays and
make-up workdays, e.g. National Day / Spring Festival).  When the package is
missing or a date falls outside its data range, the code falls back to plain
Monday..Friday, so the guard keeps working.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

from .config import BadgeConfig


@dataclass(frozen=True)
class TimeWindow:
    start: dt.time
    end: dt.time

    def contains(self, value: dt.time) -> bool:
        return self.start <= value < self.end


def parse_window(value: str) -> TimeWindow:
    try:
        start_text, end_text = value.split("-", 1)
        start = dt.datetime.strptime(start_text.strip(), "%H:%M").time()
        end = dt.datetime.strptime(end_text.strip(), "%H:%M").time()
    except Exception as exc:
        raise ValueError(f"invalid time window {value!r}, expected HH:MM-HH:MM") from exc
    if start >= end:
        raise ValueError(f"time window start must be before end: {value!r}")
    return TimeWindow(start, end)


class WorkdayCalendar:
    """Chinese public holidays + make-up workdays, with a weekday fallback."""

    def __init__(self, cfg: BadgeConfig) -> None:
        self.workdays = set(cfg.workdays)
        self.holiday_mode = cfg.holiday_mode
        self._provider: Callable[[dt.date], Optional[bool]] = (
            self._build_provider())

    def _build_provider(self) -> Callable[[dt.date], Optional[bool]]:
        if self.holiday_mode == "chinese":
            try:
                import chinese_calendar

                def provider(day: dt.date) -> Optional[bool]:
                    try:
                        return bool(chinese_calendar.is_workday(day))
                    except Exception:
                        # Out of the package's data range (newer years).
                        return None

                return provider
            except ImportError:
                log.warning(
                    "chinesecalendar not installed; "
                    "falling back to Monday-Friday workdays")
        return lambda day: None

    def is_workday(self, day: dt.date) -> bool:
        result = self._provider(day)
        if result is None:
            return day.weekday() in self.workdays
        return result


class BadgeSchedule:
    """Computes the next 3-minute slot inside configured workday windows."""

    def __init__(self, cfg: BadgeConfig,
                 calendar: Optional[WorkdayCalendar] = None) -> None:
        self.calendar = calendar or WorkdayCalendar(cfg)
        self.workdays = set(cfg.workdays)
        self.windows = [parse_window(w) for w in cfg.windows]
        self.interval_seconds = int(cfg.interval_seconds)
        self.holiday_mode = cfg.holiday_mode
        if not self.windows:
            raise ValueError("badge.windows must not be empty")
        if self.interval_seconds < 30:
            raise ValueError("badge.interval_seconds must be >= 30")

    def is_workday(self, day: dt.date) -> bool:
        return self.calendar.is_workday(day)

    def is_active(self, now: dt.datetime) -> bool:
        if not self.is_workday(now.date()):
            return False
        current = now.time()
        return any(window.contains(current) for window in self.windows)

    def current_window_start(self, now: dt.datetime) -> dt.datetime | None:
        """Start of the window containing ``now`` (used as a reminder bucket)."""

        if not self.is_workday(now.date()):
            return None
        current = now.time()
        for window in self.windows:
            if window.contains(current):
                return dt.datetime.combine(now.date(), window.start)
        return None

    def next_slot_after(self, now: dt.datetime) -> dt.datetime:
        """First slot strictly after ``now``; may be in a later window/day."""

        for day_offset in range(0, 30):
            day = (now + dt.timedelta(days=day_offset)).date()
            if not self.is_workday(day):
                continue
            for window in self.windows:
                start = dt.datetime.combine(day, window.start)
                end = dt.datetime.combine(day, window.end)
                if day_offset == 0 and now >= end:
                    continue
                if now < start:
                    return start
                if start <= now < end:
                    elapsed = (now - start).total_seconds()
                    step = int(elapsed // self.interval_seconds) + 1
                    candidate = start + dt.timedelta(
                        seconds=step * self.interval_seconds)
                    if candidate < end:
                        return candidate
        # 30 days without a matching workday means the config is unusable.
        raise RuntimeError("no badge slot found in the next 30 days")

    def next_check_at(self, now: dt.datetime) -> dt.datetime:
        """A check is due immediately when the app starts inside a window."""

        if self.is_active(now):
            return now
        return self.next_slot_after(now)


def describe_windows(cfg: BadgeConfig) -> str:
    days = ",".join(str(d) for d in sorted(cfg.workdays))
    holiday = "chinese holidays" if cfg.holiday_mode == "chinese" else "weekdays"
    return (f"workdays=[{days}] windows={cfg.windows} "
            f"every {cfg.interval_seconds}s ({holiday})")
