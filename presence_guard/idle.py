"""Cheap keyboard/mouse idle query used as a pre-filter for the camera.

DDE's ``org.freedesktop.ScreenSaver`` does not expose ``GetSessionIdleTime``,
so the standard X11 ``xprintidle`` helper is used when available.  If it is
missing (or the session is Wayland), the guard simply falls back to camera
checks; it never fails closed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

log = logging.getLogger(__name__)


class IdleMonitor:
    def __init__(self, provider: str = "auto", binary: str = "") -> None:
        self.provider = (provider or "auto").lower()
        self.binary = binary or shutil.which("xprintidle") or ""
        self._warned = False
        self._display = os.environ.get("DISPLAY") or ":0"

    @property
    def available(self) -> bool:
        return self.provider != "disabled" and bool(self.binary)

    def idle_seconds(self) -> float | None:
        """Seconds since the last X11 input event, or None if unavailable."""

        if self.provider == "disabled" or not self.binary:
            if not self._warned and self.provider != "disabled":
                log.info("xprintidle not found; camera checks will run every "
                         "interval (install xprintidle to skip them while active)")
                self._warned = True
            return None
        env = dict(os.environ)
        if not env.get("DISPLAY"):
            env["DISPLAY"] = self._display
        try:
            result = subprocess.run(
                [self.binary], capture_output=True, text=True, timeout=1.5,
                env=env, check=False,
            )
            if result.returncode != 0:
                return None
            return max(0.0, int(result.stdout.strip()) / 1000.0)
        except Exception as exc:
            if not self._warned:
                log.info("idle query failed (%s); falling back to camera checks",
                         exc)
                self._warned = True
            return None
