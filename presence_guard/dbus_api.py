"""DDE / freedesktop D-Bus actions used by the guard.

Only two desktop interfaces are needed:

* ``com.deepin.SessionManager`` (``RequestLock``, property ``Locked``)
* ``org.freedesktop.Notifications`` (``Notify``)

The lock front-end (``com.deepin.dde.lockFront.Show``) is used as a fallback
for older DDE builds.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

SESSION_MANAGER = "com.deepin.SessionManager"
SESSION_PATH = "/com/deepin/SessionManager"
SESSION_IFACE = "com.deepin.SessionManager"
LOCK_FRONT = "com.deepin.dde.lockFront"
LOCK_FRONT_PATH = "/com/deepin/dde/lockFront"
LOCK_FRONT_IFACE = "com.deepin.dde.lockFront"
NOTIFICATIONS = "org.freedesktop.Notifications"
NOTIFICATIONS_PATH = "/org/freedesktop/Notifications"
NOTIFICATIONS_IFACE = "org.freedesktop.Notifications"


class DesktopActions:
    def __init__(self, dry_run: bool = False, lock_enabled: bool = True) -> None:
        self.dry_run = dry_run
        self.lock_enabled = lock_enabled
        self._bus = None
        self._gi = None
        self._gi_error = None
        self._notification_bus_name: Optional[str] = None
        self._warned_no_bus = False

    # ------------------------------------------------------------------ bus
    def _ensure_bus(self):
        if self._bus is not None:
            return self._bus
        if self._gi_error is not None:
            return None
        try:
            import gi
            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib
            self._gi = (Gio, GLib)
            self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            return self._bus
        except Exception as exc:  # pragma: no cover - depends on desktop
            self._gi_error = exc
            if not self._warned_no_bus:
                log.warning("D-Bus session bus unavailable: %s", exc)
                self._warned_no_bus = True
            return None

    def _call(self, bus_name: str, path: str, iface: str, method: str,
              parameters=None, reply_type: Optional[str] = None,
              timeout_ms: int = 2500):
        bus = self._ensure_bus()
        if bus is None:
            return None
        Gio, GLib = self._gi
        try:
            return bus.call_sync(
                bus_name, path, iface, method,
                parameters,
                GLib.VariantType.new(reply_type) if reply_type else None,
                Gio.DBusCallFlags.NONE,
                timeout_ms,
                None,
            )
        except GLib.Error as exc:
            log.debug("D-Bus %s.%s failed: %s", iface, method, exc.message)
            return None
        except Exception as exc:  # pragma: no cover
            log.debug("D-Bus %s.%s failed: %s", iface, method, exc)
            return None

    def name_has_owner(self, name: str) -> Optional[bool]:
        bus = self._ensure_bus()
        if bus is None:
            return None
        Gio, GLib = self._gi
        try:
            reply = bus.call_sync(
                "org.freedesktop.DBus", "/org/freedesktop/DBus",
                "org.freedesktop.DBus", "NameHasOwner",
                GLib.Variant("(s)", (name,)),
                GLib.VariantType.new("(b)"), Gio.DBusCallFlags.NONE, 1500, None,
            )
            return bool(reply.unpack()[0])
        except Exception:
            return None

    # ----------------------------------------------------------------- lock
    def is_locked(self) -> Optional[bool]:
        bus = self._ensure_bus()
        if bus is None:
            return None
        Gio, GLib = self._gi
        reply = self._call(
            SESSION_MANAGER, SESSION_PATH, "org.freedesktop.DBus.Properties",
            "Get", GLib.Variant("(ss)", (SESSION_IFACE, "Locked")), "(v)",
        )
        if reply is None:
            return None
        try:
            value = reply.unpack()
            while isinstance(value, tuple) and len(value) == 1:
                value = value[0]
            return bool(value)
        except Exception:
            return None

    def request_lock(self) -> bool:
        if self.dry_run:
            log.info("DRY-RUN: would lock the screen now")
            return True
        if not self.lock_enabled:
            log.info("auto-lock disabled by configuration; would have locked")
            return True

        reply = self._call(
            SESSION_MANAGER, SESSION_PATH, SESSION_IFACE, "RequestLock", None,
        )
        if reply is not None:
            log.info("lock requested through com.deepin.SessionManager")
            return True

        reply = self._call(
            LOCK_FRONT, LOCK_FRONT_PATH, LOCK_FRONT_IFACE, "Show", None,
        )
        if reply is not None:
            log.info("lock requested through com.deepin.dde.lockFront")
            return True

        log.error("could not request lock through DDE D-Bus")
        return False

    # ---------------------------------------------------------- notification
    def notify(self, summary: str, body: str, replaces_id: int = 0,
               timeout_ms: int = 8000, urgency: int = 1) -> Optional[int]:
        if self.dry_run:
            log.info("DRY-RUN: notification %r / %r", summary, body)
            return 0
        bus = self._ensure_bus()
        if bus is None:
            return None
        Gio, GLib = self._gi

        candidates = [NOTIFICATIONS]
        if self._notification_bus_name:
            candidates = [self._notification_bus_name]
        elif self.name_has_owner(NOTIFICATIONS) is False:
            candidates.append("com.deepin.dde.Notification")

        for bus_name in candidates:
            hints = {
                "urgency": GLib.Variant("y", int(urgency)),
                "category": GLib.Variant("s", "device"),
                "desktop-entry": GLib.Variant("s", "presence-guard"),
            }
            params = GLib.Variant(
                "(susssasa{sv}i)",
                ("presence-guard", int(replaces_id), "dialog-information",
                 summary, body, [], hints, int(timeout_ms)),
            )
            try:
                reply = bus.call_sync(
                    bus_name, NOTIFICATIONS_PATH, NOTIFICATIONS_IFACE, "Notify",
                    params, GLib.VariantType.new("(u)"),
                    Gio.DBusCallFlags.NONE, 2500, None,
                )
                notification_id = int(reply.unpack()[0])
                self._notification_bus_name = bus_name
                log.info("notification shown (id=%s, service=%s)",
                         notification_id, bus_name)
                return notification_id
            except Exception as exc:
                log.debug("notification through %s failed: %s", bus_name, exc)

        log.error("could not show notification through D-Bus")
        return None

    # ----------------------------------------------------------------- misc
    def status(self) -> dict:
        self._ensure_bus()
        return {
            "session_bus": self._bus is not None,
            "session_manager": self.name_has_owner(SESSION_MANAGER),
            "notifications": self.name_has_owner(NOTIFICATIONS),
            "locked": self.is_locked(),
            "dry_run": self.dry_run,
            "lock_enabled": self.lock_enabled,
        }
