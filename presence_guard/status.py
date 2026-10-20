"""Status publishing and a tiny local control channel.

The guard daemon writes ``status.json`` after every cycle and listens on a
Unix socket for UI commands.  The GUI is a separate process: if it crashes,
detection and locking keep running.

Wire protocol (one JSON object per line)::

    -> {"cmd": "check_now"}
    <- {"ok": true, "state": "present"}

Commands: ``check_now``, ``pause`` (``seconds``), ``resume``, ``ping``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import socket
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

MAX_EVENTS = 40


def _fmt(ts: float, with_date: bool = False) -> str:
    fmt = "%m-%d %H:%M:%S" if with_date else "%H:%M:%S"
    return _dt.datetime.fromtimestamp(ts).strftime(fmt)


class StatusStore:
    """Atomic JSON status file + in-memory event ring."""

    def __init__(self, state_dir: str, filename: str = "status.json") -> None:
        self.dir = Path(state_dir).expanduser()
        self.path = self.dir / filename
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self.data: dict[str, Any] = {}
        self._load_events()

    def _load_events(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            events = payload.get("events") or []
            for item in events[-MAX_EVENTS:]:
                if isinstance(item, dict) and "text" in item:
                    self.events.appendleft(item)
        except FileNotFoundError:
            return
        except Exception:
            return

    def add_event(self, text: str, level: str = "info") -> None:
        now = time.time()
        self.events.appendleft({
            "ts": now,
            "time": _fmt(now, with_date=True),
            "text": text,
            "level": level,
        })

    def publish(self, **fields: Any) -> None:
        now = time.time()
        data = dict(fields)
        data["updated_at"] = now
        data["updated_at_text"] = _fmt(now, with_date=True)
        data["events"] = list(self.events)
        self.data = data
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp = tempfile.mkstemp(prefix=".status-", suffix=".json",
                                       dir=str(self.dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as exc:
            log.warning("cannot write status file %s: %s", self.path, exc)

    @staticmethod
    def read(state_dir: str) -> Optional[dict[str, Any]]:
        path = Path(state_dir).expanduser() / "status.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.debug("cannot read status file: %s", exc)
            return None


class ControlServer(threading.Thread):
    """Unix-socket command listener running in the daemon."""

    def __init__(self, state_dir: str,
                 handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        super().__init__(name="presence-guard-control", daemon=True)
        self.dir = Path(state_dir).expanduser()
        self.path = self.dir / "control.sock"
        self.handler = handler
        self._stop = threading.Event()
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:  # noqa: D401 - Thread.start signature
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(4)
        sock.settimeout(1.0)
        self._sock = sock
        super().start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(2.0)
                raw = b""
                while b"\n" not in raw and len(raw) < 4096:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    raw += chunk
                request: dict[str, Any] = {}
                if raw.strip():
                    try:
                        request = json.loads(raw.split(b"\n", 1)[0])
                    except Exception:
                        request = {"cmd": ""}
                response = self.handler(request) if isinstance(request, dict) else {}
                conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode())
            except Exception as exc:
                log.debug("control request failed: %s", exc)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass


class ControlClient:
    def __init__(self, state_dir: str) -> None:
        self.path = Path(state_dir).expanduser() / "control.sock"

    def send(self, cmd: str, timeout: float = 3.0, **payload: Any) -> dict[str, Any]:
        if not self.path.exists():
            return {"ok": False, "error": "control socket not found (service stopped?)"}
        message = dict(payload)
        message["cmd"] = cmd
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(str(self.path))
                sock.sendall((json.dumps(message) + "\n").encode())
                raw = b""
                while b"\n" not in raw and len(raw) < 65536:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                return json.loads(raw.split(b"\n", 1)[0].decode())
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
