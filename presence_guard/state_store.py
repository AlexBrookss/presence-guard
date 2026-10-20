"""Tiny JSON state store (notification cooldown / last check timestamps)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StateStore:
    def __init__(self, state_dir: str, filename: str = "state.json") -> None:
        self.dir = Path(state_dir).expanduser()
        self.path = self.dir / filename
        self.data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                raise ValueError("state root is not an object")
        except FileNotFoundError:
            self.data = {}
        except Exception as exc:
            log.warning("cannot read state %s (%s); starting fresh", self.path, exc)
            try:
                self.path.replace(self.path.with_suffix(".json.bak"))
            except OSError:
                pass
            self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.data.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.data.get(key, default))
        except (TypeError, ValueError):
            return default

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, **values: Any) -> None:
        self.data.update(values)

    def save(self) -> bool:
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".state-", suffix=".json", dir=str(self.dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.data, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
                os.chmod(tmp_name, 0o600)
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return True
        except Exception as exc:
            log.warning("cannot save state %s: %s", self.path, exc)
            return False
