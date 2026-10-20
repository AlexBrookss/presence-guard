"""Temporary-file housekeeping.

The guard only writes images when the user explicitly asks for a debug
snapshot (``--save-debug``).  Even so, a long-lived installation should not
accumulate them forever, so every write prunes the directory:

* keep the newest ``keep`` images (0 = no count limit);
* delete images older than ``max_age_days`` (0 = no age limit).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def cleanup_dir(directory: str | Path, keep: int = 30,
                max_age_days: int = 7) -> tuple[int, int]:
    """Delete old debug images. Returns (removed_count, freed_bytes)."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        return 0, 0

    try:
        entries = []
        for path in root.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
    except OSError as exc:
        log.debug("cannot scan cache dir %s: %s", root, exc)
        return 0, 0

    entries.sort(key=lambda item: item[0], reverse=True)
    now = time.time()
    removed = 0
    freed = 0
    for index, (mtime, size, path) in enumerate(entries):
        too_many = keep > 0 and index >= keep
        too_old = max_age_days > 0 and (now - mtime) > max_age_days * 86400
        if not (too_many or too_old):
            continue
        try:
            path.unlink()
            removed += 1
            freed += size
        except OSError as exc:
            log.debug("cannot delete %s: %s", path, exc)
    return removed, freed
