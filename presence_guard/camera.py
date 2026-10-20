"""Short-lived V4L2 camera sessions.

The camera is opened only for the duration of a check (a few hundred
milliseconds to ~1.5 s) and released immediately afterwards.  Between checks
the process is asleep and the camera LED is off.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Camera could not be opened or produced no frames."""


class CameraUnavailable(CameraError):
    """The device is missing/busy; this must never trigger auto-lock."""


def _device_target(device: str):
    if device.isdigit():
        return int(device)
    return device


class CameraCapture:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg

    def _check_device(self) -> None:
        device = self.cfg.device
        if device.startswith("/") and not Path(device).exists():
            raise CameraUnavailable(f"camera device not found: {device}")

    def capture_burst(self, count: int, width: int | None = None,
                      height: int | None = None) -> list[np.ndarray]:
        if count <= 0:
            return []
        self._check_device()
        width = int(width or self.cfg.badge_width)
        height = int(height or self.cfg.badge_height)
        started = time.monotonic()
        cap = cv2.VideoCapture(_device_target(self.cfg.device), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise CameraUnavailable(
                    f"cannot open camera {self.cfg.device} "
                    "(busy or no permission; check the video group)"
                )

            # These properties are best-effort; some UVC cameras reject MJPG.
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.cfg.fourcc[:4].ljust(4)))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        int(self.cfg.open_timeout_seconds * 1000))
            except Exception:
                pass

            frames: list[np.ndarray] = []
            deadline = time.monotonic() + max(1.0, self.cfg.open_timeout_seconds)
            reads = 0
            warmup = max(0, self.cfg.warmup_frames)
            while len(frames) < count and time.monotonic() < deadline:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    time.sleep(0.02)
                    continue
                reads += 1
                if reads <= warmup:
                    continue
                frames.append(frame)

            if not frames:
                raise CameraError(
                    f"camera {self.cfg.device} produced no frames "
                    f"within {self.cfg.open_timeout_seconds:.1f}s"
                )
            log.debug("camera burst: %d frames in %.0f ms",
                      len(frames), (time.monotonic() - started) * 1000.0)
            return frames
        finally:
            cap.release()
