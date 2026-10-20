"""Configuration loading and validation.

Python 3.10 is the target (UOS 20), so ``tomli`` is used instead of the
stdlib ``tomllib``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore


class ConfigError(ValueError):
    pass


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    / "presence-guard" / "config.toml"
)


@dataclass
class GeneralConfig:
    log_level: str = "INFO"
    state_dir: str = "~/.local/state/presence-guard"
    notify_on_camera_error: bool = False
    # os.nice() level; systemd also sets Nice=19.
    nice: int = 19
    # Debug snapshots are only written by --save-debug; prune them anyway.
    debug_dir: str = "~/.cache/presence-guard/debug"
    debug_keep_files: int = 30
    debug_max_age_days: int = 7


@dataclass
class CameraConfig:
    device: str = "/dev/video0"
    # Low resolution for the every-minute presence scan.
    presence_width: int = 320
    presence_height: int = 240
    # Higher resolution only for the badge check (a few times per window).
    badge_width: int = 640
    badge_height: int = 480
    fourcc: str = "MJPG"
    fps: int = 10
    warmup_frames: int = 2
    open_timeout_seconds: float = 8.0
    presence_frames: int = 1
    badge_frames: int = 5


@dataclass
class DetectorConfig:
    nanodet_model: str = "models/nanodet.onnx"
    yunet_model: str = "models/yunet.onnx"
    person_confidence: float = 0.30
    face_confidence: float = 0.60
    min_person_height_ratio: float = 0.18
    center_zone_ratio: float = 0.90
    face_presence_min_width_ratio: float = 0.045
    # A covered lens / black frame must never trigger auto-lock.
    min_frame_mean: float = 10.0
    min_frame_std: float = 5.0
    cv_threads: int = 1


@dataclass
class BadgeConfig:
    enabled: bool = True
    # Monday=0 ... Sunday=6 (fallback when holiday data is unavailable)
    workdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # "chinese": use the chinesecalendar package (public holidays + make-up
    # workdays); "weekdays": plain Monday..Friday.
    holiday_mode: str = "chinese"
    # Local-time windows when badge checks are performed.
    windows: list[str] = field(default_factory=lambda: ["09:30-10:00", "13:00-13:30"])
    interval_seconds: int = 180
    # OpenCV HSV ranges: H 0-180, S/V 0-255.
    hue_min: int = 95
    hue_max: int = 140
    saturation_min: int = 40
    value_min: int = 12
    value_max: int = 220
    min_roi_width: int = 56
    min_roi_height: int = 56
    min_roi_area: int = 3600
    min_face_width_ratio: float = 0.055
    min_brightness: float = 32.0
    min_blue_ratio: float = 0.006
    # Minimum bounding-box area of a blue blob (filter for clothing specks).
    min_blue_pixels: int = 300
    # Strip detection: the badge is a long, narrow dark-blue band (lanyard),
    # not scattered blue dots.
    min_strip_length_ratio: float = 0.45
    max_strip_width_ratio: float = 0.28
    # Ignore blue blobs whose centre is further than N face-widths from the
    # body centre line (fans, mugs, chairs at the edge of the chest box).
    max_horizontal_offset_ratio: float = 0.50
    # Navy badge is dark; brighter blue plastic/sky is not the badge.
    max_badge_value: int = 150
    # "strap" (default): a blue lanyard alone counts as wearing the badge.
    # "card": require a card-shaped blob, so a bare lanyard is not enough.
    mode: str = "strap"
    blue_clothing_ratio: float = 0.55
    # Card-shape constraints (only used when mode = "card").
    min_card_aspect: float = 0.45
    max_card_aspect: float = 1.15
    min_card_fill: float = 0.55
    min_card_pixels: int = 220
    min_valid_frames: int = 3
    present_frame_votes: int = 2
    # Up to N reminders per check window (09:30-10:00, 13:00-13:30).
    max_reminders_per_window: int = 3
    # Spread the reminders; 600 s means the 3rd one lands near the end of a
    # 30-minute window instead of firing three times in 9 minutes.
    notify_cooldown_seconds: int = 600
    notify_title: str = "工牌提醒"
    notify_body: str = (
        "检测到您在座位上，但没有识别到深蓝色工牌，请确认是否佩戴。"
    )
    notify_timeout_ms: int = 8000


@dataclass
class SchedulerConfig:
    presence_interval_seconds: int = 60
    absence_confirm_seconds: int = 10
    absence_recheck_seconds: int = 10
    # Skip the camera entirely while the user is actively typing/clicking.
    # 0 disables the idle gate (always scan every interval).
    idle_gate_seconds: float = 10.0
    idle_provider: str = "auto"  # auto | xprintidle | disabled
    startup_delay_seconds: int = 5
    locked_poll_seconds: int = 60
    camera_error_retry_seconds: int = 300
    camera_error_notify_cooldown_seconds: int = 3600
    lock_enabled: bool = True
    # On non-workdays (weekends / public holidays) only monitor; stay unlocked.
    lock_on_holidays: bool = False


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    badge: BadgeConfig = field(default_factory=BadgeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    config_path: Optional[str] = None
    app_root: str = str(APP_ROOT)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def resolve_model_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.app_root) / path
        return path


def _apply_section(obj: Any, data: dict[str, Any], section: str) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown key [{section}] {key!r}")
        setattr(obj, key, value)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        return _toml.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except Exception as exc:  # tomli.TOMLDecodeError and OSError
        raise ConfigError(f"cannot read config {path}: {exc}") from exc


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    config_path: Optional[Path] = None
    if path:
        config_path = Path(path).expanduser()
    elif DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH

    if config_path is not None:
        data = _load_toml(config_path)
        for section in ("general", "camera", "detector", "badge", "scheduler"):
            values = data.pop(section, {})
            if not isinstance(values, dict):
                raise ConfigError(f"[{section}] must be a table")
            _apply_section(getattr(cfg, section), values, section)
        if data:
            raise ConfigError(f"unknown top-level section(s): {', '.join(data)}")
        cfg.config_path = str(config_path)

    cfg.general.state_dir = str(Path(cfg.general.state_dir).expanduser())
    _validate(cfg)
    return cfg


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def set_config_value(path: str | Path, section: str, key: str,
                     value: object) -> bool:
    """Update one key of a TOML config in place (create it if missing).

    Returns True when the file was written.
    """

    import re

    text = ""
    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except OSError:
        return False

    if isinstance(value, str):
        rendered = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)

    lines = text.splitlines()
    current = None
    done = False
    section_header_index: int | None = None
    out: list[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\[(.+)\]$", line.strip())
        if match:
            current = match.group(1)
            if current == section:
                section_header_index = len(out)
        if current == section and re.match(rf"^{key}\s*=", line):
            out.append(f"{key} = {rendered}")
            done = True
            continue
        out.append(line)
    if not done:
        if section_header_index is not None:
            # Insert right after the existing section header.
            out.insert(section_header_index + 1, f"{key} = {rendered}")
        else:
            if out and out[-1].strip() == "":
                out.pop()
            out.append(f"[{section}]")
            out.append(f"{key} = {rendered}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_text("\n".join(out) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def validate_config(cfg: Config) -> None:
    _validate(cfg)


def _validate(cfg: Config) -> None:
    cam = cfg.camera
    det = cfg.detector
    badge = cfg.badge
    sch = cfg.scheduler

    _require(cam.presence_width >= 160 and cam.presence_height >= 120,
             "camera presence resolution is too small")
    _require(cam.badge_width >= 320 and cam.badge_height >= 240,
             "camera badge resolution is too small")
    _require(cam.fps > 0, "camera.fps must be > 0")
    _require(cam.presence_frames >= 1, "camera.presence_frames must be >= 1")
    _require(cam.badge_frames >= 3, "camera.badge_frames must be >= 3")
    _require(cam.warmup_frames >= 0, "camera.warmup_frames must be >= 0")
    _require(cam.open_timeout_seconds > 0, "camera.open_timeout_seconds must be > 0")

    _require(0.0 < det.person_confidence < 1.0,
             "detector.person_confidence must be in (0, 1)")
    _require(0.0 < det.face_confidence < 1.0,
             "detector.face_confidence must be in (0, 1)")
    _require(0.0 < det.min_person_height_ratio <= 1.0,
             "detector.min_person_height_ratio must be in (0, 1]")
    _require(0.0 < det.center_zone_ratio <= 1.0,
             "detector.center_zone_ratio must be in (0, 1]")
    _require(det.min_frame_mean >= 0, "detector.min_frame_mean must be >= 0")
    _require(det.min_frame_std >= 0, "detector.min_frame_std must be >= 0")
    _require(det.cv_threads >= 1, "detector.cv_threads must be >= 1")

    _require(0 <= badge.hue_min < badge.hue_max <= 180,
             "badge hue range must satisfy 0 <= hue_min < hue_max <= 180")
    _require(0 <= badge.saturation_min <= 255, "badge.saturation_min is out of range")
    _require(0 <= badge.value_min < badge.value_max <= 255,
             "badge value range must satisfy 0 <= value_min < value_max <= 255")
    _require(badge.min_roi_width >= 8 and badge.min_roi_height >= 8,
             "badge ROI limits are too small")
    _require(0.0 <= badge.min_blue_ratio <= 1.0, "badge.min_blue_ratio is out of range")
    _require(badge.min_blue_pixels >= 0, "badge.min_blue_pixels must be >= 0")
    _require(0.1 <= badge.min_strip_length_ratio <= 1.0,
             "badge.min_strip_length_ratio must be in [0.1, 1.0]")
    _require(0.1 <= badge.max_strip_width_ratio <= 1.0,
             "badge.max_strip_width_ratio must be in [0.1, 1.0]")
    _require(0.0 < badge.min_face_width_ratio < 1.0,
             "badge.min_face_width_ratio must be in (0, 1)")
    _require(0.0 < badge.max_horizontal_offset_ratio <= 1.5,
             "badge.max_horizontal_offset_ratio must be in (0, 1.5]")
    _require(badge.value_min < badge.max_badge_value <= 255,
             "badge.max_badge_value must be > value_min and <= 255")
    _require(badge.mode in ("strap", "card"),
             'badge.mode must be "strap" or "card"')
    _require(1 <= badge.min_valid_frames <= cam.badge_frames,
             "badge.min_valid_frames must be between 1 and camera.badge_frames")
    _require(1 <= badge.present_frame_votes <= badge.min_valid_frames,
             "badge.present_frame_votes must be <= min_valid_frames")
    _require(badge.notify_cooldown_seconds >= 0,
             "badge.notify_cooldown_seconds must be >= 0")
    _require(badge.max_reminders_per_window >= 1,
             "badge.max_reminders_per_window must be >= 1")
    _require(bool(badge.workdays), "badge.workdays must not be empty")
    _require(all(0 <= d <= 6 for d in badge.workdays),
             "badge.workdays must contain values 0..6 (Monday=0)")
    _require(badge.holiday_mode in ("chinese", "weekdays"),
             'badge.holiday_mode must be "chinese" or "weekdays"')
    _require(bool(badge.windows), "badge.windows must not be empty")
    _require(badge.interval_seconds >= 30,
             "badge.interval_seconds must be >= 30")
    from .schedule import parse_window
    for window in badge.windows:
        try:
            parse_window(window)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    _require(sch.presence_interval_seconds >= 1,
             "scheduler.presence_interval_seconds must be >= 1")
    _require(sch.absence_confirm_seconds >= 1,
             "scheduler.absence_confirm_seconds must be >= 1")
    _require(sch.absence_recheck_seconds >= 1,
             "scheduler.absence_recheck_seconds must be >= 1")
    _require(sch.idle_gate_seconds >= 0,
             "scheduler.idle_gate_seconds must be >= 0")
    _require(sch.idle_provider in ("auto", "xprintidle", "disabled"),
             "scheduler.idle_provider must be auto, xprintidle or disabled")
    _require(-20 <= cfg.general.nice <= 19,
             "general.nice must be between -20 and 19")
    _require(cfg.general.debug_keep_files >= 0,
             "general.debug_keep_files must be >= 0")
    _require(cfg.general.debug_max_age_days >= 0,
             "general.debug_max_age_days must be >= 0")
    _require(sch.startup_delay_seconds >= 0,
             "scheduler.startup_delay_seconds must be >= 0")
    _require(sch.locked_poll_seconds >= 5,
             "scheduler.locked_poll_seconds must be >= 5")
    _require(sch.camera_error_retry_seconds >= 30,
             "scheduler.camera_error_retry_seconds must be >= 30")
    _require(isinstance(sch.lock_on_holidays, bool),
             "scheduler.lock_on_holidays must be a boolean")
