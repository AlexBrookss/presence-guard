"""Shared value objects for presence-guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates (x1, y1) - (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    def to_list(self) -> list[float]:
        return [round(float(v), 2) for v in (self.x1, self.y1, self.x2, self.y2)]


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    class_id: int = 0


@dataclass
class PresenceResult:
    """Result of one presence check (a short burst of frames)."""

    present: bool
    confidence: float = 0.0
    person: Optional[Box] = None
    face: Optional[Box] = None
    votes: int = 0
    valid_frames: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "confidence": round(float(self.confidence), 3),
            "person": self.person.to_list() if self.person else None,
            "face": self.face.to_list() if self.face else None,
            "votes": self.votes,
            "valid_frames": self.valid_frames,
            "error": self.error,
        }


class BadgeDecision(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass
class BlueStats:
    """Colour statistics of one chest ROI."""

    valid: bool
    reason: str = ""
    roi: Optional[Box] = None
    blue_ratio: float = 0.0
    blue_pixels: int = 0
    largest_area: int = 0
    mean_v: float = 0.0
    face_width: float = 0.0
    max_blob_long_ratio: float = 0.0
    min_blob_short_ratio: float = 0.0
    all_blue_ratio: float = 0.0
    present: bool = False

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "roi": self.roi.to_list() if self.roi else None,
            "blue_ratio": round(float(self.blue_ratio), 4),
            "blue_pixels": int(self.blue_pixels),
            "largest_area": int(self.largest_area),
            "mean_v": round(float(self.mean_v), 1),
            "face_width": round(float(self.face_width), 1),
            "max_blob_long_ratio": round(float(self.max_blob_long_ratio), 3),
            "min_blob_short_ratio": round(float(self.min_blob_short_ratio), 3),
            "all_blue_ratio": round(float(self.all_blue_ratio), 4),
            "present": self.present,
        }


@dataclass
class BadgeResult:
    decision: BadgeDecision
    stats: list[BlueStats] = field(default_factory=list)
    valid_frames: int = 0
    present_votes: int = 0
    best_index: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "valid_frames": self.valid_frames,
            "present_votes": self.present_votes,
            "reason": self.reason,
            "frames": [s.to_dict() for s in self.stats],
        }


@dataclass
class CycleReport:
    """Result of a single scheduler cycle."""

    timestamp: str
    presence: Optional[PresenceResult] = None
    badge: Optional[BadgeResult] = None
    locked: bool = False
    action: str = "none"
    error: Optional[str] = None
    state: str = "unknown"
    state_text: str = ""
    elapsed_ms: float = 0.0
    cpu_ms: float = 0.0
    idle_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "locked": self.locked,
            "action": self.action,
            "state": self.state,
            "state_text": self.state_text,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cpu_ms": round(self.cpu_ms, 1),
            "idle_seconds": (round(self.idle_seconds, 1)
                             if self.idle_seconds is not None else None),
            "presence": self.presence.to_dict() if self.presence else None,
            "badge": self.badge.to_dict() if self.badge else None,
        }
