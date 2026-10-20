"""Offline self-test: no camera and no desktop actions are touched."""

from __future__ import annotations

import numpy as np
import cv2

from .badge import analyze_badge_roi, compute_badge_roi, decide
from .config import Config
from .types import BadgeDecision, Box, Detection
from .vision import VisionPipeline, select_primary_person


def _synthetic_frame(with_badge: bool, blue: bool = True) -> tuple[np.ndarray, Box, Box]:
    frame = np.full((480, 640, 3), 45, dtype=np.uint8)
    # torso
    cv2.rectangle(frame, (200, 120), (440, 479), (70, 70, 75), -1)
    # head
    cv2.ellipse(frame, (320, 105), (48, 60), 0, 0, 360, (150, 180, 215), -1)
    person = Box(180, 40, 460, 479)
    face = Box(275, 50, 365, 160)
    if with_badge:
        color = (110, 45, 20) if blue else (30, 30, 200)  # BGR: navy / red
        cv2.rectangle(frame, (300, 185), (345, 235), color, -1)
        cv2.rectangle(frame, (312, 150), (318, 190), color, -1)
    return frame, person, face


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    return bool(condition)


def run_self_test(cfg: Config) -> int:
    ok = True

    # 1. Models load.
    try:
        vision = VisionPipeline(cfg)
        ok &= _check("models load", True)
    except Exception as exc:
        ok &= _check("models load", False, str(exc))
        return 1

    # 2. A blank frame must not crash inference.
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = vision.person_detector.detect(blank)
        ok &= _check("NanoDet blank frame", isinstance(dets, list),
                     f"{len(dets)} detections")
        faces = vision.detect_faces(blank)
        ok &= _check("YuNet blank frame", faces is None or len(faces) == 0)
    except Exception as exc:
        ok &= _check("inference", False, str(exc))

    # 3. Badge ROI geometry.
    frame, person, face = _synthetic_frame(with_badge=True)
    roi = compute_badge_roi(frame.shape[:2], person, face, cfg.badge)
    ok &= _check("badge ROI computed", roi is not None,
                 str(roi.to_list()) if roi else "")

    # 4. Dark blue badge is detected.
    if roi is not None:
        stats = analyze_badge_roi(frame, roi, face, cfg.badge)
        ok &= _check("dark blue detected", stats.valid and stats.present,
                     f"ratio={stats.blue_ratio:.4f} largest={stats.largest_area}")
        result = decide([stats, stats, stats], cfg.badge)
        ok &= _check("3/3 votes -> present", result.decision == BadgeDecision.PRESENT,
                     result.reason)

    # 5. Red patch is not treated as the badge.
    frame2, person2, face2 = _synthetic_frame(with_badge=True, blue=False)
    roi2 = compute_badge_roi(frame2.shape[:2], person2, face2, cfg.badge)
    if roi2 is not None:
        stats2 = analyze_badge_roi(frame2, roi2, face2, cfg.badge)
        result2 = decide([stats2, stats2, stats2], cfg.badge)
        ok &= _check("red patch -> absent", result2.decision == BadgeDecision.ABSENT,
                     f"ratio={stats2.blue_ratio:.4f}")

    # 6. Too few valid frames -> unknown, never notify.
    unknown = decide([], cfg.badge)
    ok &= _check("no usable frames -> unknown",
                 unknown.decision == BadgeDecision.UNKNOWN)

    # 7. Primary-person selection prefers the large central person.
    shape = (480, 640)
    dets = [
        Detection(Box(20, 80, 100, 400), 0.9),      # large but at the edge
        Detection(Box(260, 120, 400, 470), 0.7),    # central
        Detection(Box(600, 300, 620, 340), 0.8),    # tiny
    ]
    primary = select_primary_person(dets, shape, cfg)
    ok &= _check("primary person selection",
                 primary is not None and primary.box.x1 == 260,
                 str(primary.box.to_list()) if primary else "none")

    print()
    print("self-test:", "OK" if ok else "FAILED")
    return 0 if ok else 1
