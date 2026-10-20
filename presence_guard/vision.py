"""Detection pipeline: cheap YuNet first, NanoDet only for absence confirmation."""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .badge import analyze_badge_roi, blue_mask, compute_badge_roi, decide
from .config import Config
from .geometry import clip_box, expand_box
from .nanodet import NanoDet
from .types import BadgeDecision, BadgeResult, BlueStats, Box, Detection, PresenceResult

log = logging.getLogger(__name__)


def frame_is_usable(frame: np.ndarray, cfg: Config) -> bool:
    """Reject a covered lens / all-black frame (fail-safe, never lock)."""
    return not (float(frame.mean()) < cfg.detector.min_frame_mean
                and float(frame.std()) < cfg.detector.min_frame_std)


def select_primary_person(detections: list[Detection], shape: tuple[int, int],
                          cfg: Config) -> Optional[Detection]:
    """Pick the largest, most central person; ignore tiny background people."""

    height, width = shape[:2]
    best: Optional[Detection] = None
    best_score = -1.0
    for det in detections:
        if det.class_id != 0:
            continue
        if det.confidence < cfg.detector.person_confidence:
            continue
        box = det.box
        if box.height < cfg.detector.min_person_height_ratio * height:
            continue
        cx, _ = box.center
        if abs(cx - width / 2.0) > cfg.detector.center_zone_ratio * width / 2.0:
            continue
        area_ratio = box.area / float(width * height)
        centrality = 1.0 - abs(cx - width / 2.0) / (width / 2.0)
        score = det.confidence * math.sqrt(max(area_ratio, 0.0)) * (0.5 + 0.5 * centrality)
        if score > best_score:
            best_score = score
            best = det
    return best


def select_face(faces: Optional[np.ndarray], shape: tuple[int, int],
                person: Optional[Box], cfg: Config,
                min_width_ratio: float) -> Optional[Box]:
    """Pick the largest face that belongs to the primary person."""

    if faces is None or len(faces) == 0:
        return None
    height, width = shape[:2]
    best_box: Optional[Box] = None
    best_score = -1.0
    allowed = None
    if person is not None:
        allowed = expand_box(person, 0.25, 0.25, width, height)

    for row in faces:
        x, y, w, h, score = (float(row[0]), float(row[1]), float(row[2]),
                             float(row[3]), float(row[-1]))
        if score < cfg.detector.face_confidence:
            continue
        if w < min_width_ratio * width or h < min_width_ratio * height * 0.7:
            continue
        box = clip_box(Box(x, y, x + w, y + h), width, height)
        if box is None:
            continue
        cx, cy = box.center
        if allowed is not None:
            if not (allowed.x1 <= cx <= allowed.x2 and allowed.y1 <= cy <= allowed.y2):
                continue
        else:
            if abs(cx - width / 2.0) > cfg.detector.center_zone_ratio * width / 2.0:
                continue
        centrality = 1.0 - abs(cx - width / 2.0) / (width / 2.0)
        face_score = score * box.area * (0.5 + 0.5 * centrality)
        if face_score > best_score:
            best_score = face_score
            best_box = box
    return best_box


class VisionPipeline:
    """Loads the models once and runs short inference bursts."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.person_detector = NanoDet(
            str(cfg.resolve_model_path(cfg.detector.nanodet_model)),
            prob_threshold=cfg.detector.person_confidence,
            iou_threshold=0.55,
        )
        yunet_path = str(cfg.resolve_model_path(cfg.detector.yunet_model))
        self.face_detector = cv2.FaceDetectorYN.create(
            yunet_path, "",
            (cfg.camera.badge_width, cfg.camera.badge_height),
            cfg.detector.face_confidence, 0.3, 5000,
        )
        self._face_input_size = (cfg.camera.badge_width, cfg.camera.badge_height)
        cv2.setNumThreads(max(1, cfg.detector.cv_threads))

    def detect_faces(self, frame: np.ndarray) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        if (width, height) != self._face_input_size:
            self.face_detector.setInputSize((width, height))
            self._face_input_size = (width, height)
        try:
            _, faces = self.face_detector.detect(frame)
            return faces
        except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
            log.warning("face detector failed: %s", exc)
            return None

    def analyze_presence(self, frames: list[np.ndarray],
                         person_fallback: bool = True) -> PresenceResult:
        if not frames:
            return PresenceResult(present=False, valid_frames=0,
                                  error="camera returned no frames")

        votes = 0
        valid = 0
        best_conf = 0.0
        best_person: Optional[Box] = None
        best_face: Optional[Box] = None
        errors: list[str] = []

        for frame in frames:
            try:
                if not frame_is_usable(frame, self.cfg):
                    errors.append("frame too dark/uniform")
                    continue
                # Face first: YuNet is an order of magnitude cheaper than
                # NanoDet and the user normally faces the screen.  Only when
                # no face is visible do we pay for full-person detection
                # (e.g. the user turned around).
                faces = self.detect_faces(frame)
                face = select_face(
                    faces, frame.shape[:2], None, self.cfg,
                    self.cfg.detector.face_presence_min_width_ratio,
                )
                if face is not None:
                    valid += 1
                    votes += 1
                    if best_face is None:
                        best_face = face
                        best_conf = max(best_conf, 0.6)
                    continue

                if not person_fallback:
                    # First, cheap pass: no face -> treat as a negative
                    # sample.  NanoDet is only paid for during the 10-second
                    # confirmation sample, not on every 60-second scan.
                    valid += 1
                    continue

                detections = self.person_detector.detect(frame)
                person = select_primary_person(
                    detections, frame.shape[:2], self.cfg)
                valid += 1
                if person is not None:
                    votes += 1
                    if person.confidence > best_conf:
                        best_conf = person.confidence
                        best_person = person.box
                        best_face = None
            except Exception as exc:  # keep checking other frames
                errors.append(str(exc))
                log.warning("presence frame failed: %s", exc)

        present = votes >= max(1, math.ceil(len(frames) / 2.0))
        error = None
        if valid == 0:
            error = "; ".join(errors) or "inference failed"
        return PresenceResult(
            present=present,
            confidence=best_conf,
            person=best_person,
            face=best_face,
            votes=votes,
            valid_frames=valid,
            error=error,
        )

    def analyze_badge(self, frames: list[np.ndarray],
                      presence: PresenceResult) -> BadgeResult:
        stats: list[BlueStats] = []
        for frame in frames:
            try:
                if not frame_is_usable(frame, self.cfg):
                    stats.append(BlueStats(valid=False, reason="too_dark_camera"))
                    continue
                # The badge ROI needs a face scale; NanoDet is unnecessary
                # here.  If the face is not visible the result is unknown,
                # never a notification.
                faces = self.detect_faces(frame)
                face = select_face(
                    faces, frame.shape[:2], None, self.cfg,
                    self.cfg.badge.min_face_width_ratio,
                )
                if face is None:
                    stats.append(BlueStats(valid=False, reason="no_face",
                                           roi=None, face_width=0.0))
                    continue
                roi = compute_badge_roi(
                    frame.shape[:2], None, face, self.cfg.badge)
                stats.append(analyze_badge_roi(
                    frame, roi, face, self.cfg.badge))
            except Exception as exc:
                log.warning("badge frame failed: %s", exc)
                stats.append(BlueStats(valid=False, reason=f"error: {exc}"))

        result = decide(stats, self.cfg.badge)
        if result.valid_frames == 0 and presence.person is not None:
            result.reason = "no usable badge view"
        return result

    def analyze_image(self, image: np.ndarray, check_badge: bool = False
                      ) -> tuple[PresenceResult, Optional[BadgeResult]]:
        presence = self.analyze_presence([image])
        badge = None
        if check_badge and presence.present:
            badge = self.analyze_badge([image], presence)
        return presence, badge

    def annotate(self, frame: np.ndarray, presence: PresenceResult,
                 badge: Optional[BadgeResult] = None,
                 draw_mask: bool = True) -> np.ndarray:
        out = frame.copy()
        if presence.person is not None:
            x1, y1, x2, y2 = presence.person.as_tuple()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(out, f"person {presence.confidence:.2f}", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
        if presence.face is not None:
            x1, y1, x2, y2 = presence.face.as_tuple()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

        if badge is not None and badge.stats:
            idx = badge.best_index
            if idx is None or idx >= len(badge.stats):
                idx = 0
            st = badge.stats[idx]
            if st.roi is not None:
                x1, y1, x2, y2 = st.roi.as_tuple()
                color = (0, 220, 0) if badge.decision == BadgeDecision.PRESENT else (
                    (0, 0, 230) if badge.decision == BadgeDecision.ABSENT else (0, 200, 255))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                if draw_mask:
                    crop = out[y1:y2, x1:x2]
                    if crop.size:
                        mask = blue_mask(frame[y1:y2, x1:x2], self.cfg.badge)
                        tint = np.zeros_like(crop)
                        tint[:, :] = (180, 0, 255)  # magenta overlay
                        tinted = cv2.addWeighted(crop, 1.0, tint, 0.45, 0)
                        crop[mask > 0] = tinted[mask > 0]
                        cv2.putText(
                            out, "magenta = dark-blue candidate pixels (debug only)",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 0, 255), 1, cv2.LINE_AA)
            label = (f"badge {badge.decision.value} "
                     f"ratio={st.blue_ratio:.3f} votes={badge.present_votes}/"
                     f"{badge.valid_frames}")
            cv2.putText(out, label, (10, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        elif presence.present:
            cv2.putText(out, "person present", (10, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)
        return out
