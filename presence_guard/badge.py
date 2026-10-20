"""Chest-ROI extraction and the "dark blue badge" colour heuristic.

The badge is a **dark blue strip** (lanyard band) hanging from the neck, not
scattered blue dots.  So, besides the colour/geometry filters (close to the
body centre line, dark navy only), the strap mode requires an *elongated*
structure:

1. **Geometry** - only blue blobs in the band under the chin, close to the
   body's vertical centre line, vote.  A blue object at the edge of the chest
   box (dx > ~0.5 face widths) is ignored.
2. **Colour** - the badge navy is dark and saturated; bright blue plastic /
   sky reflections are above the V ceiling.
3. **Shape** - a counted blob whose minimum-area rectangle is long
   (>= 0.45 face heights) and narrow (<= 0.28 face widths) counts as the
   lanyard strip.  Clothing dots are too short and clothing regions are too
   wide, so neither passes.

The "card" mode additionally accepts a compact solid card-shaped blob.
The generous "whole chest is blue" escape hatch (blue clothing) still avoids
nagging someone in a navy shirt.

When the view is too poor to judge (face missing, too dark, too small) the
result is UNKNOWN and no notification is sent.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import BadgeConfig
from .geometry import clip_box, expand_box
from .types import BadgeDecision, BadgeResult, BlueStats, Box


def compute_badge_roi(shape: tuple[int, int], person: Optional[Box],
                      face: Optional[Box], cfg: BadgeConfig) -> Optional[Box]:
    """Return the neck/chest ROI relative to the face.

    A face is required: it gives a stable scale and avoids guessing the chest
    position from a full-body box (which is wrong when the person is seated).
    """

    if face is None:
        return None
    height, width = shape[:2]

    if face.width < cfg.min_face_width_ratio * width:
        return None
    if person is not None:
        # The face must belong to the selected person, with some slack for
        # imperfect person boxes.
        allowed = expand_box(person, 0.25, 0.25, width, height)
        cx, cy = face.center
        if not (allowed.x1 <= cx <= allowed.x2 and allowed.y1 <= cy <= allowed.y2):
            return None

    fw = face.width
    fh = face.height
    cx = (face.x1 + face.x2) / 2.0
    chin = face.y2
    # The lanyard starts under the chin; the card usually sits 1-2 face
    # heights lower.  The ROI is intentionally generous; geometry filtering
    # happens per blob in analyze_badge_roi.
    roi = Box(
        cx - 1.30 * fw,
        chin - 0.10 * fh,
        cx + 1.30 * fw,
        chin + 2.60 * fh,
    )
    roi = clip_box(roi, width, height)
    if roi is None:
        return None
    if roi.width < cfg.min_roi_width or roi.height < cfg.min_roi_height:
        return None
    if roi.area < cfg.min_roi_area:
        return None
    return roi


def blue_mask(roi_bgr: np.ndarray, cfg: BadgeConfig) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.hue_min, cfg.saturation_min, cfg.value_min], dtype=np.uint8)
    upper = np.array([cfg.hue_max, 255, cfg.value_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _blob_is_on_lanyard(blob_box: tuple[int, int, int, int],
                        centroid: tuple[float, float],
                        geometry: "_BadgeGeometry", cfg: BadgeConfig) -> bool:
    """Reject blue blobs that are too far from the neck centre line.

    ``blob_box`` is (x, y, w, h) in ROI coordinates.
    """

    bx, by, bw, bh = blob_box
    cx, cy = centroid
    dx = abs(cx - geometry.center_x) / max(geometry.face_width, 1.0)
    if dx > cfg.max_horizontal_offset_ratio:
        return False

    # Reject tiny specks: clothing dots are small scattered blobs.
    if bw * bh < cfg.min_blue_pixels:
        return False
    return True


class _BadgeGeometry:
    __slots__ = ("center_x", "chin_y", "face_width", "face_height")

    def __init__(self, face: Box, roi: Box) -> None:
        self.center_x = (face.x1 + face.x2) / 2.0 - roi.x1
        self.chin_y = face.y2 - roi.y1
        self.face_width = max(1.0, face.width)
        self.face_height = max(1.0, face.height)


def analyze_badge_roi(frame_bgr: np.ndarray, roi: Optional[Box],
                      face: Optional[Box], cfg: BadgeConfig) -> BlueStats:
    """Measure dark-blue evidence in one frame's chest ROI."""

    if roi is None or face is None:
        return BlueStats(valid=False, reason="no_roi")
    x1, y1, x2, y2 = roi.as_tuple()
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return BlueStats(valid=False, reason="empty_roi", roi=roi,
                         face_width=face.width)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    if mean_v < cfg.min_brightness:
        return BlueStats(valid=False, reason="too_dark", roi=roi,
                         mean_v=mean_v, face_width=face.width)

    mask = blue_mask(crop, cfg)
    total = int(crop.shape[0] * crop.shape[1])
    blue_pixels = int(cv2.countNonZero(mask))

    geometry = _BadgeGeometry(face, roi)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    lanyard_pixels = 0
    lanyard_largest = 0
    card_largest = 0
    lanyard_mask = np.zeros_like(mask)
    strip_detected = False
    max_blob_long = 0.0
    min_blob_short = 1e9
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        blob_box = (int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]))
        if not _blob_is_on_lanyard(blob_box, centroids[i], geometry, cfg):
            continue
        # Bright blue plastic (fans, mugs) is usually much lighter than the
        # navy badge; only the badge rule applies this ceiling.
        blob_mask = mask[blob_box[1]:blob_box[1] + blob_box[3],
                         blob_box[0]:blob_box[0] + blob_box[2]]
        values = hsv[blob_box[1]:blob_box[1] + blob_box[3],
                     blob_box[0]:blob_box[0] + blob_box[2], 2]
        if values.size and float(np.median(values[blob_mask > 0])) > cfg.max_badge_value:
            continue
        lanyard_pixels += area
        lanyard_largest = max(lanyard_largest, area)
        lanyard_mask[blob_box[1]:blob_box[1] + blob_box[3],
                     blob_box[0]:blob_box[0] + blob_box[2]] |= blob_mask

        # Fitted rectangle of the blob: a strap segment is long and narrow.
        # The test applies to a single blob, never across blobs (otherwise a
        # wide shirt region plus a far-away dot would "combine" into a strip).
        points = cv2.findNonZero(blob_mask)
        if points is not None and len(points) >= 5:
            (_, (long, short), _) = cv2.minAreaRect(points)
            if long < short:
                long, short = short, long
            max_blob_long = max(max_blob_long, float(long))
            min_blob_short = min(min_blob_short, float(short))
            if (long >= cfg.min_strip_length_ratio * geometry.face_height
                    and short <= cfg.max_strip_width_ratio * geometry.face_width):
                strip_detected = True

        bw, bh = blob_box[2], blob_box[3]
        aspect = bw / float(max(bh, 1))
        fill = area / float(max(bw * bh, 1))
        # A compact solid rectangle; needed only for badge.mode = "card".
        if (cfg.min_card_aspect <= aspect <= cfg.max_card_aspect
                and fill >= cfg.min_card_fill
                and area >= cfg.min_card_pixels):
            card_largest = max(card_largest, area)

    ratio = lanyard_pixels / float(total) if total else 0.0
    # An all-blue torso is more likely a navy shirt than a missing badge;
    # keep the generous escape hatch so we never nag about clothing.
    all_blue_ratio = blue_pixels / float(total) if total else 0.0

    if cfg.mode == "card":
        # A bare lanyard is not a badge: require a card-shaped blob.
        present = bool(card_largest >= cfg.min_card_pixels)
    else:
        present = bool(
            strip_detected
            or all_blue_ratio >= cfg.blue_clothing_ratio
        )

    return BlueStats(
        valid=True,
        reason="ok",
        roi=roi,
        blue_ratio=ratio,
        blue_pixels=lanyard_pixels,
        largest_area=lanyard_largest,
        mean_v=mean_v,
        face_width=face.width,
        max_blob_long_ratio=(max_blob_long / max(geometry.face_height, 1.0)),
        min_blob_short_ratio=(min_blob_short / max(geometry.face_width, 1.0)
                              if min_blob_short < 1e8 else 0.0),
        all_blue_ratio=all_blue_ratio,
        present=present,
    )


def decide(stats: list[BlueStats], cfg: BadgeConfig) -> BadgeResult:
    valid = [s for s in stats if s.valid]
    if len(valid) < cfg.min_valid_frames:
        return BadgeResult(
            decision=BadgeDecision.UNKNOWN,
            stats=stats,
            valid_frames=len(valid),
            present_votes=sum(1 for s in valid if s.present),
            reason=f"only {len(valid)} valid frame(s), need {cfg.min_valid_frames}",
        )

    votes = sum(1 for s in valid if s.present)
    if votes:
        best_index = max(range(len(stats)),
                         key=lambda i: (stats[i].valid, stats[i].present,
                                        stats[i].largest_area))
    else:
        best_index = max(range(len(stats)),
                         key=lambda i: (stats[i].valid, stats[i].mean_v))

    if votes >= cfg.present_frame_votes:
        decision = BadgeDecision.PRESENT
        best = max(valid, key=lambda s: s.largest_area)
        reason = (f"{votes}/{len(valid)} frames show a dark-blue strip near "
                  f"the neck (largest {best.largest_area} px, "
                  f"blob {best.max_blob_long_ratio:.2f}fh x "
                  f"{best.min_blob_short_ratio:.2f}fw)")
    elif votes == 0:
        decision = BadgeDecision.ABSENT
        reason = f"0/{len(valid)} frames show dark blue near the neck"
    else:
        # One weak sighting is not enough to say "wearing", but it is also not
        # enough to nag.  Loose by design.
        decision = BadgeDecision.UNKNOWN
        reason = f"{votes}/{len(valid)} ambiguous frames"

    return BadgeResult(
        decision=decision,
        stats=stats,
        valid_frames=len(valid),
        present_votes=votes,
        best_index=best_index,
        reason=reason,
    )
