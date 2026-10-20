"""Small image-geometry helpers (no third-party dependencies except numpy)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import Box


def clip_box(box: Box, width: int, height: int) -> Optional[Box]:
    """Clip a box to the image and return None when nothing is left."""

    x1 = min(max(box.x1, 0.0), float(width))
    y1 = min(max(box.y1, 0.0), float(height))
    x2 = min(max(box.x2, 0.0), float(width))
    y2 = min(max(box.y2, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return Box(x1, y1, x2, y2)


def expand_box(box: Box, ratio_x: float, ratio_y: Optional[float] = None,
               width: Optional[int] = None, height: Optional[int] = None) -> Box:
    if ratio_y is None:
        ratio_y = ratio_x
    dx = box.width * ratio_x
    dy = box.height * ratio_y
    out = Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy)
    if width is not None and height is not None:
        clipped = clip_box(out, width, height)
        return clipped if clipped is not None else box
    return out


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def contains_point(box: Box, x: float, y: float) -> bool:
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad with black.

    Returns (padded_image, scale, (pad_x, pad_y)).  The same scale is used for
    both axes, which keeps boxes square.
    """

    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        target_h - new_h - pad_y,
        pad_x,
        target_w - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, scale, (pad_x, pad_y)


def unletterbox_box(box: Box, scale: float, pad: tuple[int, int],
                    width: int, height: int) -> Optional[Box]:
    pad_x, pad_y = pad
    mapped = Box(
        (box.x1 - pad_x) / scale,
        (box.y1 - pad_y) / scale,
        (box.x2 - pad_x) / scale,
        (box.y2 - pad_y) / scale,
    )
    return clip_box(mapped, width, height)


def scaled_box(box: Box, scale: float) -> Box:
    return Box(box.x1 * scale, box.y1 * scale, box.x2 * scale, box.y2 * scale)
