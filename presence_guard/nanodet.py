"""NanoDet inference through OpenCV DNN.

The post-processing follows the OpenCV Zoo NanoDet demo, which is licensed
under Apache-2.0:

    https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet

Only the small wrapper around it (letterbox mapping / typed results) is new.
The bundled model is NanoDet-m-plus-1.5x-416 (COCO, person class = 0).
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import letterbox, unletterbox_box
from .types import Box, Detection

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class NanoDet:
    """A minimal, CPU-friendly person/object detector."""

    def __init__(self, model_path: str, prob_threshold: float = 0.30,
                 iou_threshold: float = 0.55) -> None:
        self.prob_threshold = float(prob_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = (416, 416)
        # This exported model has three feature levels: 52x52, 26x26, 13x13.
        self.strides = (8, 16, 32)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self._anchors = self._build_anchors()

    def _build_anchors(self) -> list[np.ndarray]:
        anchors = []
        for stride in self.strides:
            feat_h = self.input_size[0] // stride
            feat_w = self.input_size[1] // stride
            shift_x = np.arange(0, feat_w, dtype=np.float32) * stride
            shift_y = np.arange(0, feat_h, dtype=np.float32) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.reshape(-1) + 0.5 * (stride - 1)
            cy = yv.reshape(-1) + 0.5 * (stride - 1)
            anchors.append(np.column_stack((cx, cy)).astype(np.float32))
        return anchors

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        img = rgb.astype(np.float32)
        img = (img - self.mean) / self.std
        return cv2.dnn.blobFromImage(img)

    def _postprocess(self, outputs: list[np.ndarray]) -> list[Detection]:
        # The export alternates classification and box-regression tensors.
        cls_scores: list[np.ndarray] = []
        bbox_preds: list[np.ndarray] = []
        for out in outputs:
            if out.ndim == 3:
                out = out.squeeze(axis=0)
            if out.ndim != 2:
                continue
            if out.shape[1] == len(COCO_CLASSES):
                cls_scores.append(out)
            elif out.shape[1] == 4 * (self.reg_max + 1):
                bbox_preds.append(out)

        if not cls_scores or len(cls_scores) != len(bbox_preds):
            raise RuntimeError(
                f"unexpected NanoDet output shapes: {[o.shape for o in outputs]}"
            )

        boxes_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        for stride, cls_score, bbox_pred, anchors in zip(
                self.strides, cls_scores, bbox_preds, self._anchors):
            if cls_score.shape[0] != anchors.shape[0]:
                # Fall back to the actual grid size if the export changes.
                stride = int(round((self.input_size[0] * self.input_size[1]
                                    / cls_score.shape[0]) ** 0.5))
                feat_w = self.input_size[1] // stride
                feat_h = self.input_size[0] // stride
                shift_x = np.arange(0, feat_w, dtype=np.float32) * stride
                shift_y = np.arange(0, feat_h, dtype=np.float32) * stride
                xv, yv = np.meshgrid(shift_x, shift_y)
                anchors = np.column_stack((
                    xv.reshape(-1) + 0.5 * (stride - 1),
                    yv.reshape(-1) + 0.5 * (stride - 1),
                )).astype(np.float32)

            dist = bbox_pred.reshape(-1, self.reg_max + 1)
            dist = np.exp(dist - dist.max(axis=1, keepdims=True))
            dist = dist / dist.sum(axis=1, keepdims=True)
            dist = np.dot(dist, self.project).reshape(-1, 4) * stride

            x1 = anchors[:, 0] - dist[:, 0]
            y1 = anchors[:, 1] - dist[:, 1]
            x2 = anchors[:, 0] + dist[:, 2]
            y2 = anchors[:, 1] + dist[:, 3]
            boxes_all.append(np.column_stack((x1, y1, x2, y2)))
            scores_all.append(cls_score)

        boxes = np.concatenate(boxes_all, axis=0)
        scores = np.concatenate(scores_all, axis=0)
        if scores.size == 0:
            return []

        # Apply NMS per class (COCO has 80 classes).
        best_class = scores.argmax(axis=1)
        best_score = scores[np.arange(scores.shape[0]), best_class]
        keep_mask = best_score >= self.prob_threshold
        if not np.any(keep_mask):
            return []
        boxes = boxes[keep_mask]
        best_score = best_score[keep_mask]
        best_class = best_class[keep_mask]

        wh = boxes.copy()
        wh[:, 2:4] = wh[:, 2:4] - wh[:, 0:2]
        indices = cv2.dnn.NMSBoxes(
            wh.tolist(), best_score.tolist(), self.prob_threshold, self.iou_threshold
        )
        if indices is None or len(indices) == 0:
            return []
        indices = np.asarray(indices).reshape(-1)

        # Letterbox scale/pad are applied by the caller; here boxes are in the
        # 416x416 input space, so multiply by 1/scale and subtract the pad.
        return [
            Detection(
                box=Box(float(boxes[i, 0]), float(boxes[i, 1]),
                        float(boxes[i, 2]), float(boxes[i, 3])),
                confidence=float(best_score[i]),
                class_id=int(best_class[i]),
            )
            for i in indices
        ]

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        padded, scale, pad = letterbox(rgb, self.input_size)
        blob = self._preprocess(padded)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        raw = self._postprocess(outputs)
        result: list[Detection] = []
        for det in raw:
            mapped = unletterbox_box(det.box, scale, pad,
                                     image_bgr.shape[1], image_bgr.shape[0])
            if mapped is not None:
                result.append(Detection(mapped, det.confidence, det.class_id))
        result.sort(key=lambda d: d.confidence, reverse=True)
        return result
