"""Stage 1: drone detection with Ultralytics YOLOv8.

``YoloDetector`` wraps one YOLO model. ``FusionDetector`` runs several models
on the same frame and merges their boxes with weighted box fusion. Fusion is
optional (``--fusion``); the default pipeline uses a single YOLOv8n model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iamars import config
from iamars.weights import get_weights


@dataclass
class Detections:
    """Boxes found in one frame.

    xyxy : (N, 4) float32, pixel corners [x1, y1, x2, y2]
    conf : (N,)   float32, confidence in [0, 1]
    """

    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))

    def __len__(self) -> int:
        return len(self.conf)


class YoloDetector:
    """One YOLO model. ``detect(frame)`` returns :class:`Detections`."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        weights_path: str | None = None,
        conf: float = config.DETECT_CONF,
        iou: float = config.DETECT_IOU,
        imgsz: int = config.IMG_SIZE,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # imported here so tests can run without it

        self.name = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or config.auto_device()
        self.model = YOLO(str(weights_path or get_weights(model)))

    def detect(self, frame: np.ndarray) -> Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return Detections()
        return Detections(
            xyxy=result.boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=result.boxes.conf.cpu().numpy().astype(np.float32),
        )


class FusionDetector:
    """Several YOLO models merged with weighted box fusion (WBF).

    Models run one after another. The old version used Python threads, but
    the models share one GPU, so threads did not make it faster.
    """

    def __init__(
        self,
        models: tuple[str, ...] = ("visiodect", "uav_rgb", "uav_ir"),
        conf: float = config.DETECT_CONF,
        device: str | None = None,
        fusion_iou: float = 0.55,
    ):
        from iamars.fusion import weighted_box_fusion

        self._fuse = weighted_box_fusion
        self.detectors = [YoloDetector(m, conf=conf, device=device) for m in models]
        self.weights = [config.FUSION_WEIGHTS.get(m, 1.0) for m in models]
        self.fusion_iou = fusion_iou
        self.conf = conf
        self.name = "+".join(models)

    def detect(self, frame: np.ndarray) -> Detections:
        per_model = [d.detect(frame) for d in self.detectors]
        h, w = frame.shape[:2]
        boxes, scores = self._fuse(
            [d.xyxy for d in per_model],
            [d.conf for d in per_model],
            self.weights,
            image_wh=(w, h),
            iou_thr=self.fusion_iou,
            skip_box_thr=self.conf,
        )
        return Detections(xyxy=boxes, conf=scores)


def build_detector(fusion: bool = False, device: str | None = None, **kwargs):
    """Factory used by scripts and the app."""
    if fusion:
        return FusionDetector(device=device, **kwargs)
    return YoloDetector(device=device, **kwargs)
