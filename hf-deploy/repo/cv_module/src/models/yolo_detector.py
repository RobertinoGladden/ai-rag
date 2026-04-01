from __future__ import annotations

from typing import List
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from ..config import get_cv_settings
from ..processors.image_preprocessor import ImageInput


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 1), "y1": round(self.y1, 1),
            "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            "width": round(self.width, 1), "height": round(self.height, 1),
        }


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "class_id": self.class_id,
        }


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    model_name: str = ""
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def labels_summary(self) -> dict[str, int]:
        """Ringkasan: {label: count}"""
        summary = {}
        for d in self.detections:
            summary[d.label] = summary.get(d.label, 0) + 1
        return summary

    def filter_by_label(self, label: str) -> List[Detection]:
        return [d for d in self.detections if d.label.lower() == label.lower()]

    def filter_by_confidence(self, min_conf: float) -> List[Detection]:
        return [d for d in self.detections if d.confidence >= min_conf]


class YOLODetector:
    """
    Object detection menggunakan YOLOv8 (Ultralytics).
    Model: yolov8n (nano, cepat) → yolov8m (medium, akurat)
    80 kelas COCO default, bisa di-finetune untuk domain spesifik.
    """

    def __init__(self):
        settings = get_cv_settings()
        logger.info(f"Loading YOLO model: {settings.yolo_model}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(settings.yolo_model)
        except Exception as e:
            logger.error(f"Gagal load YOLO: {e}")
            raise

        self.conf_threshold = settings.yolo_conf_threshold
        self.iou_threshold = settings.yolo_iou_threshold
        self.model_name = settings.yolo_model
        logger.info("YOLO detector ready.")

    def detect(
        self,
        image: ImageInput,
        conf_threshold: float = None,
        classes: List[int] = None,
    ) -> DetectionResult:
        """
        Deteksi objek dalam gambar.

        Args:
            image: ImageInput object
            conf_threshold: Override confidence threshold (default dari config)
            classes: Filter kelas spesifik (COCO class IDs), None = semua kelas

        Returns:
            DetectionResult berisi semua deteksi
        """
        import time
        conf = conf_threshold or self.conf_threshold

        start = time.perf_counter()
        results = self.model.predict(
            source=image.numpy,
            conf=conf,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = self.model.names[cls_id]

                detections.append(Detection(
                    label=label,
                    confidence=conf_val,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=cls_id,
                ))

        logger.debug(
            f"Detected {len(detections)} objects in {elapsed_ms:.1f}ms | "
            f"Summary: {DetectionResult(detections=detections).labels_summary}"
        )

        return DetectionResult(
            detections=detections,
            image_width=image.width,
            image_height=image.height,
            model_name=self.model_name,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def detect_and_annotate(self, image: ImageInput, **kwargs) -> "np.ndarray":
        """
        Detect dan return gambar dengan bounding box yang sudah digambar.
        Berguna untuk visualisasi / demo.
        """
        import cv2

        result_img = image.numpy.copy()
        det_result = self.detect(image, **kwargs)

        for det in det_result.detections:
            bb = det.bbox
            x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)

            # Warna berdasarkan class_id
            color = self._get_color(det.class_id)

            # Gambar bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            label_text = f"{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result_img, label_text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return result_img

    @staticmethod
    def _get_color(class_id: int) -> tuple[int, int, int]:
        """Generate warna konsisten per class_id."""
        palette = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        ]
        return palette[class_id % len(palette)]

    @property
    def available_classes(self) -> dict[int, str]:
        """Return dict semua kelas yang bisa dideteksi."""
        return self.model.names
