from __future__ import annotations

from typing import List
from dataclasses import dataclass, field
from loguru import logger

import numpy as np

from ..config import get_cv_settings
from ..processors.image_preprocessor import ImageInput


@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: list   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] format EasyOCR

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
        }


@dataclass
class OCRResult:
    full_text: str
    boxes: List[OCRBox] = field(default_factory=list)
    language: str = ""
    engine: str = ""

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class OCRProcessor:
    """
    OCR menggunakan EasyOCR dengan mode stabil (single-pass ringan).
    Fokus: tidak crash di Docker + tetap improve akurasi.
    """

    MIN_CONFIDENCE = 0.10
    MIN_OCR_DIM = 800

    def __init__(self):
        settings = get_cv_settings()
        self.engine = settings.ocr_engine
        self.languages = [l.strip() for l in settings.ocr_languages.split(",")]
        logger.info(f"Loading OCR ({self.engine}) for languages: {self.languages}")

        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=(settings.device == "cuda"),
                model_storage_directory=settings.models_cache_dir,
            )
            logger.info("OCR processor ready.")
        except Exception as e:
            logger.error(f"Gagal init OCR: {e}")
            raise

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocessing ringan:
        - upscale (jika kecil)
        - grayscale
        - CLAHE contrast enhancement
        - light sharpen
        """
        try:
            import cv2

            h, w = img.shape[:2]
            if max(h, w) < self.MIN_OCR_DIM:
                scale = self.MIN_OCR_DIM / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()

            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            logger.warning(f"OCR preprocessing fallback to original image: {e}")
            return img

    def _parse_results(self, raw_results: List) -> List[OCRBox]:
        boxes = []
        for item in raw_results:
            if len(item) == 3:
                bbox, text, confidence = item
            elif len(item) == 2:
                bbox, text = item
                confidence = 0.8
            else:
                continue

            text = str(text).strip()
            if not text or confidence < self.MIN_CONFIDENCE:
                continue

            # Convert numpy scalars/arrays to native Python types for FastAPI/Pydantic serialization
            safe_bbox = []
            try:
                for pt in bbox:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        safe_bbox.append([float(pt[0]), float(pt[1])])
                    else:
                        safe_bbox.append(pt)
            except Exception:
                safe_bbox = bbox

            boxes.append(OCRBox(
                text=text,
                confidence=float(confidence),
                bbox=safe_bbox,
            ))
        return boxes

    def _boxes_to_text(self, boxes: List[OCRBox]) -> str:
        if not boxes:
            return ""

        def cy(box: OCRBox) -> float:
            try:
                ys = [pt[1] for pt in box.bbox]
                return sum(ys) / len(ys)
            except Exception:
                return 0

        def cx(box: OCRBox) -> float:
            try:
                xs = [pt[0] for pt in box.bbox]
                return sum(xs) / len(xs)
            except Exception:
                return 0

        def h(box: OCRBox) -> float:
            try:
                ys = [pt[1] for pt in box.bbox]
                return max(ys) - min(ys)
            except Exception:
                return 20

        sorted_boxes = sorted(boxes, key=lambda b: (cy(b), cx(b)))
        lines = []
        current = [sorted_boxes[0]]
        current_y = cy(sorted_boxes[0])

        for box in sorted_boxes[1:]:
            if abs(cy(box) - current_y) < max(h(box) * 0.5, 15):
                current.append(box)
            else:
                current.sort(key=lambda b: cx(b))
                lines.append(" ".join(b.text for b in current))
                current = [box]
                current_y = cy(box)

        if current:
            current.sort(key=lambda b: cx(b))
            lines.append(" ".join(b.text for b in current))

        return "\n".join(lines)

    def extract_text(
        self,
        image: ImageInput,
        detail: bool = True,
        paragraph: bool = False,
    ) -> OCRResult:
        logger.debug(f"Running stable OCR on {image.width}x{image.height} image")

        try:
            processed = self._preprocess_for_ocr(image.numpy.copy())
            raw_results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                contrast_ths=0.1,
                adjust_contrast=0.7,
                text_threshold=0.5,
                low_text=0.3,
                link_threshold=0.3,
                width_ths=0.7,
                decoder="beamsearch",
                beamWidth=10,
            )
            boxes = self._parse_results(raw_results)

            if len(boxes) < 2:
                raw2 = self.reader.readtext(
                    image.numpy,
                    detail=1,
                    paragraph=False,
                )
                boxes2 = self._parse_results(raw2)
                if len(boxes2) > len(boxes):
                    boxes = boxes2

            full_text = self._boxes_to_text(boxes)

            return OCRResult(
                full_text=full_text,
                boxes=boxes,
                language=",".join(self.languages),
                engine=self.engine,
            )

        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raw_results = self.reader.readtext(image.numpy, detail=1, paragraph=False)
            boxes = self._parse_results(raw_results)
            full_text = self._boxes_to_text(boxes)
            return OCRResult(
                full_text=full_text,
                boxes=boxes,
                language=",".join(self.languages),
                engine=self.engine,
            )

    def extract_text_simple(self, image: ImageInput) -> str:
        result = self.extract_text(image, detail=True, paragraph=False)
        return result.full_text
