from __future__ import annotations

from typing import List
from dataclasses import dataclass, field
from loguru import logger

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
    OCR (Optical Character Recognition) menggunakan EasyOCR.
    Support 80+ bahasa termasuk Indonesia dan Inggris.

    Berguna untuk:
    - Ekstrak teks dari foto dokumen / struk / papan nama
    - Digitalisasi dokumen fisik
    - Preprocessing untuk downstream NLP
    """

    def __init__(self):
        settings = get_cv_settings()
        self.engine = settings.ocr_engine
        self.languages = [l.strip() for l in settings.ocr_languages.split(",")]
        logger.info(f"Loading OCR ({self.engine}) for languages: {self.languages}")

        try:
            import easyocr
            # gpu=False untuk CPU-only; set True jika ada GPU
            self.reader = easyocr.Reader(
                self.languages,
                gpu=(settings.device == "cuda"),
                model_storage_directory=settings.models_cache_dir,
            )
            logger.info("OCR processor ready.")
        except Exception as e:
            logger.error(f"Gagal init OCR: {e}")
            raise

    def extract_text(
        self,
        image: ImageInput,
        detail: bool = True,
        paragraph: bool = True,
    ) -> OCRResult:
        """
        Ekstrak semua teks dari gambar.

        Args:
            image: ImageInput object
            detail: Jika True, return bounding box per word
            paragraph: Jika True, gabungkan teks yang berdekatan

        Returns:
            OCRResult berisi full_text dan detail per bounding box
        """
        logger.debug(f"Running OCR on {image.width}x{image.height} image")

        raw_results = self.reader.readtext(
            image.numpy,
            detail=1,
            paragraph=paragraph,
        )

        boxes = []
        text_lines = []

        for item in raw_results:
            bbox, text, confidence = item
            if confidence < 0.1 or not text.strip():
                continue
            boxes.append(OCRBox(
                text=text.strip(),
                confidence=confidence,
                bbox=bbox,
            ))
            text_lines.append(text.strip())

        full_text = "\n".join(text_lines)

        logger.debug(f"OCR extracted {len(text_lines)} text segments, {len(full_text)} chars")

        return OCRResult(
            full_text=full_text,
            boxes=boxes,
            language=",".join(self.languages),
            engine=self.engine,
        )

    def extract_text_simple(self, image: ImageInput) -> str:
        """Shorthand — return hanya string teks tanpa detail."""
        result = self.extract_text(image, detail=True, paragraph=True)
        return result.full_text
