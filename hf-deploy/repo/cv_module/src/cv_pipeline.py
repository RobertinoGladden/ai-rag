from __future__ import annotations

import time
from typing import List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
from loguru import logger

from .config import get_cv_settings
from .processors.image_preprocessor import ImagePreprocessor, ImageInput
from .models.clip_model import CLIPModel, CLIPResult
from .models.yolo_detector import YOLODetector, DetectionResult
from .models.captioner import ImageCaptioner, CaptionResult
from .processors.ocr_processor import OCRProcessor, OCRResult


@dataclass
class CVAnalysisResult:
    """Hasil lengkap analisis gambar dari semua model."""

    # Info gambar
    image_width: int = 0
    image_height: int = 0
    source: str = ""

    # Per-model results (None jika tidak dijalankan)
    caption: Optional[CaptionResult] = None
    detections: Optional[DetectionResult] = None
    classification: Optional[CLIPResult] = None
    ocr: Optional[OCRResult] = None

    # Metadata
    models_used: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_summary(self) -> str:
        """
        Buat ringkasan teks dari hasil analisis.
        Berguna sebagai input ke LLM (integrasi dengan RAG module).
        """
        parts = []

        if self.caption:
            parts.append(f"Deskripsi gambar: {self.caption.caption}")

        if self.detections and self.detections.count > 0:
            summary = self.detections.labels_summary
            items = ", ".join(f"{count}x {label}" for label, count in summary.items())
            parts.append(f"Objek terdeteksi: {items}")

        if self.classification:
            parts.append(
                f"Klasifikasi: {self.classification.top_label} "
                f"(confidence: {self.classification.top_score:.1%})"
            )

        if self.ocr and self.ocr.full_text:
            preview = self.ocr.full_text[:300]
            if len(self.ocr.full_text) > 300:
                preview += "..."
            parts.append(f"Teks dalam gambar: {preview}")

        return "\n".join(parts) if parts else "Tidak ada informasi yang bisa diekstrak."


class CVPipeline:
    """
    Orchestrator untuk semua CV models.
    Lazy loading — model hanya di-load saat pertama kali dipakai.
    Support modular: bisa run satu atau semua model sekaligus.
    """

    def __init__(self):
        self.settings = get_cv_settings()
        self._clip: Optional[CLIPModel] = None
        self._yolo: Optional[YOLODetector] = None
        self._captioner: Optional[ImageCaptioner] = None
        self._ocr: Optional[OCRProcessor] = None
        self._setup_mlflow()
        logger.info("CVPipeline initialized (lazy loading).")

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

    # === Lazy loaders ===

    @property
    def clip(self) -> CLIPModel:
        if self._clip is None:
            self._clip = CLIPModel()
        return self._clip

    @property
    def yolo(self) -> YOLODetector:
        if self._yolo is None:
            self._yolo = YOLODetector()
        return self._yolo

    @property
    def captioner(self) -> ImageCaptioner:
        if self._captioner is None:
            self._captioner = ImageCaptioner()
        return self._captioner

    @property
    def ocr(self) -> OCRProcessor:
        if self._ocr is None:
            self._ocr = OCRProcessor()
        return self._ocr

    # === Main analysis methods ===

    def analyze(
        self,
        source: Union[str, bytes, Path],
        run_caption: bool = True,
        run_detection: bool = True,
        run_ocr: bool = False,
        classification_labels: Optional[List[str]] = None,
    ) -> CVAnalysisResult:
        """
        Full pipeline analisis gambar.

        Args:
            source: Path, bytes, URL, atau base64 string
            run_caption: Generate image caption dengan BLIP
            run_detection: Deteksi objek dengan YOLO
            run_ocr: Ekstrak teks dengan EasyOCR
            classification_labels: Jika diisi, jalankan zero-shot CLIP classification

        Returns:
            CVAnalysisResult berisi semua hasil
        """
        start = time.perf_counter()
        image = ImagePreprocessor.load(source)
        models_used = []

        with mlflow.start_run(run_name="cv_analyze"):
            mlflow.log_params({
                "source": str(source)[:100],
                "image_size": f"{image.width}x{image.height}",
                "run_caption": run_caption,
                "run_detection": run_detection,
                "run_ocr": run_ocr,
            })

            result = CVAnalysisResult(
                image_width=image.width,
                image_height=image.height,
                source=image.source,
            )

            # 1. Image Captioning
            if run_caption:
                t0 = time.perf_counter()
                result.caption = self.captioner.caption(image)
                models_used.append("BLIP-caption")
                logger.debug(f"Caption: {(time.perf_counter()-t0)*1000:.0f}ms")

            # 2. Object Detection
            if run_detection:
                t0 = time.perf_counter()
                result.detections = self.yolo.detect(image)
                models_used.append("YOLOv8")
                logger.debug(f"Detection: {(time.perf_counter()-t0)*1000:.0f}ms")

            # 3. Zero-shot Classification (opsional)
            if classification_labels:
                t0 = time.perf_counter()
                result.classification = self.clip.classify(image, classification_labels)
                models_used.append("CLIP")
                logger.debug(f"CLIP: {(time.perf_counter()-t0)*1000:.0f}ms")

            # 4. OCR (opsional, lebih berat)
            if run_ocr:
                t0 = time.perf_counter()
                result.ocr = self.ocr.extract_text(image)
                models_used.append("EasyOCR")
                logger.debug(f"OCR: {(time.perf_counter()-t0)*1000:.0f}ms")

            total_ms = (time.perf_counter() - start) * 1000
            result.models_used = models_used
            result.total_latency_ms = round(total_ms, 2)

            mlflow.log_metrics({
                "total_latency_ms": total_ms,
                "objects_detected": result.detections.count if result.detections else 0,
                "ocr_chars": len(result.ocr.full_text) if result.ocr else 0,
            })

        logger.info(
            f"CV analysis done in {total_ms:.0f}ms | "
            f"Models: {models_used} | "
            f"Objects: {result.detections.count if result.detections else 0}"
        )
        return result

    # === Individual task methods ===

    def caption_image(self, source, prompt: str = None) -> str:
        """Shorthand: return caption string langsung."""
        image = ImagePreprocessor.load(source)
        return self.captioner.caption(image, prompt=prompt).caption

    def detect_objects(self, source, conf: float = None) -> DetectionResult:
        """Shorthand: return DetectionResult."""
        image = ImagePreprocessor.load(source)
        return self.yolo.detect(image, conf_threshold=conf)

    def classify_image(self, source, labels: List[str]) -> CLIPResult:
        """Shorthand: zero-shot CLIP classification."""
        image = ImagePreprocessor.load(source)
        return self.clip.classify(image, labels)

    def extract_text(self, source) -> str:
        """Shorthand: return OCR text string."""
        image = ImagePreprocessor.load(source)
        return self.ocr.extract_text_simple(image)

    def visual_qa(self, source, question: str) -> str:
        """Visual Question Answering: tanya tentang isi gambar."""
        image = ImagePreprocessor.load(source)
        return self.captioner.visual_qa(image, question).caption

    def image_text_similarity(self, source, text: str) -> float:
        """Hitung seberapa relevan teks dengan gambar (0-1)."""
        image = ImagePreprocessor.load(source)
        return self.clip.compute_similarity(image, text)

    def annotate_image(self, source) -> bytes:
        """
        Return gambar dengan bounding box yang sudah digambar — untuk visualisasi.
        Returns JPEG bytes.
        """
        import io
        from PIL import Image

        image = ImagePreprocessor.load(source)
        annotated = self.yolo.detect_and_annotate(image)
        pil_annotated = Image.fromarray(annotated)
        buf = io.BytesIO()
        pil_annotated.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
