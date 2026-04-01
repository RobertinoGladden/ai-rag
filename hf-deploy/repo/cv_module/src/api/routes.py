from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from loguru import logger

from .schemas import (
    AnalyzeURLRequest, FullAnalysisResponse,
    ClassifyRequest, ClassificationResponse,
    SimilarityRequest, SimilarityResponse,
    VisualQARequest, VisualQAResponse,
    CaptionResponse, DetectionResponse, OCRResponse,
)
from ..cv_pipeline import CVPipeline

router = APIRouter()

_pipeline: CVPipeline = None


def get_pipeline() -> CVPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CVPipeline()
    return _pipeline


# === HEALTH ===

@router.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "service": "CV Pipeline API"}


# === FULL ANALYSIS ===

@router.post("/analyze/url", response_model=FullAnalysisResponse, tags=["analysis"])
async def analyze_from_url(req: AnalyzeURLRequest):
    """
    Analisis gambar dari URL.
    Jalankan caption, object detection, optional OCR + CLIP classification sekaligus.
    """
    try:
        result = get_pipeline().analyze(
            source=req.url,
            run_caption=req.run_caption,
            run_detection=req.run_detection,
            run_ocr=req.run_ocr,
            classification_labels=req.classification_labels,
        )
        return _to_response(result)
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/upload", response_model=FullAnalysisResponse, tags=["analysis"])
async def analyze_upload(
    file: UploadFile = File(...),
    run_caption: bool = True,
    run_detection: bool = True,
    run_ocr: bool = False,
):
    """Upload dan analisis gambar langsung (multipart)."""
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(400, detail=f"Tipe file tidak didukung: {file.content_type}")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, detail="Ukuran file maksimum 10MB")

    try:
        result = get_pipeline().analyze(
            source=data,
            run_caption=run_caption,
            run_detection=run_detection,
            run_ocr=run_ocr,
        )
        return _to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === INDIVIDUAL TASKS ===

@router.post("/caption", response_model=CaptionResponse, tags=["tasks"])
async def caption(url: str, prompt: str = None):
    """Generate deskripsi teks dari gambar."""
    try:
        from ..processors.image_preprocessor import ImagePreprocessor
        image = ImagePreprocessor.load(url)
        result = get_pipeline().captioner.caption(image, prompt=prompt)
        return CaptionResponse(caption=result.caption, model=result.model)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/detect", response_model=DetectionResponse, tags=["tasks"])
async def detect(url: str, conf: float = None):
    """Deteksi objek dalam gambar dengan YOLOv8."""
    try:
        result = get_pipeline().detect_objects(url, conf=conf)
        return DetectionResponse(
            detections=[d.to_dict() for d in result.detections],
            count=result.count,
            labels_summary=result.labels_summary,
            image_width=result.image_width,
            image_height=result.image_height,
            inference_time_ms=result.inference_time_ms,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/classify", response_model=ClassificationResponse, tags=["tasks"])
async def classify(req: ClassifyRequest):
    """
    Zero-shot image classification dengan CLIP.
    Tidak perlu training — cukup berikan daftar label kandidat.
    """
    try:
        result = get_pipeline().classify_image(req.url, req.labels)
        return ClassificationResponse(
            top_label=result.top_label,
            top_score=result.top_score,
            labels=result.labels,
            probabilities=result.probabilities,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


class OCRRequest(BaseModel):
    url: str


@router.post("/ocr", response_model=OCRResponse, tags=["tasks"])
async def ocr(req: OCRRequest):
    """Ekstrak teks dari gambar menggunakan EasyOCR."""
    try:
        from ..processors.image_preprocessor import ImagePreprocessor
        image = ImagePreprocessor.load(req.url)
        result = get_pipeline().ocr.extract_text(image)
        return OCRResponse(
            full_text=result.full_text,
            boxes=[b.to_dict() for b in result.boxes],
            word_count=result.word_count,
            language=result.language,
            engine=result.engine,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/similarity", response_model=SimilarityResponse, tags=["tasks"])
async def image_text_similarity(req: SimilarityRequest):
    """Hitung relevansi antara gambar dan teks (0.0 - 1.0)."""
    try:
        score = get_pipeline().image_text_similarity(req.url, req.text)
        interpretation = (
            "Sangat relevan" if score > 0.7
            else "Cukup relevan" if score > 0.5
            else "Kurang relevan"
        )
        return SimilarityResponse(
            similarity_score=score,
            text=req.text,
            interpretation=interpretation,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/visual-qa", response_model=VisualQAResponse, tags=["tasks"])
async def visual_qa(req: VisualQARequest):
    """Visual Question Answering — tanya tentang isi gambar."""
    try:
        answer = get_pipeline().visual_qa(req.url, req.question)
        return VisualQAResponse(question=req.question, answer=answer)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/annotate", tags=["tasks"])
async def annotate(url: str):
    """Return gambar dengan bounding box YOLO yang sudah digambar (JPEG)."""
    try:
        jpeg_bytes = get_pipeline().annotate_image(url)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# === HELPER ===

def _to_response(result) -> FullAnalysisResponse:
    """Convert CVAnalysisResult ke FullAnalysisResponse."""
    caption_r = None
    if result.caption:
        caption_r = CaptionResponse(
            caption=result.caption.caption,
            model=result.caption.model,
        )

    det_r = None
    if result.detections:
        det_r = DetectionResponse(
            detections=[d.to_dict() for d in result.detections.detections],
            count=result.detections.count,
            labels_summary=result.detections.labels_summary,
            image_width=result.detections.image_width,
            image_height=result.detections.image_height,
            inference_time_ms=result.detections.inference_time_ms,
        )

    cls_r = None
    if result.classification:
        cls_r = ClassificationResponse(
            top_label=result.classification.top_label,
            top_score=result.classification.top_score,
            labels=result.classification.labels,
            probabilities=result.classification.probabilities,
        )

    ocr_r = None
    if result.ocr:
        ocr_r = OCRResponse(
            full_text=result.ocr.full_text,
            boxes=[b.to_dict() for b in result.ocr.boxes],
            word_count=result.ocr.word_count,
            language=result.ocr.language,
            engine=result.ocr.engine,
        )

    return FullAnalysisResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        source=result.source,
        caption=caption_r,
        detections=det_r,
        classification=cls_r,
        ocr=ocr_r,
        summary_text=result.to_summary(),
        models_used=result.models_used,
        total_latency_ms=result.total_latency_ms,
    )