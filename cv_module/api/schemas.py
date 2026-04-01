from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


# === Shared ===

class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


class DetectionSchema(BaseModel):
    label: str
    confidence: float
    bbox: BBoxSchema
    class_id: int


class OCRBoxSchema(BaseModel):
    text: str
    confidence: float
    bbox: list


# === Requests ===

class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., description="URL gambar yang akan dianalisis")
    run_caption: bool = Field(True, description="Generate image caption")
    run_detection: bool = Field(True, description="Deteksi objek dengan YOLO")
    run_ocr: bool = Field(False, description="Ekstrak teks dari gambar")
    classification_labels: Optional[List[str]] = Field(
        None,
        description="Label untuk zero-shot CLIP classification, e.g. ['kucing','anjing']",
        example=["indoor", "outdoor", "nature", "city"],
    )


class ClassifyRequest(BaseModel):
    url: str
    labels: List[str] = Field(..., min_length=2, description="Minimal 2 label kandidat")


class SimilarityRequest(BaseModel):
    url: str
    text: str = Field(..., min_length=1)


class VisualQARequest(BaseModel):
    url: str
    question: str = Field(..., description="Pertanyaan tentang isi gambar")


# === Responses ===

class CaptionResponse(BaseModel):
    caption: str
    model: str


class DetectionResponse(BaseModel):
    detections: List[DetectionSchema]
    count: int
    labels_summary: dict
    image_width: int
    image_height: int
    inference_time_ms: float


class ClassificationResponse(BaseModel):
    top_label: str
    top_score: float
    labels: List[str]
    probabilities: List[float]


class OCRResponse(BaseModel):
    full_text: str
    boxes: List[OCRBoxSchema]
    word_count: int
    language: str
    engine: str


class FullAnalysisResponse(BaseModel):
    image_width: int
    image_height: int
    source: str
    caption: Optional[CaptionResponse] = None
    detections: Optional[DetectionResponse] = None
    classification: Optional[ClassificationResponse] = None
    ocr: Optional[OCRResponse] = None
    summary_text: str = Field(..., description="Ringkasan teks dari semua model — siap dipakai sebagai konteks LLM")
    models_used: List[str]
    total_latency_ms: float


class SimilarityResponse(BaseModel):
    similarity_score: float
    text: str
    interpretation: str


class VisualQAResponse(BaseModel):
    question: str
    answer: str
