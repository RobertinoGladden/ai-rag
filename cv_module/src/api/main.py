from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from .routes import router
from ..config import get_cv_settings

settings = get_cv_settings()

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}")
logger.add("./logs/cv_api.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="CV Pipeline API",
    description="""
## Multimodal AI Assistant — Computer Vision Module

Endpoint untuk analisis gambar menggunakan:
- **BLIP** — image captioning & visual QA
- **YOLOv8** — real-time object detection (80 kelas COCO)
- **CLIP** — zero-shot classification & image-text similarity
- **EasyOCR** — text extraction dari gambar (80+ bahasa)
- **MLflow** — latency & performance tracking

### Integrasi dengan RAG Module
Output `summary_text` dari `/analyze` bisa langsung dipakai sebagai
konteks untuk RAG pipeline — gambar bisa menjadi bagian dari knowledge base.
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("CV Pipeline API starting up...")
    logger.info(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")
