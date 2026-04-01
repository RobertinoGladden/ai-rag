from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from .routes import router
from ..config import get_settings

# === Logging setup ===
settings = get_settings()
logger.remove()
logger.add(sys.stderr, level=settings.log_level, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}")
logger.add(settings.log_file, rotation="10 MB", retention="7 days", level="DEBUG")

# === App ===
app = FastAPI(
    title="RAG Pipeline API",
    description="""
## Multimodal AI Assistant — RAG Module

Endpoint untuk indexing dan querying dokumen menggunakan:
- **Groq** (LLM inference — cepat & gratis)
- **Sentence Transformers** (local embeddings)
- **ChromaDB** (vector store persistent)
- **MLflow** (experiment tracking)

### Supported document types
PDF, TXT, Markdown, DOCX, JSON, JSONL, URL
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain spesifik di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routes ===
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("RAG Pipeline API starting up...")
    logger.info(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")


@app.on_event("shutdown")
async def shutdown():
    logger.info("RAG Pipeline API shutting down.")
