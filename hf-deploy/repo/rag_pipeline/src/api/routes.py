from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from loguru import logger
import tempfile
import shutil
from pathlib import Path

from .schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    SummarizeRequest, SummarizeResponse,
    StatsResponse, DeleteResponse,
)
from ..retrieval.retriever import RAGRetriever

router = APIRouter()

# Singleton retriever — di-init sekali saat startup
_retriever: RAGRetriever = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


# === HEALTH ===

@router.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "service": "RAG Pipeline API"}


# === STATS ===

@router.get("/stats", response_model=StatsResponse, tags=["system"])
async def get_stats():
    """Info tentang vector store saat ini."""
    return get_retriever().get_stats()


# === INGEST ===

@router.post("/ingest", response_model=IngestResponse, tags=["indexing"])
async def ingest_documents(request: IngestRequest):
    """
    Index dokumen dari file path atau URL ke vector store.
    Mendukung: PDF, TXT, MD, DOCX, JSON, JSONL, URL
    """
    try:
        stats = get_retriever().ingest(request.sources)
        return IngestResponse(status="success", **stats)
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/upload", tags=["indexing"])
async def ingest_upload(file: UploadFile = File(...)):
    """Upload dan index file langsung via multipart."""
    allowed_exts = {".pdf", ".txt", ".md", ".docx", ".json", ".jsonl"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Ekstensi '{ext}' tidak didukung. Gunakan: {allowed_exts}"
        )

    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        stats = get_retriever().ingest([tmp_path])
        return IngestResponse(status="success", **stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# === QUERY ===

@router.post("/query", response_model=QueryResponse, tags=["querying"])
async def query(request: QueryRequest):
    """
    Tanya jawab berdasarkan dokumen yang sudah di-index.
    Mendukung multi-turn conversation via chat_history.
    """
    if request.stream:
        # Streaming response
        def generate():
            yield from get_retriever().stream_query(
                question=request.question,
                chat_history=[m.model_dump() for m in (request.chat_history or [])],
            )
        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        result = get_retriever().query(
            question=request.question,
            chat_history=[m.model_dump() for m in (request.chat_history or [])],
            top_k=request.top_k,
            return_sources=request.return_sources,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === SUMMARIZE ===

@router.post("/summarize", response_model=SummarizeResponse, tags=["querying"])
async def summarize(request: SummarizeRequest):
    """Buat ringkasan otomatis dari dokumen."""
    try:
        summary = get_retriever().summarize(request.source)
        return SummarizeResponse(summary=summary, source=request.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === DELETE ===

@router.delete("/collection", response_model=DeleteResponse, tags=["system"])
async def delete_collection():
    """Hapus semua dokumen dari vector store. HATI-HATI: tidak bisa di-undo."""
    get_retriever().vector_store.delete_collection()
    return DeleteResponse(
        status="success",
        message="Semua dokumen berhasil dihapus dari vector store."
    )
