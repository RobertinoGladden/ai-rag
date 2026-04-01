from pydantic import BaseModel, Field
from typing import List, Optional


class IngestRequest(BaseModel):
    sources: List[str] = Field(
        ...,
        description="List of file paths atau URLs untuk di-index",
        example=["./docs/laporan.pdf", "https://example.com/artikel"],
    )


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    total_docs_in_store: int
    elapsed_seconds: float
    sources: List[str]


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Riwayat chat untuk multi-turn")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    return_sources: bool = Field(default=True)
    stream: bool = Field(default=False, description="Gunakan streaming response")


class SourceChunk(BaseModel):
    content: str
    metadata: dict
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    question: str
    latency_seconds: float
    chunks_retrieved: int
    sources: Optional[List[SourceChunk]] = None


class SummarizeRequest(BaseModel):
    source: str = Field(..., description="File path atau URL untuk diringkas")


class SummarizeResponse(BaseModel):
    summary: str
    source: str


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedding_model: str
    llm_model: str


class DeleteResponse(BaseModel):
    status: str
    message: str
