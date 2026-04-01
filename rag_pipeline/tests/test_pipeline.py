import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.routes.RAGRetriever") as MockRetriever:
        mock = MockRetriever.return_value
        mock.get_stats.return_value = {
            "total_chunks": 42,
            "collection_name": "test_collection",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "llama-3.1-70b-versatile",
        }
        mock.ingest.return_value = {
            "documents_loaded": 2,
            "chunks_indexed": 8,
            "sources": ["test.pdf"],
            "elapsed_seconds": 1.5,
            "total_docs_in_store": 8,
        }
        mock.query.return_value = {
            "answer": "Ini adalah jawaban dari RAG.",
            "question": "Apa itu RAG?",
            "latency_seconds": 0.8,
            "chunks_retrieved": 3,
            "sources": [],
        }
        mock.summarize.return_value = "Ringkasan dokumen: ..."

        from src.api.main import app
        yield TestClient(app)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats(client):
    r = client.get("/api/v1/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_chunks"] == 42
    assert "embedding_model" in data


def test_ingest(client):
    r = client.post("/api/v1/ingest", json={"sources": ["test.pdf"]})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] == 8


def test_query(client):
    r = client.post("/api/v1/query", json={"question": "Apa itu RAG?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["chunks_retrieved"] == 3


def test_query_with_chat_history(client):
    r = client.post("/api/v1/query", json={
        "question": "Bisa dijelaskan lebih lanjut?",
        "chat_history": [
            {"role": "user", "content": "Apa itu RAG?"},
            {"role": "assistant", "content": "RAG adalah Retrieval Augmented Generation."},
        ]
    })
    assert r.status_code == 200


def test_query_invalid_empty_question(client):
    r = client.post("/api/v1/query", json={"question": ""})
    assert r.status_code == 422  # Pydantic validation error


def test_summarize(client):
    r = client.post("/api/v1/summarize", json={"source": "test.pdf"})
    assert r.status_code == 200
    assert "summary" in r.json()
