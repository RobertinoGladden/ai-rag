import pytest
from unittest.mock import MagicMock, patch
from src.loaders.base_loader import Document


# === VectorStore tests (mocked ChromaDB) ===

@patch("src.retrieval.vector_store.Chroma")
@patch("src.retrieval.vector_store.DocumentEmbedder")
def test_vector_store_add_documents(MockEmbedder, MockChroma):
    from src.retrieval.vector_store import VectorStore

    mock_embedder = MockEmbedder.return_value
    mock_embedder.get_embeddings_model.return_value = MagicMock()
    mock_embedder.chunk_documents.return_value = [
        Document(content="chunk 1", metadata={"source": "test.pdf"}),
        Document(content="chunk 2", metadata={"source": "test.pdf"}),
    ]

    mock_chroma = MockChroma.return_value
    mock_chroma._collection.count.return_value = 2
    mock_chroma.add_documents.return_value = None

    vs = VectorStore(embedder=mock_embedder)
    count = vs.add_documents([Document(content="full doc", metadata={})])

    assert count == 2
    mock_chroma.add_documents.assert_called_once()


@patch("src.retrieval.vector_store.Chroma")
@patch("src.retrieval.vector_store.DocumentEmbedder")
def test_vector_store_similarity_search(MockEmbedder, MockChroma):
    from src.retrieval.vector_store import VectorStore
    from langchain.schema import Document as LCDoc

    mock_embedder = MockEmbedder.return_value
    mock_embedder.get_embeddings_model.return_value = MagicMock()

    mock_chroma = MockChroma.return_value
    mock_chroma._collection.count.return_value = 5
    mock_chroma.similarity_search.return_value = [
        LCDoc(page_content="hasil relevan", metadata={"source": "doc.pdf"}),
    ]

    vs = VectorStore(embedder=mock_embedder)
    results = vs.similarity_search("pertanyaan test")

    assert len(results) == 1
    assert results[0].page_content == "hasil relevan"


@patch("src.retrieval.vector_store.Chroma")
@patch("src.retrieval.vector_store.DocumentEmbedder")
def test_vector_store_count(MockEmbedder, MockChroma):
    from src.retrieval.vector_store import VectorStore

    mock_embedder = MockEmbedder.return_value
    mock_embedder.get_embeddings_model.return_value = MagicMock()
    mock_chroma = MockChroma.return_value
    mock_chroma._collection.count.return_value = 42

    vs = VectorStore(embedder=mock_embedder)
    assert vs.count() == 42


# === RAGRetriever tests (mocked) ===

@patch("src.retrieval.retriever.VectorStore")
@patch("src.retrieval.retriever.GroqClient")
def test_rag_get_stats(MockGroq, MockVS):
    from src.retrieval.retriever import RAGRetriever

    MockVS.return_value.count.return_value = 10
    retriever = RAGRetriever()
    stats = retriever.get_stats()

    assert "total_chunks" in stats
    assert "embedding_model" in stats
    assert "llm_model" in stats


@patch("src.retrieval.retriever.VectorStore")
@patch("src.retrieval.retriever.GroqClient")
@patch("src.retrieval.retriever.LoaderFactory")
def test_rag_ingest(MockLoader, MockGroq, MockVS):
    from src.retrieval.retriever import RAGRetriever

    MockLoader.load_many.return_value = [
        Document(content="doc 1", metadata={}),
        Document(content="doc 2", metadata={}),
    ]
    MockVS.return_value.add_documents.return_value = 5
    MockVS.return_value.count.return_value = 5

    retriever = RAGRetriever()
    stats = retriever.ingest(["fake.pdf"])

    assert stats["documents_loaded"] == 2
    assert stats["chunks_indexed"] == 5
