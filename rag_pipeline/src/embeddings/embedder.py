from typing import List
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config import get_settings
from ..loaders.base_loader import Document


class DocumentEmbedder:
    """
    Bertanggung jawab untuk:
    1. Chunking dokumen panjang jadi potongan yang bisa di-embed
    2. Membuat embedding vektor pakai model lokal (no API cost!)
    """

    def __init__(self):
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info("Embedder ready.")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split dokumen panjang jadi chunks.
        Metadata dari dokumen asli diwarisi ke setiap chunk.
        """
        chunks = []
        for doc in documents:
            texts = self.splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    "parent_doc_id": doc.doc_id,
                }
                chunks.append(Document(
                    content=text,
                    metadata=chunk_metadata,
                ))

        logger.info(f"Chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def get_embeddings_model(self):
        """Return LangChain-compatible embeddings object untuk ChromaDB."""
        return self.embeddings
