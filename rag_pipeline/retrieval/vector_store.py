from typing import List, Optional
from loguru import logger
from langchain_chroma import Chroma
from langchain.schema import Document as LCDocument

from ..config import get_settings
from ..loaders.base_loader import Document
from ..embeddings.embedder import DocumentEmbedder


class VectorStore:
    """
    Wrapper di atas ChromaDB.
    Menangani indexing, persistence, dan similarity search.
    """

    def __init__(self, embedder: Optional[DocumentEmbedder] = None):
        settings = get_settings()
        self.embedder = embedder or DocumentEmbedder()
        self.settings = settings

        self.db = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embedder.get_embeddings_model(),
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(
            f"VectorStore ready. Collection: '{settings.chroma_collection_name}' "
            f"| Docs: {self.db._collection.count()}"
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Chunk dan index dokumen ke ChromaDB.
        Returns: jumlah chunks yang berhasil di-index.
        """
        chunks = self.embedder.chunk_documents(documents)

        lc_docs = [
            LCDocument(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]

        self.db.add_documents(lc_docs)
        logger.info(f"Indexed {len(chunks)} chunks ke ChromaDB.")
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> List[LCDocument]:
        """Cari dokumen paling relevan berdasarkan query."""
        k = k or self.settings.top_k_retrieval
        results = self.db.similarity_search(query, k=k, filter=filter)
        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[tuple]:
        """Sama seperti similarity_search tapi return (doc, score)."""
        k = k or self.settings.top_k_retrieval
        return self.db.similarity_search_with_score(query, k=k)

    def reset_collection(self):
        """
        Hapus semua dokumen TANPA mematikan collection.
        Pakai reset_collection() dari langchain-chroma — collection tetap hidup
        dan langsung siap untuk ingest berikutnya.
        """
        self.db.reset_collection()
        logger.warning(
            f"Collection '{self.settings.chroma_collection_name}' di-reset. "
            "Semua dokumen dihapus, collection siap dipakai kembali."
        )

    def delete_collection(self):
        """Alias ke reset_collection() — collection tidak dimatikan, aman."""
        self.reset_collection()

    def count(self) -> int:
        """Jumlah chunks yang tersimpan."""
        return self.db._collection.count()

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Return LangChain retriever untuk dipakai di chain."""
        search_kwargs = search_kwargs or {"k": self.settings.top_k_retrieval}
        return self.db.as_retriever(search_kwargs=search_kwargs)