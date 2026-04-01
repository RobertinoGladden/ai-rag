from typing import List, Optional, Iterator
from loguru import logger
import mlflow
import time

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage

from ..config import get_settings
from ..retrieval.vector_store import VectorStore
from ..llm.groq_client import GroqClient
from ..llm.prompt_templates import RAG_PROMPT, SUMMARY_PROMPT
from ..loaders import LoaderFactory, Document


class RAGRetriever:
    """
    Core class yang menyatukan semua komponen RAG:
    Document Loading → Chunking → Embedding → Retrieval → Generation
    """

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.groq = GroqClient()
        self._setup_mlflow()
        logger.info("RAGRetriever initialized.")

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

    # === INDEXING ===

    def ingest(self, sources: List[str]) -> dict:
        """
        Load, chunk, embed, dan index dokumen dari berbagai sources.
        
        Args:
            sources: List of file paths atau URLs
            
        Returns:
            dict berisi stats indexing
        """
        logger.info(f"Ingesting {len(sources)} sources...")
        start = time.time()

        with mlflow.start_run(run_name="ingest"):
            mlflow.log_params({
                "sources_count": len(sources),
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "embedding_model": self.settings.embedding_model,
            })

            # Load semua dokumen
            documents = LoaderFactory.load_many(sources)

            # Index ke vector store
            chunks_indexed = self.vector_store.add_documents(documents)

            elapsed = time.time() - start
            stats = {
                "documents_loaded": len(documents),
                "chunks_indexed": chunks_indexed,
                "sources": sources,
                "elapsed_seconds": round(elapsed, 2),
                "total_docs_in_store": self.vector_store.count(),
            }

            mlflow.log_metrics({
                "documents_loaded": len(documents),
                "chunks_indexed": chunks_indexed,
                "elapsed_seconds": elapsed,
            })

        logger.info(f"Ingestion selesai: {stats}")
        return stats

    # === QUERYING ===

    def query(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        top_k: Optional[int] = None,
        return_sources: bool = True,
    ) -> dict:
        """
        Jawab pertanyaan menggunakan RAG.
        
        Args:
            question: Pertanyaan user
            chat_history: Riwayat chat [{"role": "user"/"assistant", "content": "..."}]
            top_k: Jumlah chunks yang diretrieve
            return_sources: Sertakan source chunks di response
            
        Returns:
            dict dengan 'answer', 'sources', dan 'metadata'
        """
        start = time.time()
        logger.info(f"Query: '{question[:80]}...'")

        with mlflow.start_run(run_name="query"):
            mlflow.log_param("question", question[:250])
            mlflow.log_param("model", self.settings.groq_model)

            # Retrieve relevant chunks
            k = top_k or self.settings.top_k_retrieval
            retrieved = self.vector_store.similarity_search_with_score(question, k=k)

            # Format context
            context_parts = []
            sources = []
            for i, (doc, score) in enumerate(retrieved):
                source_info = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
                page_info = f" (hal. {doc.metadata['page']})" if "page" in doc.metadata else ""
                context_parts.append(
                    f"[Sumber {i+1}: {source_info}{page_info} | Relevansi: {1-score:.2f}]\n{doc.page_content}"
                )
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": round(1 - score, 4),
                })

            context = "\n\n---\n\n".join(context_parts)

            # Build prompt dan generate
            formatted_prompt = RAG_PROMPT.format_messages(
                context=context,
                question=question,
                chat_history=[
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else HumanMessage(content=m["content"])
                    for m in (chat_history or [])
                ],
            )

            answer = self.groq.invoke(formatted_prompt)
            elapsed = time.time() - start

            mlflow.log_metrics({
                "chunks_retrieved": len(retrieved),
                "answer_length": len(answer),
                "latency_seconds": elapsed,
            })

        result = {
            "answer": answer,
            "question": question,
            "latency_seconds": round(elapsed, 2),
            "chunks_retrieved": len(retrieved),
        }
        if return_sources:
            result["sources"] = sources

        return result

    def stream_query(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Streaming version dari query — yield token per token."""
        retrieved = self.vector_store.similarity_search(question)
        context = "\n\n---\n\n".join(
            f"[Sumber: {doc.metadata.get('filename', 'Unknown')}]\n{doc.page_content}"
            for doc in retrieved
        )
        formatted = RAG_PROMPT.format_messages(
            context=context,
            question=question,
            chat_history=[],
        )
        groq_stream = GroqClient(streaming=True)
        yield from groq_stream.stream(formatted)

    def summarize(self, source: str) -> str:
        """Buat ringkasan dari satu dokumen."""
        documents = LoaderFactory.load(source)
        full_text = "\n\n".join(doc.content for doc in documents)

        # Truncate jika terlalu panjang
        if len(full_text) > 12000:
            full_text = full_text[:12000] + "\n...[dokumen dipotong untuk efisiensi]"

        messages = SUMMARY_PROMPT.format_messages(document=full_text)
        return self.groq.invoke(messages)

    def get_stats(self) -> dict:
        """Statistik vector store saat ini."""
        return {
            "total_chunks": self.vector_store.count(),
            "collection_name": self.settings.chroma_collection_name,
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.groq_model,
        }
