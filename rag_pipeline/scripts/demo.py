"""
Demo script — jalankan RAG pipeline end-to-end tanpa API server.
Cocok untuk testing lokal dan demo ke interviewer.

Usage:
    python scripts/demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def run_demo():
    logger.info("=== RAG Pipeline Demo ===\n")

    # 1. Init pipeline
    logger.info("1. Initializing RAG pipeline...")
    from src.retrieval.retriever import RAGRetriever
    rag = RAGRetriever()

    # 2. Buat sample dokumen untuk demo
    import tempfile, os
    sample_text = """
    Retrieval-Augmented Generation (RAG) adalah teknik yang menggabungkan
    pencarian informasi (retrieval) dengan kemampuan generasi teks dari LLM.

    Cara kerja RAG:
    1. Dokumen diubah menjadi vektor embedding dan disimpan di vector store
    2. Saat ada pertanyaan, sistem mencari chunk paling relevan
    3. Chunk tersebut diberikan sebagai konteks ke LLM
    4. LLM menghasilkan jawaban berdasarkan konteks tersebut

    Keunggulan RAG dibanding fine-tuning:
    - Tidak perlu training ulang model
    - Bisa update knowledge base kapan saja
    - Lebih hemat biaya komputasi
    - Jawaban bisa dilacak ke sumber aslinya (traceable)

    Stack yang umum digunakan:
    - LangChain / LlamaIndex untuk orchestration
    - ChromaDB / Pinecone / Weaviate untuk vector store
    - OpenAI / Groq / Ollama untuk LLM
    - Sentence Transformers untuk embedding lokal
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(sample_text)
        tmp_path = f.name

    try:
        # 3. Ingest dokumen
        logger.info("\n2. Ingesting sample document...")
        stats = rag.ingest([tmp_path])
        logger.info(f"   Chunks indexed: {stats['chunks_indexed']}")

        # 4. Query
        questions = [
            "Bagaimana cara kerja RAG?",
            "Apa keunggulan RAG dibanding fine-tuning?",
            "Tools apa saja yang dipakai dalam RAG stack?",
        ]

        logger.info("\n3. Running queries...\n")
        for q in questions:
            logger.info(f"Q: {q}")
            result = rag.query(q, return_sources=False)
            logger.info(f"A: {result['answer']}")
            logger.info(f"   (latency: {result['latency_seconds']}s, "
                        f"chunks: {result['chunks_retrieved']})\n")

        # 5. Stats
        logger.info(f"\n4. Vector store stats: {rag.get_stats()}")
        logger.info("\n=== Demo selesai! ===")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    run_demo()
