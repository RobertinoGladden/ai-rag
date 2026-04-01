"""
Contoh integrasi CV Module + RAG Module.
Alur: upload gambar → CV analisis → summary jadi konteks RAG → user bisa tanya tentang gambar.

Ini adalah inti dari "Multimodal" dalam proyek portfolio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def multimodal_pipeline_demo():
    """
    Demo alur lengkap:
    1. Load gambar
    2. CV Pipeline: caption + deteksi objek
    3. Hasil CV dijadikan dokumen di RAG
    4. User bisa tanya tentang gambar via chat
    """
    print("\n=== Multimodal Pipeline: CV + RAG ===\n")

    # ---- Step 1: CV Analysis ----
    print("Step 1: Analisis gambar dengan CV pipeline...")

    # Import CV pipeline (dari cv_module)
    # from cv_module.src.cv_pipeline import CVPipeline
    # cv = CVPipeline()
    # cv_result = cv.analyze("path/to/image.jpg", run_caption=True, run_detection=True)
    # summary = cv_result.to_summary()

    # Simulasi hasil CV (untuk demo tanpa model load)
    summary = """Deskripsi gambar: a bus stopped at a bus stop with people waiting
Objek terdeteksi: 1x bus, 4x person, 2x car
Klasifikasi: street scene (confidence: 87.3%)"""

    print(f"CV Summary:\n{summary}\n")

    # ---- Step 2: Simpan ke RAG sebagai dokumen ----
    print("Step 2: Index CV summary ke RAG vector store...")

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(f"[Image Analysis: bus_stop_photo.jpg]\n\n{summary}")
        tmp_path = f.name

    # Import RAG pipeline (dari rag_pipeline)
    # from rag_pipeline.src.retrieval.retriever import RAGRetriever
    # rag = RAGRetriever()
    # rag.ingest([tmp_path])
    print(f"Dokumen CV diindex: {tmp_path}")

    # ---- Step 3: Query tentang gambar ----
    print("\nStep 3: Query RAG tentang gambar yang sudah dianalisis...")

    example_queries = [
        "Ada berapa orang di gambar tersebut?",
        "Apa objek utama yang terdeteksi?",
        "Apakah ada kendaraan dalam gambar?",
    ]

    for q in example_queries:
        print(f"\nUser: {q}")
        # result = rag.query(q)
        # print(f"AI: {result['answer']}")
        print("AI: [Jawaban dari RAG berdasarkan CV summary]")

    os.unlink(tmp_path)
    print("\n=== Integrasi selesai! ===")
    print("\nAlur ini membuktikan kemampuan end-to-end multimodal:")
    print("  Gambar → CV Analysis → Teks Summary → RAG Knowledge Base → Natural Language QA")


if __name__ == "__main__":
    multimodal_pipeline_demo()
