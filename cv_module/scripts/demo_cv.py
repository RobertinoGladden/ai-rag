"""
Demo CV Pipeline — jalankan analisis gambar end-to-end.
Menunjukkan semua kapabilitas: captioning, detection, CLIP, OCR, visual QA.

Usage:
    python scripts/demo_cv.py
    python scripts/demo_cv.py --image path/to/gambar.jpg
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


DEMO_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def run_demo(image_source: str = None):
    logger.info("=== CV Pipeline Demo ===\n")
    source = image_source or DEMO_IMAGE_URL

    from src.cv_pipeline import CVPipeline
    pipeline = CVPipeline()

    # 1. Full analysis
    logger.info(f"1. Full analysis: {source}\n")
    result = pipeline.analyze(
        source=source,
        run_caption=True,
        run_detection=True,
        run_ocr=False,
        classification_labels=["bus", "car", "street", "indoor scene", "nature"],
    )

    print("\n" + "="*50)
    print(f"  Image size   : {result.image_width}x{result.image_height}")
    print(f"  Models used  : {', '.join(result.models_used)}")
    print(f"  Total latency: {result.total_latency_ms:.0f}ms")
    print("="*50)

    if result.caption:
        print(f"\nCaption      : {result.caption.caption}")

    if result.detections:
        print(f"\nDetections   : {result.detections.count} objects")
        for label, count in result.detections.labels_summary.items():
            print(f"  - {label}: {count}x")

    if result.classification:
        print(f"\nCLIP Top     : {result.classification.top_label} ({result.classification.top_score:.1%})")
        for lbl, prob in zip(result.classification.labels, result.classification.probabilities):
            bar = "█" * int(prob * 20)
            print(f"  {lbl:20s} {bar} {prob:.1%}")

    print(f"\nSummary text :\n{result.to_summary()}")

    # 2. Visual QA demo
    logger.info("\n2. Visual QA demo...")
    questions = [
        "How many people are in the image?",
        "What is the main subject of this image?",
    ]
    for q in questions:
        answer = pipeline.visual_qa(source, q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    # 3. Image-text similarity
    logger.info("\n3. Image-text similarity...")
    texts = ["a bus on the street", "a cat sleeping", "people walking"]
    for text in texts:
        score = pipeline.image_text_similarity(source, text)
        print(f"  '{text}': {score:.3f}")

    logger.info("\n=== Demo selesai! ===")
    logger.info("Output summary_text bisa langsung dikirim ke RAG pipeline sebagai konteks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path atau URL gambar (default: demo URL)")
    args = parser.parse_args()
    run_demo(args.image)
