---
title: Multimodal AI Platform
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal AI Platform

Platform AI multimodal yang menggabungkan:
- **RAG Pipeline** — Upload dokumen (PDF/TXT/DOCX), tanya jawab berbasis konteks menggunakan LLM (Groq + LLaMA)
- **Computer Vision** — Analisis gambar: captioning, object detection (YOLO), classification (CLIP), OCR (EasyOCR)

## Setup Secret

Setelah Space dibuat, tambahkan secret di **Settings → Secrets**:
- `GROQ_API_KEY` — API key dari [console.groq.com](https://console.groq.com)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (LLaMA 3.1 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Object Detection | YOLOv8 |
| Classification | CLIP (ViT-B-32) |
| Captioning | BLIP |
| OCR | EasyOCR |
| API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
