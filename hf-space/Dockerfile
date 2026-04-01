# ============================================================
# Multimodal AI — Unified Dockerfile for Hugging Face Spaces
# Combines: RAG Pipeline + CV Module + Nginx (frontend)
# ============================================================
FROM python:3.10-slim

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    nginx supervisor \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch CPU (cached layer) ──────────────────────
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── RAG dependencies ────────────────────────────────────────
COPY rag_pipeline/requirements.txt /tmp/rag-requirements.txt
# Remove torch/torchvision lines (already installed above)
RUN sed -i '/^torch==/d; /^torchvision==/d' /tmp/rag-requirements.txt && \
    pip install --no-cache-dir -r /tmp/rag-requirements.txt

# ── CV dependencies ─────────────────────────────────────────
COPY cv_module/requirements.txt /tmp/cv-requirements.txt
RUN sed -i '/^torch==/d; /^torchvision==/d' /tmp/cv-requirements.txt && \
    pip install --no-cache-dir -r /tmp/cv-requirements.txt

# ── Patch chromadb DefaultEmbeddingFunction ──────────────────
RUN python -c "\
import pathlib, re; \
ef = pathlib.Path('/usr/local/lib/python3.10/site-packages/chromadb/utils/embedding_functions.py'); \
txt = ef.read_text(); \
txt = re.sub(r'def DefaultEmbeddingFunction\(\)[^:]*:.*?return ONNXMiniLM_L6_V2\(\)', \
'def DefaultEmbeddingFunction():\n    return None', txt, flags=re.DOTALL); \
ef.write_text(txt); print('chromadb patched OK')" || echo "chromadb patch skipped"

# ── Copy source code ────────────────────────────────────────
COPY rag_pipeline/src/ /app/rag/src/
COPY cv_module/src/    /app/cv/src/

# ── Copy frontend ───────────────────────────────────────────
COPY frontend/index.html /app/frontend/index.html

# ── Copy configs ────────────────────────────────────────────
COPY hf-space/nginx.conf      /etc/nginx/nginx.conf
COPY hf-space/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY hf-space/start.sh        /app/start.sh
RUN chmod +x /app/start.sh

# ── Create directories ──────────────────────────────────────
RUN mkdir -p /app/rag/chroma_db /app/rag/mlruns /app/rag/logs \
             /app/cv/model_cache /app/cv/mlruns /app/cv/logs /app/cv/uploads \
             /var/log/supervisor /run

# ── Expose HF Spaces default port ───────────────────────────
EXPOSE 7860

CMD ["/app/start.sh"]
