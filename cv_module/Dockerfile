FROM python:3.10-slim

WORKDIR /app

# System deps (tesseract untuk OCR)
RUN apt-get update && apt-get install -y \
    build-essential curl git \
    tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU dulu
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install requirements lainnya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Cache model dir
RUN mkdir -p model_cache mlruns logs
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
