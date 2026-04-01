FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU dulu (terpisah agar layer cache efisien)
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

# Patch chromadb DefaultEmbeddingFunction (bypass onnxruntime)
RUN python -c "import pathlib, re; ef = pathlib.Path('/usr/local/lib/python3.10/site-packages/chromadb/utils/embedding_functions.py'); txt = ef.read_text(); txt = re.sub(r'def DefaultEmbeddingFunction\(\)[^:]*:.*?return ONNXMiniLM_L6_V2\(\)', 'def DefaultEmbeddingFunction():\n    return None', txt, flags=re.DOTALL); ef.write_text(txt); print('chromadb patched OK')"

# Directories
RUN mkdir -p chroma_db mlruns logs

# CORS fix di main.py sudah ditangani via env
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
