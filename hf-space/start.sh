#!/bin/bash
set -e

echo "=== Multimodal AI Platform — Starting ==="
echo "GROQ_API_KEY: ${GROQ_API_KEY:+SET (hidden)}${GROQ_API_KEY:-NOT SET — RAG queries will fail!}"

# Export GROQ_API_KEY so child processes (supervisord → uvicorn) can access it
export GROQ_API_KEY="${GROQ_API_KEY:-}"

# Ensure directories exist
mkdir -p /app/rag/chroma_db /app/rag/mlruns /app/rag/logs \
         /app/cv/model_cache /app/cv/mlruns /app/cv/logs /app/cv/uploads \
         /var/log/supervisor /run

echo "Starting supervisord (nginx + rag-api + cv-api)..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
