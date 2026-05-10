# ── DocQA-Pro Backend Dockerfile ──────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="DocQA-Pro"
LABEL description="Hybrid RAG backend — FAISS + BM25 + Flan-T5 + FastAPI"

# System dependencies for PDF/DOCX parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY .env.example .env

# Pre-create data directories
RUN mkdir -p data/uploads data/vectorstores

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
