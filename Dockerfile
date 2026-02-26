# ── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py config.py config.yaml run_rag.py run_pipeline.py ./
COPY rag/ ./rag/
COPY etl/ ./etl/

# data/ is volume-mounted at runtime so the FAISS index persists
# between container restarts. Create the directory so the mount point exists.
RUN mkdir -p data logs

# Expose FastAPI port
EXPOSE 8000

# Environment variable defaults (override via docker run -e or .env bind-mount)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
