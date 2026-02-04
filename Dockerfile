# 2api.ai Server Dockerfile
# Optimized for Google Cloud Run (Python 3.11, non-root, multi-stage, venv-based)

############################
# Build stage
############################
FROM python:3.11-slim AS builder

WORKDIR /app

# Faster + more deterministic python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System build deps (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into venv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PATH="/opt/venv/bin:$PATH"

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/

# Create non-root user for security
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check (Cloud Run may ignore Docker HEALTHCHECK, but it's still useful locally)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import os, httpx; httpx.get(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8080\")}/health', timeout=5.0)" || exit 1

EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
