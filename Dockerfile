# ==============================================================================
# CSRAG — Production Dockerfile
# Multi-stage build: builder installs deps, production runs the app.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into a venv
# ------------------------------------------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /app

# Build tools needed by some wheels (psycopg, qdrant-client, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: production — lean runtime image
# ------------------------------------------------------------------------------
FROM python:3.13-slim AS production

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code only
COPY app/ ./app/

# Ownership
RUN chown -R appuser:appgroup /app

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Liveness probe — uses httpx (already in requirements)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
