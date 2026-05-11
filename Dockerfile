# ─────────────────────────────────────────────
# Stage 1: build dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS base
WORKDIR /app
# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*
# Python deps (no playwright in runtime image — scraper runs offline)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
# ─────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────
FROM base AS runtime
WORKDIR /app
COPY main.py agent.py catalog.py ./
COPY data/ ./data/
# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser
# Expose port
EXPOSE 8000
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# Start uvicorn — single worker keeps memory low on free-tier hosts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "35"]