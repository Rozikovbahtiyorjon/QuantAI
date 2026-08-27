# =========================================================
# QuantAI Professional - Dockerfile
# Multi-stage build for production deployment
# =========================================================

# =========================================================
# Stage 1: Builder - Install dependencies and build
# =========================================================
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY run.py .

# Run tests to verify build
RUN python -m pytest tests/ -x -q --tb=short 2>&1 | tail -20

# =========================================================
# Stage 2: Runtime - Minimal production image
# =========================================================
FROM python:3.12-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libffi8 \
    libstdc++6 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quantai && useradd -r -g quantai quantai

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/checkpoints /app/config && \
    chown -R quantai:quantai /app

# Switch to non-root user
USER quantai

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from src.monitoring.health import HealthChecker; import asyncio; hc = HealthChecker(); result = asyncio.run(hc.health.liveness()); exit(0 if result['status'] == 'alive' else 1)"

# Expose metrics port
EXPOSE 9090

# Default command
ENTRYPOINT ["python", "run.py"]
CMD ["--help"]

# =========================================================
# Labels
# =========================================================
LABEL maintainer="QuantAI Team" \
      version="5.1.0" \
      description="QuantAI Professional - AI-driven Cryptocurrency Trading Platform" \
      org.opencontainers.image.source="https://github.com/DepthSight-Pro/QuantAI" \
      org.opencontainers.image.title="QuantAI Professional" \
      org.opencontainers.image.version="5.1.0"