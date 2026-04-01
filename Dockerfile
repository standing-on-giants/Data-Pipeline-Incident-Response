# ────────────────────────────────────────────────────────────────────
# Data Pipeline Incident Response — OpenEnv Container
# Build:  docker build -t data-pipeline-env .
# Run:    docker run -p 8001:8001 data-pipeline-env
# Test:   curl http://localhost:8001/health
# ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/        ./src/
COPY openenv.yaml .
COPY inference.py .

# Expose WebSocket / HTTP port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Environment variables (override at runtime)
ENV PORT=8001
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start the OpenEnv server
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]