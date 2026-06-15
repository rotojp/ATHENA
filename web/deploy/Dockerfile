FROM python:3.11-slim

WORKDIR /app

# Build deps (for some optional packages); keep slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY web/openai_server.py ./openai_server.py
RUN pip install --no-cache-dir ".[web]"

# Health check hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-9000}/health || exit 1

EXPOSE 9000

CMD ["python", "openai_server.py"]
