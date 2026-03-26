FROM python:3.12-slim AS base

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/
COPY nodes/ nodes/
COPY configs/ configs/

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
