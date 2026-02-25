# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV PYTHONPATH=/app
# Use CPU-only torch index to avoid pulling CUDA drivers (~5 GB) into the image
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Copy dependency files first so this layer is cached unless deps change
COPY pyproject.toml uv.lock /app/
RUN uv sync --all-extras

# Copy app code after deps so code changes don't invalidate the dep cache layer
COPY src /app/src
COPY scripts /app/scripts
COPY README.md /app/README.md

# data/ and artifacts/ are mounted as volumes at runtime — not baked into image

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
