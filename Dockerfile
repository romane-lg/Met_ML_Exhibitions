# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/
RUN python -m uv sync --all-extras

COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data
COPY artifacts /app/artifacts
COPY README.md /app/README.md

EXPOSE 8000
CMD ["python", "-m", "uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
