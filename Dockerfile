# Base image for the clinical-agent runtime.
#
# Phase 1: base + dev deps only. ML deps (torch, transformers, qdrant-client,
# etc.) are added in Phase 2 once the retrieval tool needs them. Keeping this
# layer slim makes CI builds fast.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps kept minimal; expand in Phase 2 with build-essential when adding
# bitsandbytes / faiss / etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for cache efficiency.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Copy the rest (tests, scripts, configs).
COPY . .

CMD ["pytest", "-v"]
