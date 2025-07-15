# syntax=docker/dockerfile:1.5

# ---- Base Stage: Common setup for final image ----
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


# ---- Builder Stage: Build dependencies and cache models ----
# Use a full python image to have build tools available for C extensions
FROM python:3.11 as builder

# Build argument to choose CPU-only or full ML packages
ARG USE_CPU_ONLY=false

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    XDG_CACHE_HOME=/app/.cache

WORKDIR /app

RUN pip install --no-cache-dir uv
RUN uv venv

ENV UV_HTTP_TIMEOUT=1800

# Copy requirements files for conditional installation
COPY requirements-base.txt requirements-ml.txt requirements-cpu.txt ./

# Install packages with uv. This is a single RUN command to optimize layer caching.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    bash -c ' \
    set -e && \
    . .venv/bin/activate && \
    export UV_CONCURRENT_DOWNLOADS=$(nproc) && \
    export UV_CONCURRENT_INSTALLS=$(nproc) && \
    \
    echo "🚀 Installing base packages..." && \
    uv pip install --no-cache-dir -r requirements-base.txt && \
    \
    if [ "$USE_CPU_ONLY" = "true" ] ; then \
        echo "⚡ Installing CPU-only ML packages..." && \
        uv pip install \
            --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch && \
        uv pip install \
            --no-cache-dir \
            -r requirements-cpu.txt; \
    else \
        echo "🎯 Installing full ML packages (with potential CUDA support)..." && \
        uv pip install \
            --no-cache-dir \
            -r requirements-ml.txt; \
    fi && \
    echo "✅ All packages installed successfully!" \
    '

# Pre-download and cache sentence-transformer models to avoid download on first run
RUN echo "📦 Caching sentence-transformer models..." && \
    bash -c '. .venv/bin/activate && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\"sentence-transformers/paraphrase-multilingual-mpnet-base-v2\"); SentenceTransformer(\"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\")"'

RUN echo "📦 Caching cross-encoder models..." && \
    bash -c '. .venv/bin/activate && python -c "from sentence_transformers import CrossEncoder; CrossEncoder(\"cross-encoder/ms-marco-MiniLM-L-6-v2\")"'

# Tree-sitter grammars will be downloaded on first use by the application.
# This avoids pre-caching issues and ensures compatibility.
RUN echo "📦 Tree-sitter grammars will be downloaded on first use..."

# ---- Final Stage: Create the lean final image ----
FROM base as final

# Copy the virtual environment with all dependencies from the builder stage
COPY --from=builder /app/.venv ./.venv
# Copy the cache with pre-downloaded models and grammars
COPY --from=builder /app/.cache /app/.cache

# Activate venv for the final image by adding it to PATH and set ENV vars
ENV PATH="/app/.venv/bin:$PATH" \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    XDG_CACHE_HOME=/app/.cache

# Copy the application source code
COPY app ./app
COPY templates ./templates

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 