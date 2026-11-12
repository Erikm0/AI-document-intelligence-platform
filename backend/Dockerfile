# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------------
# DocIntel backend - FastAPI + Sentence Transformers (CPU) + ChromaDB client
# ---------------------------------------------------------------------------------
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml .
# Resolve runtime + ML dependencies straight from pyproject.toml (single source of
# truth). CPU-only PyTorch is installed first so the dependency resolver sees it
# as satisfied - the image stays ~1.5 GB smaller than with the CUDA build.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['ml']))" > requirements.txt \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

FROM ${PYTHON_IMAGE} AS runtime
ARG PRELOAD_MODELS=true
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRELOAD_MODELS=${PRELOAD_MODELS} \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models/hf \
    UPLOAD_DIR=/data/uploads \
    CHROMA_PERSIST_DIR=/data/chroma \
    AUTO_CREATE_SCHEMA=false

RUN groupadd --system app && useradd --system --gid app --home /app app \
 && mkdir -p /app /data/uploads /models/hf && chown -R app:app /app /data /models
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

# Bake the embedding and reranking models into the image so containers start
# fast and work fully offline (set PRELOAD_MODELS=false for a slimmer image).
USER app
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"; \
    fi

COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app app ./app
COPY --chown=app:app --chmod=755 docker-entrypoint.sh ./

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
