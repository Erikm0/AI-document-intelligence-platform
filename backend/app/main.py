"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app import __version__
from app.api.routes import chat, conversations, documents, system
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, request_id_var

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
**DocIntel** answers natural-language questions about your PDF, DOCX and TXT files
with grounded, cited answers - running entirely on local, open-source models.

* **Ingestion** - parsing, cleaning, structure-aware chunking with page provenance.
* **Retrieval** - hybrid dense (Sentence Transformers + ChromaDB) and sparse (BM25)
  search fused with Reciprocal Rank Fusion, then cross-encoder reranking.
* **Generation** - Ollama-hosted LLM, constrained to cite numbered sources.
* **Guardrails** - evidence gate, refusal detection, citation validation & repair,
  lexical grounding score.

All errors share the envelope `{"error": {"code", "message", "details"}, "request_id"}`.
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or build_container(settings)
        c.startup()
        app.state.container = c
        logger.info(
            "DocIntel %s started (llm=%s, embedder=%s, reranker=%s, vectors=%s)",
            __version__,
            c.llm.model,
            c.embedder.name,
            c.reranker.name,
            c.vector_store.name,
        )
        try:
            yield
        finally:
            c.shutdown()

    app = FastAPI(
        title="DocIntel API",
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        elapsed = (time.perf_counter() - started) * 1000
        if not request.url.path.endswith("/health"):
            logger.info(
                "%s %s -> %s (%.1f ms) [%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
                request_id,
            )
        return response

    register_exception_handlers(app)
    for router in (system.router, documents.router, chat.router, conversations.router):
        app.include_router(router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()  # `uvicorn app.main:app`
