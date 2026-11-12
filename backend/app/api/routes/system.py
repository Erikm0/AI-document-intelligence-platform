from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.api.deps import ContainerDep
from app.schemas.system import ComponentHealth, HealthResponse, SystemInfo

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe (database, vector store, LLM)",
    responses={503: {"model": HealthResponse, "description": "A critical dependency is down"}},
)
def readiness(container: ContainerDep) -> JSONResponse:
    components: dict[str, ComponentHealth] = {}
    try:
        container.db.ping()
        components["database"] = ComponentHealth(status="ok")
    except Exception as exc:
        components["database"] = ComponentHealth(status="down", detail=str(exc)[:300])
    try:
        container.vector_store.ping()
        components["vector_store"] = ComponentHealth(
            status="ok",
            detail=f"{container.vector_store.name}: {container.vector_store.count()} vectors",
        )
    except Exception as exc:
        components["vector_store"] = ComponentHealth(status="down", detail=str(exc)[:300])
    ok, detail = container.llm.health()
    # The LLM is not critical: upload, browse and search keep working without it.
    components["llm"] = ComponentHealth(status="ok" if ok else "degraded", detail=detail)
    components["lexical_index"] = ComponentHealth(
        status="ok", detail=f"{len(container.lexical_index)} chunks"
    )

    critical_down = any(components[k].status == "down" for k in ("database", "vector_store"))
    overall: Literal["ok", "degraded", "down"] = (
        "down" if critical_down else ("ok" if ok else "degraded")
    )
    body = HealthResponse(status=overall, version=__version__, components=components)
    return JSONResponse(status_code=503 if critical_down else 200, content=body.model_dump())


@router.get("/system/info", response_model=SystemInfo, summary="Active model & pipeline config")
def system_info(container: ContainerDep) -> SystemInfo:
    s = container.settings
    return SystemInfo(
        version=__version__,
        environment=s.environment,
        llm_provider=s.llm_provider.value,
        llm_model=container.llm.model,
        embedding_provider=s.embedding_provider.value,
        embedding_model=container.embedder.name,
        reranker=container.reranker.name,
        vector_store=container.vector_store.name,
        chunk_size_tokens=s.chunk_size_tokens,
        chunk_overlap_tokens=s.chunk_overlap_tokens,
        retrieval_top_k=s.retrieval_top_k,
        min_relevance_score=container.pipeline.config.min_relevance,
        min_semantic_score=container.pipeline.config.min_semantic,
        max_upload_size_mb=s.max_upload_size_mb,
        allowed_extensions=s.allowed_extensions,
    )
