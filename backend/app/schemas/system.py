from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: dict[str, ComponentHealth] = {}


class SystemInfo(BaseModel):
    version: str
    environment: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    reranker: str
    vector_store: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_top_k: int
    min_relevance_score: float
    min_semantic_score: float | None = None
    max_upload_size_mb: int
    allowed_extensions: list[str]
