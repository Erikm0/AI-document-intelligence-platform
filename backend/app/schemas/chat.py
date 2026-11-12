from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from app.db.models import MessageRole
from app.schemas.common import APIModel


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"  # refused


class RetrievalMode(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class Citation(APIModel):
    index: int = Field(description="The [n] marker used in the answer text")
    chunk_id: str
    document_id: uuid.UUID
    filename: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    text: str = Field(description="Full text of the cited chunk")
    relevance: float = Field(ge=0.0, le=1.0, description="Normalised reranker score")
    cited: bool = Field(
        default=True, description="False when the source was retrieved but not cited"
    )


class RetrievalInfo(APIModel):
    standalone_query: str
    mode: RetrievalMode = RetrievalMode.HYBRID
    candidates: int = 0
    used_chunks: int = 0
    top_relevance: float = 0.0
    reranker: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ChatRequest(APIModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Restrict retrieval to these documents. Omit/null for all documents.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ChatResponse(APIModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    refused: bool
    refusal_reason: str | None = None
    confidence: Confidence
    grounding_score: float
    citations: list[Citation]
    retrieval: RetrievalInfo
    model: str
    latency_ms: float


class MessageRead(APIModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    created_at: datetime


class ConversationRead(APIModel):
    id: uuid.UUID
    title: str
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationRead):
    messages: list[MessageRead] = Field(default_factory=list)


class ConversationCreate(APIModel):
    title: str | None = Field(default=None, max_length=255)
    document_ids: list[uuid.UUID] = Field(default_factory=list)


class ConversationUpdate(APIModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    document_ids: list[uuid.UUID] | None = None


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True


class SearchHit(APIModel):
    chunk_id: str
    document_id: uuid.UUID
    filename: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    text: str
    score: float
    relevance: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None


class SearchResponse(APIModel):
    query: str
    mode: RetrievalMode
    reranked: bool
    hits: list[SearchHit]
    latency_ms: float
