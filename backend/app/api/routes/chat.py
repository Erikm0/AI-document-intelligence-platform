from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import ChatServiceDep, ContainerDep
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.schemas.common import ERROR_RESPONSES

router = APIRouter(tags=["chat"], responses=ERROR_RESPONSES)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about your documents",
    description=(
        "Runs the full RAG pipeline (query rewriting, hybrid retrieval, reranking, "
        "grounded generation, citation validation). Pass `conversation_id` to continue "
        "a conversation and `document_ids` to restrict the search. When the documents "
        "do not support an answer the response has `refused: true`."
    ),
)
def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    return service.ask(request)


@router.post(
    "/chat/stream",
    summary="Ask a question - streamed as Server-Sent Events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "`text/event-stream` with events: `sources` (retrieved evidence), "
                "`token` (answer deltas), `done` (final validated ChatResponse) or `error`."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
def chat_stream(request: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    events = service.ask_stream(request)
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieval only (debugging & evaluation)",
    description="Runs semantic, lexical or hybrid retrieval with optional reranking - no LLM.",
)
def search(request: SearchRequest, container: ContainerDep) -> SearchResponse:
    started = time.perf_counter()
    result = container.retriever.retrieve(
        request.query,
        top_k=request.top_k,
        document_ids=[str(d) for d in request.document_ids] if request.document_ids else None,
        mode=request.mode,
        rerank=request.rerank,
    )
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        reranked=request.rerank,
        hits=[
            SearchHit(
                chunk_id=c.chunk_id,
                document_id=c.document_id,  # type: ignore[arg-type]
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section,
                text=c.text,
                score=round(c.score, 6),
                relevance=None if c.relevance is None else round(c.relevance, 6),
                semantic_rank=c.semantic_rank,
                lexical_rank=c.lexical_rank,
            )
            for c in result.chunks
        ],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
