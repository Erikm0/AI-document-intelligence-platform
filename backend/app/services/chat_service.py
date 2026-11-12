"""Chat use-case: conversation state + RAG pipeline + persistence.

Database sessions are deliberately *not* held open while the LLM generates, so
slow generations never pin a connection-pool slot.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import (
    AppError,
    ConversationNotFoundError,
    DocumentNotFoundError,
    DocumentNotReadyError,
)
from app.db.models import DocumentStatus, MessageRole
from app.db.session import Database
from app.rag.pipeline import RAGAnswer, RAGPipeline
from app.rag.prompts import NO_DOCUMENTS_MESSAGE, HistoryTurn
from app.repositories.conversations import ConversationRepository
from app.repositories.documents import DocumentRepository
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    Confidence,
    RetrievalInfo,
    RetrievalMode,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Context:
    conversation_id: uuid.UUID | None
    history: list[HistoryTurn]
    scope: list[str] | None
    no_documents: bool


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 80 else title[:77].rstrip() + "..."


def citations_from(answer: RAGAnswer) -> list[Citation]:
    return [
        Citation(
            index=ref.index,
            chunk_id=ref.chunk.chunk_id,
            document_id=uuid.UUID(ref.chunk.document_id),
            filename=ref.chunk.filename,
            page_start=ref.chunk.page_start,
            page_end=ref.chunk.page_end,
            section=ref.chunk.section,
            text=ref.chunk.text,
            relevance=round(min(max(ref.chunk.relevance or 0.0, 0.0), 1.0), 4),
            cited=ref.cited,
        )
        for ref in answer.sources
    ]


def retrieval_info(answer: RAGAnswer) -> RetrievalInfo:
    return RetrievalInfo(
        standalone_query=answer.standalone_query,
        mode=RetrievalMode.HYBRID,
        candidates=answer.retrieval.candidates,
        used_chunks=sum(1 for s in answer.sources if s.cited),
        top_relevance=round(answer.retrieval.top_relevance, 4),
        reranker=answer.retrieval.reranker,
        timings_ms=answer.timings_ms,
    )


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class ChatService:
    def __init__(self, db: Database, pipeline: RAGPipeline, *, history_turns: int = 4) -> None:
        self.db = db
        self.pipeline = pipeline
        self.history_turns = history_turns

    # ------------------------------------------------------------------------------
    def _load_context(self, request: ChatRequest) -> _Context:
        with self.db.session() as session:
            conversations = ConversationRepository(session)
            documents = DocumentRepository(session)

            conversation = None
            if request.conversation_id is not None:
                conversation = conversations.get(request.conversation_id)
                if conversation is None:
                    raise ConversationNotFoundError(
                        f"Conversation {request.conversation_id} not found"
                    )

            history: list[HistoryTurn] = []
            if conversation is not None:
                history = [
                    HistoryTurn(role=m.role.value, content=m.content)
                    for m in conversations.recent_messages(conversation.id, self.history_turns * 2)
                ]

            # Scope precedence: explicit request scope ([] means "all documents")
            # > the conversation's saved scope > all documents.
            explicit = request.document_ids
            if explicit:
                requested = list(dict.fromkeys(explicit))
                found = documents.get_many(requested)
                missing = {str(i) for i in requested} - {str(d.id) for d in found}
                if missing:
                    raise DocumentNotFoundError(
                        "Some selected documents do not exist",
                        details={"missing": sorted(missing)},
                    )
                ready = [str(d.id) for d in found if d.status == DocumentStatus.READY]
                if not ready:
                    raise DocumentNotReadyError(
                        "None of the selected documents have finished processing",
                        details={"document_ids": [str(i) for i in requested]},
                    )
                return _Context(conversation.id if conversation else None, history, ready, False)

            if explicit is None and conversation is not None and conversation.document_ids:
                found = documents.get_many(uuid.UUID(i) for i in conversation.document_ids)
                ready = [str(d.id) for d in found if d.status == DocumentStatus.READY]
                if ready:
                    return _Context(conversation.id, history, ready, False)

            no_docs = not documents.ready_ids()
            return _Context(conversation.id if conversation else None, history, None, no_docs)

    def _persist(
        self,
        request: ChatRequest,
        ctx: _Context,
        answer: RAGAnswer | None,
        response: dict[str, Any],
    ) -> tuple[uuid.UUID, uuid.UUID]:
        with self.db.session() as session:
            repo = ConversationRepository(session)
            existing = repo.get(ctx.conversation_id) if ctx.conversation_id else None
            if existing is None:  # new conversation (or deleted mid-request)
                conversation = repo.create(_title_from(request.question), request.document_ids)
            else:
                conversation = existing
                if request.document_ids is not None:
                    conversation.document_ids = [str(d) for d in request.document_ids]
            repo.add_message(conversation, MessageRole.USER, request.question)
            metadata = {
                "confidence": response["confidence"],
                "grounding_score": response["grounding_score"],
                "refusal_reason": response["refusal_reason"],
                "retrieval": response["retrieval"],
                "model": response["model"],
                "latency_ms": response["latency_ms"],
            }
            if answer is not None and answer.diagnostics:
                metadata["diagnostics"] = answer.diagnostics
            message = repo.add_message(
                conversation,
                MessageRole.ASSISTANT,
                response["answer"],
                citations=response["citations"],
                refused=response["refused"],
                metadata=metadata,
            )
            return conversation.id, message.id

    def _build_response(self, answer: RAGAnswer) -> dict[str, Any]:
        return {
            "answer": answer.answer,
            "refused": answer.refused,
            "refusal_reason": answer.refusal_reason,
            "confidence": answer.confidence.value,
            "grounding_score": answer.grounding_score,
            "citations": [c.model_dump(mode="json") for c in citations_from(answer)],
            "retrieval": retrieval_info(answer).model_dump(mode="json"),
            "model": answer.model,
            "latency_ms": answer.timings_ms.get("total_ms", 0.0),
        }

    def _no_documents_response(self, request: ChatRequest, started: float) -> dict[str, Any]:
        return {
            "answer": NO_DOCUMENTS_MESSAGE,
            "refused": True,
            "refusal_reason": "no_documents",
            "confidence": Confidence.NONE.value,
            "grounding_score": 0.0,
            "citations": [],
            "retrieval": RetrievalInfo(standalone_query=request.question).model_dump(mode="json"),
            "model": self.pipeline.llm.model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # ------------------------------------------------------------------------------
    def ask(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        ctx = self._load_context(request)
        answer: RAGAnswer | None = None
        if ctx.no_documents:
            body = self._no_documents_response(request, started)
        else:
            answer = self.pipeline.answer(
                request.question, history=ctx.history, document_ids=ctx.scope, top_k=request.top_k
            )
            body = self._build_response(answer)
        conversation_id, message_id = self._persist(request, ctx, answer, body)
        return ChatResponse(conversation_id=conversation_id, message_id=message_id, **body)

    def ask_stream(self, request: ChatRequest) -> Iterator[str]:
        """Server-Sent Events: ``sources`` -> ``token``* -> ``done`` (or ``error``).

        Request validation (unknown conversation / documents) happens eagerly so
        those errors still produce proper HTTP status codes; failures after the
        stream has started are reported as an ``error`` event.
        """
        started = time.perf_counter()
        ctx = self._load_context(request)
        return self._stream(request, ctx, started)

    def _stream(self, request: ChatRequest, ctx: _Context, started: float) -> Iterator[str]:
        try:
            if ctx.no_documents:
                body = self._no_documents_response(request, started)
                conversation_id, message_id = self._persist(request, ctx, None, body)
                yield sse(
                    "done", {"conversation_id": conversation_id, "message_id": message_id, **body}
                )
                return

            final: RAGAnswer | None = None
            for event in self.pipeline.stream(
                request.question, history=ctx.history, document_ids=ctx.scope, top_k=request.top_k
            ):
                if event.type == "retrieval":
                    prepared = event.data
                    yield sse(
                        "sources",
                        {
                            "standalone_query": prepared.standalone_query,
                            "refused_early": prepared.refused_early,
                            "sources": [
                                {
                                    "index": i,
                                    "chunk_id": c.chunk_id,
                                    "document_id": c.document_id,
                                    "filename": c.filename,
                                    "page_start": c.page_start,
                                    "page_end": c.page_end,
                                    "section": c.section,
                                    "relevance": round(c.relevance or 0.0, 4),
                                }
                                for i, c in enumerate(prepared.evidence, start=1)
                            ],
                        },
                    )
                elif event.type == "token":
                    yield sse("token", {"t": event.data})
                elif event.type == "final":
                    final = event.data

            assert final is not None
            body = self._build_response(final)
            conversation_id, message_id = self._persist(request, ctx, final, body)
            yield sse(
                "done", {"conversation_id": conversation_id, "message_id": message_id, **body}
            )
        except AppError as exc:
            logger.warning("Chat stream failed: %s", exc.message)
            yield sse(
                "error", {"code": exc.code, "message": exc.message, "status": exc.status_code}
            )
        except Exception:
            logger.exception("Chat stream crashed")
            yield sse(
                "error",
                {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "status": 500,
                },
            )
