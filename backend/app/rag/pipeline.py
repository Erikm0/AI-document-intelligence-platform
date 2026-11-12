"""The question-answering pipeline.

question + history
    -> query rewriting (follow-ups only)
    -> hybrid retrieval (dense + BM25 -> RRF) -> cross-encoder rerank
    -> evidence gate (refuse early if nothing is relevant enough)
    -> grounded generation with numbered sources (Ollama)
    -> guardrails: refusal detection, citation validation/repair, grounding score
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.rag.grounding import GroundingReport, evaluate_answer
from app.rag.llm import ChatMessage, LLMClient
from app.rag.prompts import (
    REFUSAL_MESSAGE,
    HistoryTurn,
    build_answer_messages,
    select_context,
)
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import HybridRetriever
from app.rag.types import RetrievalResult, RetrievedChunk
from app.schemas.chat import Confidence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    top_k: int = 5
    min_relevance: float = 0.1
    min_semantic: float | None = None
    max_context_chars: int = 12_000
    grounding_high: float = 0.6
    grounding_medium: float = 0.35
    strict_grounding: bool = False


@dataclass(slots=True)
class PreparedQuery:
    question: str
    standalone_query: str
    retrieval: RetrievalResult
    evidence: list[RetrievedChunk]
    messages: list[ChatMessage]
    refusal_reason: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def refused_early(self) -> bool:
        return self.refusal_reason is not None


@dataclass(slots=True)
class SourceRef:
    index: int
    chunk: RetrievedChunk
    cited: bool


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    refused: bool
    refusal_reason: str | None
    confidence: Confidence
    grounding_score: float
    sources: list[SourceRef]
    standalone_query: str
    retrieval: RetrievalResult
    model: str
    timings_ms: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    """Events emitted by :meth:`RAGPipeline.stream`."""

    type: str  # "retrieval" | "token" | "final"
    data: Any


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMClient,
        rewriter: QueryRewriter,
        config: PipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.rewriter = rewriter
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------------------
    def prepare(
        self,
        question: str,
        *,
        history: Sequence[HistoryTurn] = (),
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> PreparedQuery:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        standalone = self.rewriter.rewrite(question, history)
        timings["rewrite_ms"] = _ms(t0)

        retrieval = self.retriever.retrieve(
            standalone, top_k=top_k or self.config.top_k, document_ids=document_ids
        )
        timings.update(retrieval.timings_ms)

        evidence = [c for c in retrieval.chunks if self.is_evidence(c)]
        evidence = select_context(evidence, self.config.max_context_chars)
        refusal_reason = None
        if not retrieval.chunks:
            refusal_reason = "no_relevant_chunks"
        elif not evidence:
            refusal_reason = "low_retrieval_relevance"

        messages = (
            build_answer_messages(question, evidence, history, standalone_query=standalone)
            if evidence
            else []
        )
        return PreparedQuery(
            question=question,
            standalone_query=standalone,
            retrieval=retrieval,
            evidence=evidence,
            messages=messages,
            refusal_reason=refusal_reason,
            timings_ms=timings,
        )

    def is_evidence(self, chunk: RetrievedChunk) -> bool:
        """Two-signal evidence gate: reranker relevance OR dense similarity.

        The gate only rejects clearly off-topic material; whether the evidence
        actually *answers* the question is decided by the LLM and the
        post-generation guardrails.
        """
        if (chunk.relevance or 0.0) >= self.config.min_relevance:
            return True
        return (
            self.config.min_semantic is not None
            and chunk.semantic_score is not None
            and chunk.semantic_score >= self.config.min_semantic
        )

    def answer(
        self,
        question: str,
        *,
        history: Sequence[HistoryTurn] = (),
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        prepared = self.prepare(question, history=history, document_ids=document_ids, top_k=top_k)
        if prepared.refused_early:
            return self._early_refusal(prepared, started)
        t0 = time.perf_counter()
        raw = self.llm.chat(prepared.messages)
        prepared.timings_ms["generate_ms"] = _ms(t0)
        return self.finalize(prepared, raw, started)

    def stream(
        self,
        question: str,
        *,
        history: Sequence[HistoryTurn] = (),
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> Iterator[StreamEvent]:
        started = time.perf_counter()
        prepared = self.prepare(question, history=history, document_ids=document_ids, top_k=top_k)
        yield StreamEvent("retrieval", prepared)
        if prepared.refused_early:
            yield StreamEvent("final", self._early_refusal(prepared, started))
            return
        t0 = time.perf_counter()
        parts: list[str] = []
        for token in self.llm.stream_chat(prepared.messages):
            parts.append(token)
            yield StreamEvent("token", token)
        prepared.timings_ms["generate_ms"] = _ms(t0)
        yield StreamEvent("final", self.finalize(prepared, "".join(parts), started))

    # ------------------------------------------------------------------------------
    def finalize(self, prepared: PreparedQuery, raw_answer: str, started: float) -> RAGAnswer:
        report: GroundingReport = evaluate_answer(raw_answer, prepared.evidence)
        refused = report.refused
        reason = report.refusal_reason
        if (
            not refused
            and self.config.strict_grounding
            and report.grounding_score < self.config.grounding_medium
        ):
            refused, reason = True, "weak_grounding"

        cited = set(report.cited_indices)
        sources = [
            SourceRef(index=i, chunk=c, cited=(i in cited) and not refused)
            for i, c in enumerate(prepared.evidence, start=1)
        ]
        confidence = Confidence.NONE if refused else self._confidence(report, prepared)
        diagnostics = {
            "raw_answer": raw_answer
            if (refused or report.repaired or report.invalid_citations)
            else None,
            "invalid_citations": report.invalid_citations,
            "citations_repaired": report.repaired,
            "unsupported_numbers": report.unsupported_numbers,
        }
        timings = dict(prepared.timings_ms)
        timings["total_ms"] = _ms(started)
        return RAGAnswer(
            answer=REFUSAL_MESSAGE if refused else report.answer,
            refused=refused,
            refusal_reason=reason,
            confidence=confidence,
            grounding_score=0.0 if refused else report.grounding_score,
            sources=sources,
            standalone_query=prepared.standalone_query,
            retrieval=prepared.retrieval,
            model=self.llm.model,
            timings_ms=timings,
            diagnostics={k: v for k, v in diagnostics.items() if v},
        )

    def _early_refusal(self, prepared: PreparedQuery, started: float) -> RAGAnswer:
        # Show the closest (insufficient) passages so the user can see why.
        near_misses = [
            SourceRef(index=i, chunk=c, cited=False)
            for i, c in enumerate(prepared.retrieval.chunks[:3], start=1)
        ]
        timings = dict(prepared.timings_ms)
        timings["total_ms"] = _ms(started)
        return RAGAnswer(
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=prepared.refusal_reason,
            confidence=Confidence.NONE,
            grounding_score=0.0,
            sources=near_misses,
            standalone_query=prepared.standalone_query,
            retrieval=prepared.retrieval,
            model=self.llm.model,
            timings_ms=timings,
        )

    def _confidence(self, report: GroundingReport, prepared: PreparedQuery) -> Confidence:
        score = report.grounding_score
        if report.unsupported_numbers:
            score = min(score, self.config.grounding_medium - 0.01)
        top = prepared.retrieval.top_relevance
        if score >= self.config.grounding_high and top >= 0.5:
            return Confidence.HIGH
        if score >= self.config.grounding_medium:
            return Confidence.MEDIUM
        return Confidence.LOW


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
