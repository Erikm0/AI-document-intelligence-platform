"""RAG pipeline orchestration and guardrails with a stubbed retriever."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.rag.llm import ChatMessage, FakeLLM
from app.rag.pipeline import PipelineConfig, RAGPipeline
from app.rag.query_rewriter import QueryRewriter
from app.rag.types import RetrievalResult, RetrievedChunk
from app.schemas.chat import Confidence

LEAVE = RetrievedChunk(
    chunk_id="h:0",
    document_id="h",
    filename="handbook.pdf",
    page_start=4,
    page_end=4,
    text="Full-time employees receive 26 days of paid annual leave per calendar year.",
    relevance=0.97,
    semantic_score=0.8,
)
REMOTE = RetrievedChunk(
    chunk_id="h:1",
    document_id="h",
    filename="handbook.pdf",
    page_start=2,
    page_end=2,
    text="Employees may work remotely for up to three days per week.",
    relevance=0.4,
    semantic_score=0.6,
)


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(
        self, query: str, *, top_k: int, document_ids=None, **_: object
    ) -> RetrievalResult:  # type: ignore[no-untyped-def]
        self.queries.append(query)
        return RetrievalResult(
            query=query, chunks=self.chunks[:top_k], candidates=len(self.chunks), reranker="stub"
        )


def make_pipeline(
    chunks: list[RetrievedChunk],
    responder: Callable[[list[ChatMessage]], str] | None = None,
    **config: object,
) -> tuple[RAGPipeline, FakeLLM, StubRetriever]:
    llm = FakeLLM(responder=responder)
    retriever = StubRetriever(chunks)
    pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        llm=llm,
        rewriter=QueryRewriter(llm),
        config=PipelineConfig(min_relevance=0.1, min_semantic=0.5, **config),  # type: ignore[arg-type]
    )
    return pipeline, llm, retriever


def test_grounded_answer_with_citations() -> None:
    pipeline, _, _ = make_pipeline([LEAVE, REMOTE])
    result = pipeline.answer("How many days of paid annual leave do employees get?")
    assert not result.refused
    assert "26 days" in result.answer and "[1]" in result.answer
    assert result.confidence in (Confidence.HIGH, Confidence.MEDIUM)
    cited = [s for s in result.sources if s.cited]
    assert [s.chunk.chunk_id for s in cited] == ["h:0"]
    assert result.sources[1].cited is False  # retrieved but not cited
    assert "total_ms" in result.timings_ms


def test_no_chunks_refuses_without_calling_llm() -> None:
    pipeline, llm, _ = make_pipeline([])
    result = pipeline.answer("Anything?")
    assert result.refused and result.refusal_reason == "no_relevant_chunks"
    assert llm.calls == []


def test_low_relevance_refuses_without_calling_llm_and_shows_near_misses() -> None:
    weak = RetrievedChunk(
        chunk_id="w",
        document_id="w",
        filename="w.txt",
        text="Unrelated.",
        relevance=0.01,
        semantic_score=0.3,
    )
    pipeline, llm, _ = make_pipeline([weak])
    result = pipeline.answer("Who won the World Cup?")
    assert result.refused and result.refusal_reason == "low_retrieval_relevance"
    assert llm.calls == []
    assert result.sources and not any(s.cited for s in result.sources)
    assert result.confidence == Confidence.NONE


def test_semantic_signal_rescues_paraphrase() -> None:
    paraphrase = RetrievedChunk(
        chunk_id="p",
        document_id="p",
        filename="p.txt",
        relevance=0.0001,
        semantic_score=0.58,
        text="Up to five unused leave days may be carried over into the next year.",
    )
    pipeline, llm, _ = make_pipeline([paraphrase], responder=lambda m: "Five days carry over [1].")
    result = pipeline.answer("What happens to vacation I don't use?")
    assert not result.refused
    assert len(llm.calls) == 1


def test_model_refusal_is_respected() -> None:
    pipeline, _, _ = make_pipeline([LEAVE], responder=lambda m: "INSUFFICIENT_EVIDENCE")
    result = pipeline.answer("What is the dental coverage?")
    assert result.refused and result.refusal_reason == "model_insufficient_evidence"
    assert result.answer.startswith("I couldn't find enough evidence")


@pytest.mark.parametrize(
    "hallucination",
    [
        "Paris is the capital of France.",  # world knowledge, no citation
        "The CEO earns 4 million euros per year [3].",  # cites a source that does not exist
    ],
)
def test_hallucinations_are_refused(hallucination: str) -> None:
    pipeline, _, _ = make_pipeline([LEAVE], responder=lambda m: hallucination)
    result = pipeline.answer("Tell me something")
    assert result.refused
    assert result.diagnostics.get("raw_answer") == hallucination


def test_fabricated_numbers_lower_confidence() -> None:
    pipeline, _, _ = make_pipeline(
        [LEAVE], responder=lambda m: "Employees receive 40 days of paid annual leave per year [1]."
    )
    result = pipeline.answer("How much annual leave?")
    assert not result.refused
    assert result.confidence == Confidence.LOW
    assert result.diagnostics["unsupported_numbers"] == ["40"]


def test_strict_grounding_refuses_weak_answers() -> None:
    pipeline, _, _ = make_pipeline(
        [LEAVE],
        responder=lambda m: "Managers approve bonuses quarterly via spreadsheets [1].",
        strict_grounding=True,
    )
    result = pipeline.answer("How are bonuses approved?")
    assert result.refused and result.refusal_reason == "weak_grounding"


def test_follow_up_is_rewritten_for_retrieval() -> None:
    def responder(messages: list[ChatMessage]) -> str:
        if "Standalone question:" in messages[-1].content:
            return "How many remote work days are allowed per week?"
        return "Up to three days per week [2]."

    pipeline, _, retriever = make_pipeline([LEAVE, REMOTE], responder=responder)
    from app.rag.prompts import HistoryTurn

    result = pipeline.answer(
        "And what about remote work?",
        history=[HistoryTurn("user", "How much leave?"), HistoryTurn("assistant", "26 days [1].")],
    )
    assert retriever.queries == ["How many remote work days are allowed per week?"]
    assert result.standalone_query == "How many remote work days are allowed per week?"
    assert not result.refused


def test_stream_emits_retrieval_tokens_and_final() -> None:
    pipeline, _, _ = make_pipeline([LEAVE])
    events = list(pipeline.stream("How many days of paid annual leave do employees get?"))
    types = [e.type for e in events]
    assert types[0] == "retrieval" and types[-1] == "final"
    assert types.count("token") >= 3
    streamed = "".join(e.data for e in events if e.type == "token")
    assert "26 days" in streamed and not events[-1].data.refused


def test_stream_early_refusal_has_no_tokens() -> None:
    pipeline, _, _ = make_pipeline([])
    assert [e.type for e in pipeline.stream("Anything?")] == ["retrieval", "final"]
