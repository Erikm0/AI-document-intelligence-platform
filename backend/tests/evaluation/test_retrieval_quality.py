"""Retrieval-quality and hallucination-resistance regression gates.

Runs the full evaluation harness (real parsers, chunker, BM25, fusion, reranker,
guardrails) over the synthetic corpus. The default run uses the dependency-free
stack so it is fast and deterministic in CI; set ``RUN_ML_EVAL=1`` to also run
it with the real Sentence-Transformers embedder and cross-encoder.
"""

from __future__ import annotations

import os

import pytest

from app.core.config import EmbeddingProvider, RerankerProvider
from evaluation.harness import EvalConfig, EvalReport, run_evaluation

pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return run_evaluation(EvalConfig())


def test_ingestion_produces_page_aware_chunks(report: EvalReport) -> None:
    assert report.ingestion["documents"] == 4
    assert report.ingestion["pages"] == 11
    assert report.ingestion["chunks"] >= 10


def test_hybrid_rerank_quality_gate(report: EvalReport) -> None:
    metrics = report.retrieval["hybrid + rerank"]
    assert metrics["recall@5"] >= 0.95
    assert metrics["recall@1"] >= 0.85
    assert metrics["MRR"] >= 0.90
    assert metrics["page_acc"] >= 0.95


def test_hybrid_beats_single_retrievers(report: EvalReport) -> None:
    hybrid = report.retrieval["hybrid (RRF)"]["MRR"]
    assert hybrid >= report.retrieval["semantic"]["MRR"]
    assert hybrid >= report.retrieval["lexical (BM25)"]["MRR"]
    assert report.retrieval["hybrid + rerank"]["MRR"] >= hybrid


def test_no_hallucinations_on_unanswerable_questions(report: EvalReport) -> None:
    assert report.answers["refusal_rate (unanswerable)"] == 1.0
    hallucinations = [f for f in report.failures if f["kind"] == "hallucination"]
    assert hallucinations == []


def test_answers_cite_the_right_evidence(report: EvalReport) -> None:
    assert report.answers["citation_accuracy (answered)"] >= 0.95
    assert report.answers["mean_grounding_score"] >= 0.8
    # The extractive fake LLM cannot paraphrase, so this floor is conservative;
    # with a real LLM the answer rate is much higher (see README).
    assert report.answers["answer_rate (answerable)"] >= 0.5


def test_report_renders_markdown(report: EvalReport) -> None:
    markdown = report.to_markdown()
    assert "| hybrid + rerank |" in markdown and "hallucination_rate" in markdown


@pytest.mark.ml
@pytest.mark.skipif(
    not os.environ.get("RUN_ML_EVAL"), reason="set RUN_ML_EVAL=1 (downloads models)"
)
def test_real_models_quality_gate() -> None:
    pytest.importorskip("sentence_transformers")
    real = run_evaluation(
        EvalConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            reranker_provider=RerankerProvider.CROSS_ENCODER,
        )
    )
    metrics = real.retrieval["hybrid + rerank"]
    assert metrics["recall@1"] >= 0.9 and metrics["MRR"] >= 0.95
    assert real.answers["citation_accuracy (answered)"] >= 0.95
    # Every off-topic question must be stopped by the evidence gate or the LLM.
    assert real.answers["refusal_rate (unanswerable)"] >= 0.9
