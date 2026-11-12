"""Adversarial hallucination tests.

A scripted "LLM" produces the classic failure modes of generative models; the
pipeline's guardrails must refuse or down-rank every one of them. Optional live
tests run the same unanswerable questions against a real Ollama model.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.container import Container
from app.rag.llm import ChatMessage, FakeLLM
from app.schemas.chat import Confidence
from evaluation.harness import EvalConfig, build_eval_container, ingest_corpus, load_dataset

pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def container() -> Iterator[Container]:
    with tempfile.TemporaryDirectory(prefix="docintel-halluc-") as tmp:
        c = build_eval_container(EvalConfig(), Path(tmp))
        ingest_corpus(c, Path(tmp) / "corpus")
        yield c
        c.shutdown()


def run_with(container: Container, responder: Callable[[list[ChatMessage]], str], question: str):  # type: ignore[no-untyped-def]
    original = container.pipeline.llm
    container.pipeline.llm = FakeLLM(responder=responder)
    try:
        return container.pipeline.answer(question)
    finally:
        container.pipeline.llm = original


ANSWERABLE = "How many days of paid annual leave do full-time Aurora employees receive?"


@pytest.mark.parametrize(
    ("name", "output"),
    [
        ("world_knowledge_without_citation", "Most European companies offer 25 days of leave."),
        ("cites_nonexistent_source", "Employees receive 30 days of leave [9]."),
        ("fabricated_source_marker", "According to [Source 12], leave is unlimited."),
        ("empty_output", ""),
        ("explicit_refusal_phrase", "The provided documents do not contain this information."),
    ],
)
def test_guardrails_refuse_unsupported_outputs(
    container: Container, name: str, output: str
) -> None:
    result = run_with(container, lambda _m: output, ANSWERABLE)
    assert result.refused, name
    assert result.answer.startswith("I couldn't find enough evidence")
    assert not any(s.cited for s in result.sources)


def test_invented_numbers_are_flagged(container: Container) -> None:
    result = run_with(
        container,
        lambda _m: "Full-time employees receive 27 days of paid annual leave [1].",
        ANSWERABLE,
    )
    assert not result.refused
    assert result.confidence == Confidence.LOW
    assert "27" in result.diagnostics["unsupported_numbers"]


def test_off_topic_questions_never_reach_the_llm(container: Container) -> None:
    calls: list[list[ChatMessage]] = []

    def spy(messages: list[ChatMessage]) -> str:
        calls.append(messages)
        return "Paris [1]."

    for question in ("What is the capital of France?", "Write a short poem about the ocean."):
        result = run_with(container, spy, question)
        assert result.refused and result.refusal_reason in {
            "low_retrieval_relevance",
            "no_relevant_chunks",
        }
    assert calls == []


def test_every_unanswerable_question_is_refused(container: Container) -> None:
    for item in load_dataset()["unanswerable"]:
        result = container.pipeline.answer(item["question"])
        assert result.refused, f"{item['id']}: {result.answer}"


def test_correct_answer_survives_guardrails(container: Container) -> None:
    result = run_with(
        container,
        lambda _m: (
            "Full-time employees receive 26 days of paid annual leave per calendar year [1]."
        ),
        ANSWERABLE,
    )
    assert not result.refused
    assert result.confidence in (Confidence.HIGH, Confidence.MEDIUM)
    assert result.sources[0].cited and "26 days" in result.sources[0].chunk.text


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_EVAL"), reason="set RUN_LIVE_EVAL=1 with Ollama running"
)
def test_live_ollama_refuses_unanswerable_questions() -> None:
    from app.core.config import LLMProvider

    with tempfile.TemporaryDirectory() as tmp:
        c = build_eval_container(EvalConfig(llm_provider=LLMProvider.OLLAMA), Path(tmp))
        ingest_corpus(c, Path(tmp) / "corpus")
        refused = sum(
            c.pipeline.answer(i["question"]).refused for i in load_dataset()["unanswerable"]
        )
        c.shutdown()
    assert refused >= 9
