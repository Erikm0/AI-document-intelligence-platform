from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import LLMUnavailableError
from app.rag.llm import ChatMessage, FakeLLM, OllamaClient
from app.rag.prompts import (
    REFUSAL_SENTINEL,
    HistoryTurn,
    build_answer_messages,
    format_source,
    select_context,
)
from app.rag.query_rewriter import QueryRewriter, needs_rewrite
from app.rag.types import RetrievedChunk

MESSAGES = [ChatMessage("user", "hi")]


def _client(handler) -> OllamaClient:  # type: ignore[no-untyped-def]
    return OllamaClient(
        "http://ollama:11434", "llama3.2:3b", transport=httpx.MockTransport(handler)
    )


class TestOllamaClient:
    def test_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/api/chat"
            assert body["model"] == "llama3.2:3b" and body["stream"] is False
            assert body["options"]["temperature"] == 0.1
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": "Hello [1]"}}
            )

        assert _client(handler).chat(MESSAGES) == "Hello [1]"

    def test_think_flag_is_sent_only_when_configured(self) -> None:
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        _client(handler).chat(MESSAGES)
        OllamaClient(
            "http://o:11434", "m", think=False, transport=httpx.MockTransport(handler)
        ).chat(MESSAGES)
        assert "think" not in bodies[0]
        assert bodies[1]["think"] is False

    def test_stream(self) -> None:
        lines = [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        payload = "\n".join(json.dumps(x) for x in lines).encode()
        client = _client(lambda r: httpx.Response(200, content=payload))
        assert list(client.stream_chat(MESSAGES)) == ["Hel", "lo"]

    def test_missing_model_gives_pull_hint(self) -> None:
        client = _client(lambda r: httpx.Response(404, json={"error": "model not found"}))
        with pytest.raises(LLMUnavailableError, match=r"ollama pull llama3\.2:3b"):
            client.chat(MESSAGES)

    def test_connection_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMUnavailableError, match="Cannot reach Ollama"):
            _client(handler).chat(MESSAGES)

    def test_stream_error_line(self) -> None:
        client = _client(lambda r: httpx.Response(200, content=b'{"error": "out of memory"}\n'))
        with pytest.raises(LLMUnavailableError, match="out of memory"):
            list(client.stream_chat(MESSAGES))

    def test_health(self) -> None:
        ok_client = _client(
            lambda r: httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        )
        assert ok_client.health()[0] is True
        missing = _client(lambda r: httpx.Response(200, json={"models": [{"name": "mistral:7b"}]}))
        ok, detail = missing.health()
        assert ok is False and "not pulled" in detail


class TestPrompts:
    chunk = RetrievedChunk(
        chunk_id="x:0",
        document_id="x",
        filename="report.pdf",
        text="Body text.",
        page_start=3,
        page_end=4,
        section="Results",
    )

    def test_format_source(self) -> None:
        assert (
            format_source(2, self.chunk)
            == "[2] report.pdf | pp. 3-4 | Section: Results\nBody text."
        )

    def test_answer_messages_include_history_sources_and_sentinel(self) -> None:
        messages = build_answer_messages(
            "and the budget?",
            [self.chunk],
            [HistoryTurn("user", "What is Vega-9?"), HistoryTurn("assistant", "An orbiter [1].")],
            standalone_query="What is the Vega-9 budget?",
        )
        assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
        final = messages[-1].content
        assert "[1] report.pdf" in final and REFUSAL_SENTINEL in final
        assert "Interpreted in context as: What is the Vega-9 budget?" in final

    def test_select_context_respects_budget_but_keeps_one(self) -> None:
        big = RetrievedChunk(chunk_id="b", document_id="b", filename="b", text="x" * 5000)
        assert len(select_context([big, big, big], max_chars=6000)) == 1
        assert len(select_context([big], max_chars=10)) == 1


class TestQueryRewriter:
    history = [
        HistoryTurn("user", "What is the leave policy?"),
        HistoryTurn("assistant", "26 days [1]."),
    ]

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What about its exceptions?", True),
            ("and for contractors?", True),
            ("why?", True),
            ("How many days of paid annual leave do employees receive?", False),
        ],
    )
    def test_needs_rewrite(self, question: str, expected: bool) -> None:
        assert needs_rewrite(question) is expected

    def test_rewrites_follow_up(self) -> None:
        llm = FakeLLM(
            responder=lambda m: (
                'Standalone question: "What are the exceptions to the leave policy?"'
            )
        )
        rewriter = QueryRewriter(llm)
        assert rewriter.rewrite("What about its exceptions?", self.history) == (
            "What are the exceptions to the leave policy?"
        )

    def test_no_history_or_disabled_skips_llm(self) -> None:
        llm = FakeLLM()
        assert QueryRewriter(llm).rewrite("What about it?", []) == "What about it?"
        assert (
            QueryRewriter(llm, enabled=False).rewrite("What about it?", self.history)
            == "What about it?"
        )
        assert llm.calls == []

    def test_fallback_on_garbage(self) -> None:
        rewriter = QueryRewriter(FakeLLM(responder=lambda m: ""))
        assert rewriter.rewrite("and its exceptions?", self.history) == (
            "What is the leave policy? and its exceptions?"
        )

    def test_llm_outage_propagates(self) -> None:
        def boom(_: list[ChatMessage]) -> str:
            raise LLMUnavailableError("down")

        with pytest.raises(LLMUnavailableError):
            QueryRewriter(FakeLLM(responder=boom)).rewrite("and its exceptions?", self.history)


class TestFakeLLM:
    def test_extractive_answer_and_refusal(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="a",
                document_id="a",
                filename="hb.txt",
                text="Employees receive 26 days of paid annual leave per year.",
            )
        ]
        answer = FakeLLM().chat(build_answer_messages("How many days of annual leave?", chunks))
        assert answer.endswith("[1]") and "26 days" in answer
        refusal = FakeLLM().chat(build_answer_messages("What is the CEO salary bonus?", chunks))
        assert refusal == REFUSAL_SENTINEL
