"""Chat, conversations, search and streaming through the HTTP API."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.container import Container
from app.core.exceptions import LLMUnavailableError
from app.rag.llm import ChatMessage, FakeLLM
from tests.factories import SAMPLE_TXT

pytestmark = pytest.mark.integration

API = "/api/v1"

POLICY_TXT = """Remote Work Policy

Employees may work remotely for up to three days per week with manager approval.

Annual Leave

Full-time employees receive 26 days of paid annual leave per calendar year.
"""


@pytest.fixture
def docs(client: TestClient) -> dict[str, str]:
    orion = client.post(
        f"{API}/documents", files={"file": ("orion.txt", SAMPLE_TXT.encode())}
    ).json()
    policy = client.post(
        f"{API}/documents", files={"file": ("policy.txt", POLICY_TXT.encode())}
    ).json()
    return {"orion": orion["id"], "policy": policy["id"]}


def ask(client: TestClient, question: str, **extra: Any) -> dict[str, Any]:
    response = client.post(f"{API}/chat", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_answer_with_page_citations(client: TestClient, docs: dict[str, str]) -> None:
    body = ask(client, "How many craters with water ice did Vega-9 detect?")
    assert not body["refused"]
    assert "14 craters" in body["answer"]
    cited = [c for c in body["citations"] if c["cited"]]
    assert cited and cited[0]["filename"] == "orion.txt"
    assert cited[0]["page_start"] == 2
    assert f"[{cited[0]['index']}]" in body["answer"]
    assert body["confidence"] in ("high", "medium")
    assert body["retrieval"]["reranker"] == "lexical-coverage"
    assert body["retrieval"]["timings_ms"]["total_ms"] > 0


def test_unsupported_question_is_refused(client: TestClient, docs: dict[str, str]) -> None:
    body = ask(client, "Who won the 2022 FIFA World Cup final?")
    assert body["refused"] is True
    assert body["confidence"] == "none"
    assert not any(c["cited"] for c in body["citations"])
    assert "couldn't find enough evidence" in body["answer"]


def test_no_documents_refusal(client: TestClient) -> None:
    body = ask(client, "What is in my documents?")
    assert body["refused"] and body["refusal_reason"] == "no_documents"


def test_document_scope_is_respected(client: TestClient, docs: dict[str, str]) -> None:
    scoped = ask(
        client, "How many days of paid annual leave do employees get?", document_ids=[docs["orion"]]
    )
    assert scoped["refused"]  # the leave policy is in the other document
    assert all(c["document_id"] == docs["orion"] for c in scoped["citations"])

    unscoped = ask(client, "How many days of paid annual leave do employees get?")
    assert not unscoped["refused"] and "26 days" in unscoped["answer"]


def test_scope_validation(client: TestClient, docs: dict[str, str]) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"{API}/chat", json={"question": "hi there", "document_ids": [missing]})
    assert response.status_code == 404
    assert response.json()["error"]["details"]["missing"] == [missing]

    broken = client.post(f"{API}/documents", files={"file": ("bad.pdf", b"%PDF-1.4 junk")}).json()
    response = client.post(
        f"{API}/chat", json={"question": "hi there", "document_ids": [broken["id"]]}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "document_not_ready"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 4001},
        {"question": "ok", "top_k": 0},
        {},
    ],
)
def test_chat_validation(client: TestClient, payload: dict[str, Any]) -> None:
    response = client.post(f"{API}/chat", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_conversation_history_and_follow_up(
    client: TestClient, docs: dict[str, str], fake_llm: FakeLLM
) -> None:
    first = ask(client, "How many craters with water ice did Vega-9 detect?")
    conversation_id = first["conversation_id"]

    second = ask(client, "And what is planned next for it?", conversation_id=conversation_id)
    assert second["conversation_id"] == conversation_id
    # The follow-up triggered a query-rewrite call that saw the earlier turn.
    rewrite_prompts = [
        m[-1].content for m in fake_llm.calls if "Standalone question:" in m[-1].content
    ]
    assert rewrite_prompts and "Vega-9" in rewrite_prompts[-1]
    # The answer prompt carried the history as prior chat turns.
    answer_call = fake_llm.calls[-1]
    assert [m.role for m in answer_call[:3]] == ["system", "user", "assistant"]

    detail = client.get(f"{API}/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["citations"]
    assert detail["title"].startswith("How many craters")

    listing = client.get(f"{API}/conversations").json()
    assert listing["total"] == 1 and listing["items"][0]["message_count"] == 4


def test_conversation_crud(client: TestClient, docs: dict[str, str]) -> None:
    created = client.post(
        f"{API}/conversations", json={"title": "Research", "document_ids": [docs["policy"]]}
    )
    assert created.status_code == 201
    cid = created.json()["id"]

    # Conversation scope is used when the request does not specify documents.
    body = ask(client, "How many craters with water ice did Vega-9 detect?", conversation_id=cid)
    assert body["refused"]

    # An explicit empty scope overrides the saved one and means "all documents".
    widened = ask(
        client,
        "How many craters with water ice did Vega-9 detect?",
        conversation_id=cid,
        document_ids=[],
    )
    assert not widened["refused"]
    assert client.get(f"{API}/conversations/{cid}").json()["document_ids"] == []

    patched = client.patch(
        f"{API}/conversations/{cid}", json={"title": "Renamed", "document_ids": []}
    )
    assert patched.json()["title"] == "Renamed" and patched.json()["document_ids"] == []
    assert client.delete(f"{API}/conversations/{cid}").status_code == 204
    assert client.get(f"{API}/conversations/{cid}").status_code == 404
    missing = client.post(f"{API}/chat", json={"question": "hello", "conversation_id": cid})
    assert (
        missing.status_code == 404 and missing.json()["error"]["code"] == "conversation_not_found"
    )


def test_streaming_chat(client: TestClient, docs: dict[str, str]) -> None:
    with client.stream(
        "POST",
        f"{API}/chat/stream",
        json={"question": "How many craters with water ice did Vega-9 detect?"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = parse_sse(text)
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert "token" in names
    streamed = "".join(d["t"] for e, d in events if e == "token")
    done = events[-1][1]
    assert "14 craters" in streamed and done["answer"].startswith(streamed.split(" [")[0][:20])
    assert done["message_id"] and done["conversation_id"]
    assert events[0][1]["sources"][0]["filename"] == "orion.txt"


def test_streaming_validation_errors_are_http_errors(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        f"{API}/chat/stream", json={"question": "hi", "conversation_id": missing}
    )
    assert response.status_code == 404


def test_llm_outage_returns_503(
    client: TestClient, container: Container, docs: dict[str, str]
) -> None:
    def down(_: list[ChatMessage]) -> str:
        raise LLMUnavailableError("Cannot reach Ollama at http://ollama:11434")

    container.pipeline.llm = FakeLLM(responder=down)
    response = client.post(
        f"{API}/chat", json={"question": "How many craters with water ice did Vega-9 detect?"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"

    with client.stream(
        "POST",
        f"{API}/chat/stream",
        json={"question": "How many craters with water ice did Vega-9 detect?"},
    ) as streamed:
        events = parse_sse("".join(streamed.iter_text()))
    assert events[-1][0] == "error" and events[-1][1]["code"] == "llm_unavailable"


def test_hallucinating_llm_is_caught_end_to_end(
    client: TestClient, container: Container, docs: dict[str, str]
) -> None:
    container.pipeline.llm = FakeLLM(
        responder=lambda m: "Vega-9 was built by NASA in 1998 for 4 billion dollars."
    )
    body = ask(client, "How many craters with water ice did Vega-9 detect?")
    assert body["refused"] and body["refusal_reason"] == "no_supporting_citations"
    detail = client.get(f"{API}/conversations/{body['conversation_id']}").json()
    assistant = detail["messages"][-1]
    assert assistant["refused"] is True
    assert "NASA" in assistant["metadata"]["diagnostics"]["raw_answer"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_search_endpoint(client: TestClient, docs: dict[str, str], mode: str) -> None:
    response = client.post(
        f"{API}/search", json={"query": "Shackleton crater ice deposit", "mode": mode, "top_k": 3}
    )
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits and "Shackleton" in hits[0]["text"]
    assert hits[0]["page_start"] == 2


def test_deleted_document_disappears_from_answers(client: TestClient, docs: dict[str, str]) -> None:
    client.delete(f"{API}/documents/{docs['orion']}")
    body = ask(client, "How many craters with water ice did Vega-9 detect?")
    assert body["refused"]
