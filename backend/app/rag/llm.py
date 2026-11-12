"""LLM clients: Ollama (local inference) and a deterministic fake for tests/demos."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from app.core.exceptions import LLMUnavailableError
from app.core.text import content_terms, split_sentences

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[ChatMessage], *, temperature: float | None = None) -> str: ...

    def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float | None = None
    ) -> Iterator[str]: ...

    def health(self) -> tuple[bool, str]: ...


class OllamaClient:
    """Minimal, dependency-light client for Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout: float = 180.0,
        think: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    def _payload(
        self, messages: list[ChatMessage], stream: bool, temperature: float | None
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self.model}' is not available in Ollama. Run: ollama pull {self.model}",
                details={"model": self.model},
            )
        if response.status_code >= 400:
            try:
                body = response.read().decode(errors="replace")[:500]
            except Exception:
                body = ""
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {body}".strip()
            )

    def chat(self, messages: list[ChatMessage], *, temperature: float | None = None) -> str:
        try:
            response = self._client.post(
                "/api/chat", json=self._payload(messages, False, temperature)
            )
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        self._raise_for_status(response)
        data = response.json()
        return str(data.get("message", {}).get("content", ""))

    def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float | None = None
    ) -> Iterator[str]:
        try:
            with self._client.stream(
                "POST", "/api/chat", json=self._payload(messages, True, temperature)
            ) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMUnavailableError(f"Ollama error: {data['error']}")
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc

    def _connection_error(self, exc: Exception) -> LLMUnavailableError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMUnavailableError(f"Ollama timed out while generating ({exc})")
        return LLMUnavailableError(
            f"Cannot reach Ollama at {self.base_url}. Is it running? ({exc})",
            details={"base_url": self.base_url},
        )

    def list_models(self) -> list[str]:
        response = self._client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", [])]

    def health(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except Exception as exc:
            return False, f"unreachable: {exc}"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in models and self.model not in models:
            return (
                False,
                f"model '{self.model}' not pulled (available: {', '.join(models) or 'none'})",
            )
        return True, f"model '{self.model}' ready"

    def close(self) -> None:
        self._client.close()


# Generic question words that carry no topic of their own.
_QUESTION_FILLER = {
    "many",
    "much",
    "long",
    "get",
    "happen",
    "kind",
    "type",
    "way",
    "make",
    "use",
    "need",
    "take",
    "anyone",
    "someone",
    "thing",
    "anything",
    "exactly",
    "actual",
    "current",
    "know",
}

_IDENTIFIER_RE = re.compile(r"\b(?:[IVX]{2,}|[A-Za-z]*\d(?:[\w.,]*\w)?)\b")

_SOURCE_BLOCK_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|^Question:)", re.S | re.M)


class FakeLLM:
    """Deterministic stand-in for an LLM.

    * Query rewriting prompts -> returns the follow-up question unchanged.
    * Answer prompts -> extractive answer: the source sentences that best overlap
      the question, each with a [n] citation; ``INSUFFICIENT_EVIDENCE`` when no
      sentence overlaps enough.

    Tests can inject a ``responder`` to script arbitrary (e.g. hallucinated)
    outputs and assert that the guardrails catch them.
    """

    model = "fake-extractive"

    def __init__(self, responder: Callable[[list[ChatMessage]], str] | None = None) -> None:
        self.responder = responder
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: list[ChatMessage], *, temperature: float | None = None) -> str:
        self.calls.append(messages)
        if self.responder is not None:
            return self.responder(messages)
        prompt = messages[-1].content
        if "Standalone question:" in prompt:
            match = re.search(r"Follow-up question:\s*(.+?)\s*Standalone question:", prompt, re.S)
            return match.group(1).strip() if match else prompt
        return self._extractive_answer(prompt)

    def stream_chat(
        self, messages: list[ChatMessage], *, temperature: float | None = None
    ) -> Iterator[str]:
        text = self.chat(messages, temperature=temperature)
        yield from re.findall(r"\S+\s*", text)

    def health(self) -> tuple[bool, str]:
        return True, "fake LLM (no inference)"

    @staticmethod
    def _extractive_answer(prompt: str) -> str:
        q_match = re.search(r"^Question:\s*(.+)$", prompt, re.M)
        question = q_match.group(1) if q_match else ""
        q_terms = set(content_terms(question))
        if not q_terms:
            return "INSUFFICIENT_EVIDENCE"

        sources: list[tuple[int, set[str], list[str], str]] = []
        vocabulary: set[str] = set()
        for match in _SOURCE_BLOCK_RE.finditer(prompt):
            header = match.group(0).split("\n", 1)[0]
            passage = match.group(0).lower()
            header_terms = set(content_terms(header))
            sentences = [
                s
                for line in match.group(2).split("\n")
                for s in split_sentences(line.strip())
                # Skip headings: real answers come from full sentences.
                if len(s.split()) >= 5 and s.rstrip()[-1:] in ".!?)"
            ]
            vocabulary |= header_terms | set(content_terms(match.group(2)))
            sources.append((int(match.group(1)), header_terms, sentences, passage))

        # A question about something the sources never mention ("Phase III",
        # "dental insurance") must not be answered with a loosely related sentence.
        if len(q_terms - vocabulary - _QUESTION_FILLER) > 1:
            return "INSUFFICIENT_EVIDENCE"

        # Identifiers in the question ("Phase III", "Helios-7", "2027", "C3") must
        # literally occur in the sources, like a careful reader would check.
        all_passages = "\n".join(p for *_, p in sources)
        for ident in {t.lower() for t in _IDENTIFIER_RE.findall(question)}:
            if not re.search(rf"(?<![a-z0-9]){re.escape(ident)}(?![a-z0-9])", all_passages):
                return "INSUFFICIENT_EVIDENCE"

        scored: list[tuple[float, int, str]] = []
        for idx, header_terms, sentences, _passage in sources:
            for sentence in sentences:
                terms = set(content_terms(sentence)) | header_terms
                overlap = len(q_terms & terms) / len(q_terms)
                scored.append((overlap, idx, sentence))
        scored.sort(key=lambda s: s[0], reverse=True)
        best = [s for s in scored if s[0] >= 0.6][:2]
        if not best:
            return "INSUFFICIENT_EVIDENCE"
        return " ".join(f"{sentence.rstrip()} [{idx}]" for _, idx, sentence in best)
