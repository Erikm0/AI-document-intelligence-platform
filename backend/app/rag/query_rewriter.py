"""Conversational query rewriting ("condense question").

Follow-ups like "and what about its penalties?" retrieve poorly on their own.
When the question depends on earlier turns we ask the LLM to rewrite it into a
standalone query *for retrieval only*; the original wording is still shown to the
LLM when answering.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.core.exceptions import LLMUnavailableError
from app.core.text import content_terms, tokenize
from app.rag.llm import LLMClient
from app.rag.prompts import HistoryTurn, build_rewrite_messages

logger = logging.getLogger(__name__)

_REFERRING = {
    "it",
    "its",
    "it's",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "there",
    "former",
    "latter",
    "above",
    "previous",
    "same",
    "mentioned",
    "earlier",
    "also",
    "else",
    "more",
    "other",
    "another",
    "one",
    "ones",
}
_FOLLOW_UP_START = re.compile(
    r"^(and|but|so|also|what about|how about|why|what else|anything else|more|elaborate|explain)\b",
    re.I,
)


def needs_rewrite(question: str) -> bool:
    words = tokenize(question)
    if not words:
        return False
    if _FOLLOW_UP_START.match(question.strip()):
        return True
    if any(w in _REFERRING for w in words):
        return True
    return len(content_terms(question)) < 3


class QueryRewriter:
    def __init__(self, llm: LLMClient, *, enabled: bool = True) -> None:
        self.llm = llm
        self.enabled = enabled

    def rewrite(self, question: str, history: Sequence[HistoryTurn]) -> str:
        if not self.enabled or not history or not needs_rewrite(question):
            return question
        try:
            raw = self.llm.chat(build_rewrite_messages(question, history), temperature=0.0)
        except LLMUnavailableError:
            raise
        except Exception as exc:  # never fail a request because rewriting failed
            logger.warning("Query rewrite failed, using heuristic fallback: %s", exc)
            return self._fallback(question, history)
        rewritten = self._clean(raw)
        if not rewritten or len(rewritten) > 2 * len(question) + 300:
            return self._fallback(question, history)
        return rewritten

    @staticmethod
    def _clean(raw: str) -> str:
        text = raw.strip().split("\n")[0].strip()
        text = re.sub(
            r"^(standalone question|rewritten question|question)\s*:\s*", "", text, flags=re.I
        )
        return text.strip().strip('"').strip("'").strip()

    @staticmethod
    def _fallback(question: str, history: Sequence[HistoryTurn]) -> str:
        last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
        return f"{last_user} {question}".strip() if last_user else question
