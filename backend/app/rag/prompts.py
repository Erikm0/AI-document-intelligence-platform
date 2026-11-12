"""Prompt templates and builders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.llm import ChatMessage
from app.rag.types import RetrievedChunk

REFUSAL_SENTINEL = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are DocIntel, a meticulous document analyst. You answer questions \
using ONLY the numbered sources provided in the user's message.

Rules:
1. Every factual statement must be supported by the sources and must end with a citation \
marker such as [1] or [2][3] that refers to the source numbers.
2. Do not use prior knowledge. Do not guess, speculate or fill gaps.
3. If the sources do not contain the information needed to answer, reply with exactly: \
{REFUSAL_SENTINEL}
4. If the sources only partially answer the question, answer the supported part and say \
clearly what the documents do not cover.
5. If sources disagree, point out the conflict and cite each side.
6. Be concise and precise. Prefer short paragraphs or bullet points. Quote figures, names \
and dates exactly as written in the sources.
7. Never cite a source number that does not exist and never invent quotes."""

QUERY_REWRITE_PROMPT = """Rewrite the follow-up question as a standalone question that can \
be understood without the conversation. Resolve pronouns and references ("it", "they", \
"that policy", "the second one") using the conversation. Keep names, numbers and technical \
terms unchanged. Do not answer the question. Output only the rewritten question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""

REFUSAL_MESSAGE = (
    "I couldn't find enough evidence in the selected documents to answer that question. "
    "Try rephrasing it, selecting different documents, or uploading a document that covers "
    "this topic."
)
NO_DOCUMENTS_MESSAGE = (
    "There are no processed documents to search yet. Upload a PDF, DOCX or TXT file "
    "and wait for it to finish processing, then ask again."
)


@dataclass(slots=True)
class HistoryTurn:
    role: str  # "user" | "assistant"
    content: str


def format_source(index: int, chunk: RetrievedChunk) -> str:
    # The document title tells the model which document a chunk belongs to
    # (a "Budget" section alone does not say *whose* budget it is).
    document = chunk.filename
    if chunk.title and chunk.title != chunk.filename:
        document = f'{chunk.filename} ("{chunk.title}")'
    location = " | ".join(
        p
        for p in (document, chunk.page_label(), chunk.section and f"Section: {chunk.section}")
        if p
    )
    return f"[{index}] {location}\n{chunk.text.strip()}"


def select_context(chunks: Sequence[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """Keep the best-ranked chunks that fit the context budget (always >= 1)."""
    selected: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        size = len(chunk.text) + 120
        if selected and used + size > max_chars:
            break
        selected.append(chunk)
        used += size
    return selected


def build_answer_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[HistoryTurn] = (),
    standalone_query: str | None = None,
) -> list[ChatMessage]:
    messages = [ChatMessage("system", SYSTEM_PROMPT)]
    for turn in history:
        content = turn.content.strip()
        if len(content) > 1500:
            content = content[:1500] + " ..."
        messages.append(ChatMessage("user" if turn.role == "user" else "assistant", content))

    sources = "\n\n".join(format_source(i, c) for i, c in enumerate(chunks, start=1))
    interpreted = ""
    if standalone_query and standalone_query.strip().lower() != question.strip().lower():
        interpreted = f"\n(Interpreted in context as: {standalone_query})"
    user = (
        f"Sources:\n{sources}\n\n"
        f"Question: {question}{interpreted}\n\n"
        f"Answer using only the sources above and cite them with [n] markers. "
        f"If they do not contain the answer, reply exactly {REFUSAL_SENTINEL}."
    )
    messages.append(ChatMessage("user", user))
    return messages


def build_rewrite_messages(question: str, history: Sequence[HistoryTurn]) -> list[ChatMessage]:
    lines = []
    for turn in history:
        content = turn.content.strip().replace("\n", " ")
        if len(content) > 600:
            content = content[:600] + " ..."
        lines.append(f"{'User' if turn.role == 'user' else 'Assistant'}: {content}")
    prompt = QUERY_REWRITE_PROMPT.format(history="\n".join(lines), question=question)
    return [ChatMessage("user", prompt)]
