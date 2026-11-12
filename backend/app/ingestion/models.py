"""Intermediate representations produced by parsers and the chunker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"


@dataclass(slots=True)
class Block:
    """A logical unit of a document (heading, paragraph, table...) with provenance."""

    text: str
    page: int | None = None
    kind: BlockKind = BlockKind.PARAGRAPH
    heading_level: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[Block]
    page_count: int | None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(slots=True)
class ChunkData:
    index: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None
    token_count: int
    char_start: int
    char_end: int


def build_index_text(text: str, *, title: str | None, section: str | None) -> str:
    """Text that is embedded / BM25-indexed for a chunk.

    Prepending the document title and section ("contextual chunk headers") gives
    short or pronoun-heavy chunks enough context to be retrievable on their own.
    """
    header = " > ".join(p for p in (title, section) if p)
    return f"{header}\n{text}" if header else text
