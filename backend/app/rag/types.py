from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ChunkInfo:
    """Chunk payload resolved from the database (system of record)."""

    chunk_id: str
    document_id: str
    text: str
    filename: str
    title: str | None
    page_start: int | None
    page_end: int | None
    section: str | None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    filename: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    fusion_score: float = 0.0
    relevance: float | None = None  # normalised reranker score in [0, 1]

    @property
    def score(self) -> float:
        return self.relevance if self.relevance is not None else self.fusion_score

    def page_label(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end is None or self.page_end == self.page_start:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    candidates: int
    reranker: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_relevance(self) -> float:
        return max((c.relevance or 0.0 for c in self.chunks), default=0.0)
