"""Structure-aware, page-preserving chunking.

Strategy
--------
1. Every block (heading / paragraph / list item / table) is split into sentence
   *units* that remember their page, section heading and character offset.
   Oversized sentences are split on word windows; table rows are kept whole.
2. Units are packed greedily into chunks of ``chunk_size`` tokens. A new chunk is
   started early at a heading boundary when the current chunk is already at least
   half full, so chunks tend to align with document sections.
3. Consecutive chunks overlap by up to ``overlap`` tokens of whole sentences
   (never across a section boundary), so facts that straddle a boundary remain
   retrievable.
4. Each chunk records ``page_start``/``page_end`` and its section heading, which is
   what powers page-level citations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.text import count_tokens, split_sentences
from app.ingestion.models import Block, BlockKind, ChunkData


@dataclass(slots=True)
class _Unit:
    text: str
    page: int | None
    section: str | None
    tokens: int
    char_start: int
    para_start: bool  # first unit of a block -> joined with a blank line
    is_heading: bool


class Chunker:
    def __init__(
        self, chunk_size: int = 220, overlap: int = 40, min_chunk_tokens: int = 20
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------------------
    def chunk(self, blocks: list[Block]) -> list[ChunkData]:
        units = self._to_units(blocks)
        if not units:
            return []
        groups = self._pack(units)
        groups = self._merge_small_tail(groups)
        return [self._build(i, g) for i, g in enumerate(groups)]

    # ------------------------------------------------------------------------------
    def _to_units(self, blocks: list[Block]) -> list[_Unit]:
        units: list[_Unit] = []
        heading_path: dict[int, str] = {}
        offset = 0
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if block.kind == BlockKind.HEADING:
                level = block.heading_level or 1
                heading_path = {k: v for k, v in heading_path.items() if k < level}
                heading_path[level] = text
            section = " > ".join(heading_path[k] for k in sorted(heading_path)) or None

            if block.kind == BlockKind.HEADING:
                pieces = [text]
            elif block.kind == BlockKind.TABLE:
                pieces = self._split_long(text.split("\n"))
            else:
                pieces = self._split_long(split_sentences(text) or [text])

            cursor = 0
            for j, piece in enumerate(pieces):
                found = text.find(piece, cursor)
                local = found if found >= 0 else cursor
                cursor = local + len(piece)
                units.append(
                    _Unit(
                        text=piece,
                        page=block.page,
                        section=section,
                        tokens=max(1, count_tokens(piece)),
                        char_start=offset + local,
                        para_start=j == 0,
                        is_heading=block.kind == BlockKind.HEADING,
                    )
                )
            offset += len(text) + 2  # blocks are joined with "\n\n"
        return units

    def _split_long(self, pieces: list[str]) -> list[str]:
        """Break any piece longer than the chunk size into word windows."""
        out: list[str] = []
        limit_words = max(8, int(self.chunk_size / 1.3) - 2)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if count_tokens(piece) <= self.chunk_size:
                out.append(piece)
                continue
            words = piece.split()
            for i in range(0, len(words), limit_words):
                out.append(" ".join(words[i : i + limit_words]))
        return out

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        fresh = 0  # number of non-overlap units in `current`
        fresh_content = 0  # tokens of non-overlap, non-heading units

        for unit in units:
            # Start a new chunk at a heading once the current one has real content:
            # section-aligned chunks give far more precise citations. Consecutive
            # headings ("Policies" > "Security") stay together with their content.
            heading_break = unit.is_heading and fresh_content >= self.min_chunk_tokens
            overflow = current_tokens + unit.tokens > self.chunk_size and fresh_content > 0
            if heading_break or overflow:
                groups.append(current)
                current = [] if unit.is_heading else self._overlap_tail(current, unit)
                current_tokens = sum(u.tokens for u in current)
                fresh = fresh_content = 0
            current.append(unit)
            current_tokens += unit.tokens
            fresh += 1
            if not unit.is_heading:
                fresh_content += unit.tokens

        if current and fresh > 0:
            groups.append(current)
        return groups

    def _overlap_tail(self, group: list[_Unit], next_unit: _Unit) -> list[_Unit]:
        if self.overlap <= 0:
            return []
        tail: list[_Unit] = []
        total = 0
        for unit in reversed(group):
            if unit.is_heading or unit.section != next_unit.section:
                break
            if total + unit.tokens > self.overlap:
                break
            tail.insert(0, unit)
            total += unit.tokens
        # Never carry the entire previous chunk over.
        if len(tail) == len(group):
            tail = tail[1:]
        return tail

    def _merge_small_tail(self, groups: list[list[_Unit]]) -> list[list[_Unit]]:
        if len(groups) < 2:
            return groups
        last = groups[-1]
        last_tokens = sum(u.tokens for u in last)
        prev_tokens = sum(u.tokens for u in groups[-2])
        if (
            last_tokens < self.min_chunk_tokens
            and prev_tokens + last_tokens <= self.chunk_size * 1.5
        ):
            prev_ids = {id(u) for u in groups[-2]}
            groups[-2] = groups[-2] + [u for u in last if id(u) not in prev_ids]
            groups.pop()
        return groups

    @staticmethod
    def _build(index: int, units: list[_Unit]) -> ChunkData:
        parts: list[str] = []
        for i, unit in enumerate(units):
            if i == 0:
                parts.append(unit.text)
            elif unit.para_start or units[i - 1].is_heading:
                parts.append("\n\n" + unit.text)
            else:
                parts.append(" " + unit.text)
        text = "".join(parts).strip()
        pages = [u.page for u in units if u.page is not None]
        # Attribute the chunk to the section holding most of its content tokens.
        weights: dict[str | None, int] = {}
        for u in units:
            if not u.is_heading:
                weights[u.section] = weights.get(u.section, 0) + u.tokens
        section = max(weights, key=lambda s: weights[s]) if weights else units[0].section
        last = units[-1]
        return ChunkData(
            index=index,
            text=text,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section=section,
            token_count=count_tokens(text),
            char_start=units[0].char_start,
            char_end=last.char_start + len(last.text),
        )
