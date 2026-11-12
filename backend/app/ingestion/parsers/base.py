"""Parser protocol and shared layout heuristics for line-oriented text (PDF, TXT)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from app.ingestion.models import Block, BlockKind, ParsedDocument


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^.!?]{1,90})$")
_SETEXT_RE = re.compile(r"^(={3,}|-{3,})$")
_BULLET_RE = re.compile(r"^(?:[•●◦▪‣⁃*\-]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", '."', '?"', ".)")
_MINOR_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}


def heading_level(line: str) -> int | None:
    """Return a heading level if ``line`` looks like a heading, else None."""
    line = line.strip()
    if not line or len(line) > 100:
        return None
    if m := _MD_HEADING_RE.match(line):
        return len(m.group(1))
    m = _NUMBERED_HEADING_RE.match(line)
    if m and len(line.split()) <= 12 and not line.endswith(_TERMINAL_PUNCT):
        return m.group(1).count(".") + 1
    letters = [c for c in line if c.isalpha()]
    if (
        len(letters) >= 5
        and all(c.isupper() for c in letters)
        and len(line.split()) <= 10
        and not line.endswith((".", ",", ";"))
    ):
        return 1
    return None


def looks_like_title_case_heading(line: str) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 8) or line.endswith((*_TERMINAL_PUNCT, ",")):
        return False
    if not line[0].isupper():
        return False
    significant = [w for w in words if w.lower() not in _MINOR_WORDS]
    if not significant:
        return False
    capitalised = sum(1 for w in significant if w[0].isupper() or w[0].isdigit())
    return capitalised / len(significant) >= 0.8


def clean_heading(line: str) -> str:
    line = line.strip()
    if m := _MD_HEADING_RE.match(line):
        return m.group(2).strip()
    return line


def segment_lines(text: str, page: int | None) -> list[Block]:
    """Group raw lines into heading / paragraph / list-item blocks.

    Works for both PDF page text (lines without blank separators) and
    hard-wrapped plain text (paragraphs separated by blank lines).
    """
    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return []
    max_len = max(len(ln) for ln in non_empty)

    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_kind = BlockKind.PARAGRAPH

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if buffer:
            joined = " ".join(s.strip() for s in buffer).strip()
            if joined:
                blocks.append(Block(text=joined, page=page, kind=buffer_kind))
        buffer = []
        buffer_kind = BlockKind.PARAGRAPH

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush()
            continue

        # Setext headings ("Title\n=====") - the underline promotes the buffered line.
        if _SETEXT_RE.match(line):
            if len(buffer) == 1:
                setext_level = 1 if line.startswith("=") else 2
                blocks.append(
                    Block(
                        text=buffer[0].strip(),
                        page=page,
                        kind=BlockKind.HEADING,
                        heading_level=setext_level,
                    )
                )
                buffer = []
            continue

        prev = buffer[-1] if buffer else None
        starts_new_para = prev is None or prev.rstrip().endswith(_TERMINAL_PUNCT)
        next_line = next((ln.strip() for ln in lines[i + 1 :] if ln.strip()), "")

        level = heading_level(line)
        if level is None and starts_new_para and looks_like_title_case_heading(line):
            is_short = len(line) < 0.7 * max_len or max_len < 40
            if is_short and next_line[:1].isupper():
                level = 2
        if level is not None:
            flush()
            blocks.append(
                Block(
                    text=clean_heading(line), page=page, kind=BlockKind.HEADING, heading_level=level
                )
            )
            continue

        if _BULLET_RE.match(line):
            flush()
            buffer = [line]
            buffer_kind = BlockKind.LIST_ITEM
            continue

        if prev is not None:
            prev_is_short = len(prev) < 0.75 * max_len
            if prev.rstrip().endswith(_TERMINAL_PUNCT) and prev_is_short:
                flush()
        buffer.append(line)

    flush()
    return blocks


def pick_title(candidate: str | None, blocks: list[Block], fallback: str) -> str:
    bad = ("untitled", "microsoft word", "document", "slide")
    if candidate:
        cand = candidate.strip()
        if len(cand) > 2 and not cand.lower().startswith(bad):
            return cand[:300]
    for block in blocks[:10]:
        if block.kind == BlockKind.HEADING and 2 < len(block.text) <= 200:
            return block.text
    return fallback
