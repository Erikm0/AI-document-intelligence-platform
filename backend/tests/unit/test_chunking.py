from __future__ import annotations

from itertools import pairwise

import pytest

from app.core.text import count_tokens
from app.ingestion.chunking import Chunker
from app.ingestion.models import Block, BlockKind, build_index_text


def _paragraph(n: int, page: int, words: int = 12) -> Block:
    sentence = " ".join(f"word{n}x{i}" for i in range(words)) + "."
    return Block(text=f"{sentence} {sentence}", page=page)


def test_empty_input() -> None:
    assert Chunker().chunk([]) == []


def test_rejects_overlap_ge_size() -> None:
    with pytest.raises(ValueError):
        Chunker(chunk_size=50, overlap=50)


def test_chunks_respect_size_and_are_contiguous() -> None:
    blocks = [_paragraph(i, page=1 + i // 3) for i in range(12)]
    chunks = Chunker(chunk_size=80, overlap=0, min_chunk_tokens=5).chunk(blocks)
    assert len(chunks) > 3
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.token_count <= 80 * 1.1
    joined = " ".join(c.text for c in chunks)
    for i in range(12):
        assert f"word{i}x0" in joined


def test_page_ranges_are_preserved() -> None:
    blocks = [
        Block(text="Alpha fact on page one. " * 3, page=1),
        Block(text="Beta fact on page two. " * 3, page=2),
    ]
    chunks = Chunker(chunk_size=200, overlap=0).chunk(blocks)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_overlap_repeats_trailing_sentences() -> None:
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(30)]
    blocks = [Block(text=" ".join(sentences), page=1)]
    chunks = Chunker(chunk_size=60, overlap=20, min_chunk_tokens=5).chunk(blocks)
    assert len(chunks) >= 3
    for prev, nxt in pairwise(chunks):
        last_sentence = prev.text.split(". ")[-1].rstrip(".")
        assert last_sentence in nxt.text  # the boundary sentence appears in both


def test_sections_are_tracked_and_headings_start_chunks() -> None:
    blocks = [
        Block(text="Remote Work", kind=BlockKind.HEADING, heading_level=1, page=1),
        *[
            Block(text=f"Remote rule {i} applies to all staff members here.", page=1)
            for i in range(6)
        ],
        Block(text="Leave", kind=BlockKind.HEADING, heading_level=1, page=2),
        Block(text="Employees get 26 days of annual leave every year.", page=2),
    ]
    chunks = Chunker(chunk_size=60, overlap=10, min_chunk_tokens=5).chunk(blocks)
    leave = next(c for c in chunks if "26 days" in c.text)
    assert leave.section == "Leave"
    assert leave.text.startswith("Leave")
    assert "Remote rule" not in leave.text  # no overlap across a section boundary
    assert chunks[0].section == "Remote Work"


def test_nested_heading_path() -> None:
    blocks = [
        Block(text="Policies", kind=BlockKind.HEADING, heading_level=1),
        Block(text="Security", kind=BlockKind.HEADING, heading_level=2),
        Block(text="All laptops must be encrypted at rest with strong keys."),
    ]
    chunks = Chunker(chunk_size=100, overlap=0, min_chunk_tokens=1).chunk(blocks)
    assert chunks[0].section == "Policies > Security"


def test_long_sentence_is_split() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(400)) + "."
    chunks = Chunker(chunk_size=50, overlap=0, min_chunk_tokens=1).chunk(
        [Block(text=long_sentence)]
    )
    assert len(chunks) >= 6
    assert all(count_tokens(c.text) <= 60 for c in chunks)


def test_tiny_tail_is_merged() -> None:
    blocks = [
        Block(text="A fairly long sentence about retrieval quality and ranking. " * 4),
        Block(text="Tail."),
    ]
    chunks = Chunker(chunk_size=60, overlap=0, min_chunk_tokens=10).chunk(blocks)
    assert chunks[-1].text.endswith("Tail.")
    assert chunks[-1].token_count > 5


def test_char_offsets_point_into_joined_text() -> None:
    blocks = [Block(text="First block text."), Block(text="Second block text here.")]
    full = "\n\n".join(b.text for b in blocks)
    chunk = Chunker(chunk_size=100, overlap=0, min_chunk_tokens=1).chunk(blocks)[0]
    assert full[chunk.char_start : chunk.char_end] == full


def test_build_index_text() -> None:
    assert build_index_text("body", title="Doc", section="Intro") == "Doc > Intro\nbody"
    assert build_index_text("body", title=None, section=None) == "body"
