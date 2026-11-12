"""Text normalisation and boilerplate removal."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f​-‍﻿]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")
_MULTI_SPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*\d{1,4}\s*(?:of\s+\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"\d+")

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‐": "-",
        "‑": "-",
        "−": "-",
        "­": "",  # soft hyphen
    }
)


def normalize_text(text: str) -> str:
    """Unicode-normalise, de-hyphenate line breaks and tidy whitespace.

    Keeps paragraph structure (blank lines) intact; that is needed downstream to
    recover paragraphs.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)  # also expands ligatures like "fi"
    text = text.translate(_QUOTES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(line))


def _signature(line: str) -> str:
    """Normalise digits so 'Page 3' and 'Page 4' footers compare equal."""
    return _DIGITS_RE.sub("#", line.strip().lower())


def remove_repeated_headers_footers(
    pages: list[str], *, edge_lines: int = 3, min_ratio: float = 0.5
) -> list[str]:
    """Strip running headers/footers and bare page numbers from per-page text.

    * A line is boilerplate if (after digit normalisation) it appears among the
      first/last ``edge_lines`` lines of at least ``min_ratio`` of the pages
      (only for documents with 3+ pages, to avoid false positives).
    * A bare page number ("7", "Page 7 of 9", "- 7 -") is removed only when it is
      the first line or one of the last two lines of a page; a lone number in the
      middle of a page is probably table content.
    """

    def window(n_lines: int) -> int:
        # Never let the "edge" of a short page cover more than half of it.
        return min(edge_lines, max(1, n_lines // 2))

    boilerplate: set[str] = set()
    if len(pages) >= 3:
        counts: Counter[str] = Counter()
        for page in pages:
            lines = [ln for ln in page.split("\n") if ln.strip()]
            w = window(len(lines))
            edges = set(lines[:w] + lines[-w:])
            counts.update({_signature(ln) for ln in edges})
        threshold = max(2, int(len(pages) * min_ratio + 0.5))
        boilerplate = {sig for sig, n in counts.items() if n >= threshold and len(sig) < 120}

    cleaned: list[str] = []
    for page in pages:
        lines = page.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        w = window(len(non_empty))
        edge_idx = set(non_empty[:w] + non_empty[-w:])
        number_idx = set(non_empty[:1] + non_empty[-2:])
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (
                (i in edge_idx and _signature(ln) in boilerplate)
                or (i in number_idx and is_page_number_line(ln))
            )
        ]
        cleaned.append("\n".join(kept))
    return cleaned
