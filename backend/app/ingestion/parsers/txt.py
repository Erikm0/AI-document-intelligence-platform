"""Plain-text extraction with encoding detection.

Pages are delimited by form-feed characters (``\\f``) when present - that is how
``pdftotext`` and many legacy exports mark page breaks. Otherwise the whole file
is treated as a single page.
"""

from __future__ import annotations

import codecs
from pathlib import Path

from app.core.exceptions import DocumentParsingError, EmptyDocumentError
from app.ingestion.cleaning import normalize_text
from app.ingestion.models import Block, ParsedDocument
from app.ingestion.parsers.base import pick_title, segment_lines


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode bytes, returning ``(text, encoding)``."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


class TxtParser:
    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_bytes()
        text, encoding = decode_text(raw)
        sample = text[:4096]
        if "\x00" in sample:
            raise DocumentParsingError("File appears to be binary, not plain text")

        raw_pages = text.split("\f")
        blocks: list[Block] = []
        for page_number, page_text in enumerate(raw_pages, start=1):
            blocks.extend(segment_lines(normalize_text(page_text), page=page_number))

        if not blocks:
            raise EmptyDocumentError("The text file is empty")

        return ParsedDocument(
            blocks=blocks,
            page_count=len(raw_pages),
            title=pick_title(None, blocks, path.stem),
            metadata={
                "encoding": encoding,
                "line_count": text.count("\n") + 1,
                "page_numbering": "form_feed" if len(raw_pages) > 1 else "single_page",
            },
        )
