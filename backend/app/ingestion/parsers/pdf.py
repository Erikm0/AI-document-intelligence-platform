"""PDF text extraction with page provenance (pypdf)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.exceptions import DocumentParsingError, EmptyDocumentError
from app.ingestion.cleaning import normalize_text, remove_repeated_headers_footers
from app.ingestion.models import Block, ParsedDocument
from app.ingestion.parsers.base import pick_title, segment_lines

logger = logging.getLogger(__name__)


class PdfParser:
    def parse(self, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:  # pragma: no cover - hard dependency
            raise DocumentParsingError("pypdf is not installed") from exc

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        raise DocumentParsingError("PDF is password protected")
                except DocumentParsingError:
                    raise
                except Exception as exc:
                    raise DocumentParsingError("PDF is password protected") from exc
            raw_pages: list[str] = []
            for page in reader.pages:
                try:
                    raw_pages.append(page.extract_text() or "")
                except Exception as exc:  # one broken page must not sink the document
                    logger.warning("Failed to extract a PDF page: %s", exc)
                    raw_pages.append("")
        except DocumentParsingError:
            raise
        except PdfReadError as exc:
            raise DocumentParsingError(f"Invalid or corrupted PDF: {exc}") from exc
        except Exception as exc:
            raise DocumentParsingError(f"Could not read PDF: {exc}") from exc

        pages = remove_repeated_headers_footers([normalize_text(p) for p in raw_pages])
        blocks: list[Block] = []
        for page_number, page_text in enumerate(pages, start=1):
            blocks.extend(segment_lines(page_text, page=page_number))

        if not any(b.text.strip() for b in blocks):
            raise EmptyDocumentError(
                "No extractable text found in the PDF. It may be a scanned image - "
                "OCR is not enabled."
            )

        metadata = self._metadata(reader)
        empty_pages = [i for i, p in enumerate(pages, start=1) if not p.strip()]
        if empty_pages:
            metadata["empty_pages"] = empty_pages[:100]
        return ParsedDocument(
            blocks=blocks,
            page_count=len(pages),
            title=pick_title(metadata.get("title"), blocks, path.stem),
            metadata=metadata,
        )

    @staticmethod
    def _metadata(reader: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"page_numbering": "native"}
        try:
            meta = reader.metadata
        except Exception:
            meta = None
        if not meta:
            return out
        for key, attr in (
            ("title", "title"),
            ("author", "author"),
            ("subject", "subject"),
            ("creator", "creator"),
            ("producer", "producer"),
        ):
            try:
                value = getattr(meta, attr, None)
            except Exception:
                value = None
            if value:
                out[key] = str(value).strip()
        try:
            created = meta.creation_date
            if created:
                out["created"] = created.isoformat()
        except Exception:
            pass
        return out
