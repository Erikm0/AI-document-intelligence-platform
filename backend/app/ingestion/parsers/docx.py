"""DOCX extraction (python-docx) preserving reading order, headings, tables and,
where Word recorded them, page boundaries."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from app.core.exceptions import DocumentParsingError, EmptyDocumentError
from app.ingestion.cleaning import normalize_text
from app.ingestion.models import Block, BlockKind, ParsedDocument
from app.ingestion.parsers.base import heading_level, pick_title

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _q(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


_T, _TAB, _BR, _CR = _q("t"), _q("tab"), _q("br"), _q("cr")
_RENDERED_BREAK = _q("lastRenderedPageBreak")
_PAGE_BREAK_BEFORE = _q("pageBreakBefore")
_TYPE = _q("type")
_P, _TBL, _TR, _TC = _q("p"), _q("tbl"), _q("tr"), _q("tc")
_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


class _PageTracker:
    """Tracks the current page while walking the document body.

    Word writes ``<w:lastRenderedPageBreak/>`` wherever a page ended when it last
    laid the file out. If those markers are present they are authoritative (they
    already include explicit breaks); otherwise we fall back to explicit
    ``<w:br w:type="page"/>`` / ``pageBreakBefore`` breaks.
    """

    def __init__(self, use_rendered: bool) -> None:
        self.use_rendered = use_rendered
        self.page = 1

    def is_break(self, el: Any) -> bool:
        if self.use_rendered:
            return el.tag == _RENDERED_BREAK
        return el.tag == _BR and el.get(_TYPE) == "page"


class DocxParser:
    def parse(self, path: Path) -> ParsedDocument:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - hard dependency
            raise DocumentParsingError("python-docx is not installed") from exc

        if not zipfile.is_zipfile(path):
            raise DocumentParsingError("File is not a valid DOCX (Office Open XML) archive")
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise DocumentParsingError(f"Could not open DOCX: {exc}") from exc

        body = document.element.body
        has_rendered = next(body.iter(_RENDERED_BREAK), None) is not None
        has_explicit = any(br.get(_TYPE) == "page" for br in body.iter(_BR)) or (
            next(body.iter(_PAGE_BREAK_BEFORE), None) is not None
        )
        paginated = has_rendered or has_explicit
        tracker = _PageTracker(use_rendered=has_rendered)
        styles = self._style_names(document)

        blocks: list[Block] = []
        for child in body.iterchildren():
            if child.tag == _P:
                block = self._paragraph(child, tracker, styles)
                if block:
                    blocks.append(block)
            elif child.tag == _TBL:
                block = self._table(child, tracker)
                if block:
                    blocks.append(block)

        if not paginated:
            for block in blocks:
                block.page = None

        if not blocks:
            raise EmptyDocumentError("The DOCX file contains no text")

        metadata = self._metadata(document)
        metadata["page_numbering"] = (
            "rendered" if has_rendered else "explicit_breaks" if has_explicit else "unavailable"
        )
        return ParsedDocument(
            blocks=blocks,
            page_count=tracker.page if paginated else None,
            title=pick_title(metadata.get("title"), blocks, path.stem),
            metadata=metadata,
        )

    # ------------------------------------------------------------------------------
    @staticmethod
    def _style_names(document: Any) -> dict[str, str]:
        names: dict[str, str] = {}
        try:
            for style in document.styles:
                if style.style_id and style.name:
                    names[style.style_id] = style.name
        except Exception:
            pass
        return names

    @staticmethod
    def _walk_text(element: Any, tracker: _PageTracker) -> str:
        """Collect text in document order, advancing the page on breaks."""
        parts: list[str] = []
        pending_break = False
        for el in element.iter():
            if tracker.is_break(el):
                if "".join(parts).strip():
                    pending_break = True  # text already on this page -> break after
                else:
                    tracker.page += 1
            elif el.tag == _T and el.text:
                parts.append(el.text)
            elif el.tag == _TAB:
                parts.append("\t")
            elif el.tag in (_BR, _CR):
                parts.append("\n")
        if pending_break:
            tracker.page += 1
        return "".join(parts)

    def _paragraph(self, p: Any, tracker: _PageTracker, styles: dict[str, str]) -> Block | None:
        ppr = p.find(_q("pPr"))
        style_name = ""
        if ppr is not None:
            if not tracker.use_rendered and ppr.find(_PAGE_BREAK_BEFORE) is not None:
                tracker.page += 1
            pstyle = ppr.find(_q("pStyle"))
            if pstyle is not None:
                style_id = pstyle.get(_q("val"), "")
                style_name = styles.get(style_id, style_id)

        page = tracker.page
        text = normalize_text(self._walk_text(p, tracker)).replace("\n", " ")
        if not text:
            return None

        lowered = style_name.lower()
        if lowered == "title":
            return Block(text=text, page=page, kind=BlockKind.HEADING, heading_level=1)
        if m := _HEADING_STYLE_RE.match(lowered):
            return Block(
                text=text, page=page, kind=BlockKind.HEADING, heading_level=int(m.group(1))
            )
        if "list" in lowered or (ppr is not None and ppr.find(_q("numPr")) is not None):
            return Block(text=text, page=page, kind=BlockKind.LIST_ITEM)
        # Documents that fake headings with bold/caps instead of styles.
        if len(text) < 100 and (level := heading_level(text)) is not None:
            return Block(text=text, page=page, kind=BlockKind.HEADING, heading_level=level)
        return Block(text=text, page=page, kind=BlockKind.PARAGRAPH)

    def _table(self, tbl: Any, tracker: _PageTracker) -> Block | None:
        page = tracker.page
        rows: list[str] = []
        for tr in tbl.iter(_TR):
            cells: list[str] = []
            for tc in tr.findall(_TC):
                cell_text = " ".join(
                    normalize_text(self._walk_text(p, tracker)).replace("\n", " ")
                    for p in tc.findall(_P)
                ).strip()
                # Horizontally merged cells repeat their content - keep one copy.
                if cell_text and (not cells or cells[-1] != cell_text):
                    cells.append(cell_text)
            if cells:
                rows.append(" | ".join(cells))
        if not rows:
            return None
        return Block(text="\n".join(rows), page=page, kind=BlockKind.TABLE)

    @staticmethod
    def _metadata(document: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            props = document.core_properties
        except Exception:
            return out
        for key in ("title", "author", "subject", "keywords", "category"):
            value = getattr(props, key, None)
            if value:
                out[key] = str(value).strip()
        for key in ("created", "modified"):
            value = getattr(props, key, None)
            if value:
                out[key] = value.isoformat()
        return out
