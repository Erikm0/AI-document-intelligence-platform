from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import DocumentParsingError, EmptyDocumentError
from app.ingestion.models import BlockKind
from app.ingestion.parsers.base import heading_level, segment_lines
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.pdf import PdfParser
from app.ingestion.parsers.txt import TxtParser, decode_text
from tests.factories import SAMPLE_DOCX_PAGES, SAMPLE_TXT, make_docx, make_pdf


class TestLayoutHeuristics:
    @pytest.mark.parametrize(
        ("line", "level"),
        [
            ("# Intro", 1),
            ("### Deep", 3),
            ("2.1 Data Retention", 2),
            ("3 Results", 1),
            ("EXECUTIVE SUMMARY", 1),
        ],
    )
    def test_heading_detection(self, line: str, level: int) -> None:
        assert heading_level(line) == level

    @pytest.mark.parametrize(
        "line", ["This is a normal sentence.", "3 apples were eaten by the team.", "NASA", ""]
    )
    def test_not_headings(self, line: str) -> None:
        assert heading_level(line) is None

    def test_segment_joins_wrapped_lines_and_splits_paragraphs(self) -> None:
        text = (
            "Introduction\n"
            "The quick brown fox jumps over the lazy dog and keeps running far\n"
            "into the forest where nobody can find it again.\n"
            "Short end.\n"
            "A new paragraph starts here with plenty of words in this line.\n"
            "- bullet item one\n"
            "- bullet item two"
        )
        blocks = segment_lines(text, page=3)
        kinds = [b.kind for b in blocks]
        assert kinds[0] == BlockKind.HEADING and blocks[0].text == "Introduction"
        assert "keeps running far into the forest" in blocks[1].text
        assert kinds.count(BlockKind.LIST_ITEM) == 2
        assert all(b.page == 3 for b in blocks)

    def test_setext_heading(self) -> None:
        blocks = segment_lines("My Title\n========\n\nBody text here.", page=1)
        assert blocks[0].kind == BlockKind.HEADING and blocks[0].text == "My Title"


class TestTxtParser:
    def test_pages_from_form_feeds(self, tmp_path: Path) -> None:
        path = tmp_path / "report.txt"
        path.write_text(SAMPLE_TXT, encoding="utf-8")
        parsed = TxtParser().parse(path)
        assert parsed.page_count == 2
        assert parsed.title == "Orion Space Agency Annual Report"
        results = next(b for b in parsed.blocks if "Shackleton" in b.text)
        assert results.page == 2
        headings = [b.text for b in parsed.blocks if b.kind == BlockKind.HEADING]
        assert "2. Budget" in headings and "3. Results" in headings

    @pytest.mark.parametrize(
        ("raw", "encoding"),
        [
            ("café".encode(), "utf-8"),
            ("café".encode("cp1252"), "cp1252"),
            ("café".encode("utf-16"), "utf-16"),
        ],
    )
    def test_encoding_detection(self, raw: bytes, encoding: str) -> None:
        text, detected = decode_text(raw)
        assert text == "café" and detected == encoding

    def test_binary_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_bytes(b"abc\x00\x01\x02" * 10)
        with pytest.raises(DocumentParsingError):
            TxtParser().parse(path)

    def test_empty_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(EmptyDocumentError):
            TxtParser().parse(path)


class TestDocxParser:
    def test_headings_pages_lists_tables_and_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.docx"
        path.write_bytes(
            make_docx(
                SAMPLE_DOCX_PAGES,
                title="Retention Handbook",
                table=[["Data", "Period"], ["Logs", "90 days"]],
            )
        )
        parsed = DocxParser().parse(path)
        assert parsed.title == "Retention Handbook"
        assert parsed.page_count == 2
        assert parsed.metadata["page_numbering"] == "explicit_breaks"

        by_text = {b.text: b for b in parsed.blocks}
        assert by_text["Data Retention Policy"].kind == BlockKind.HEADING
        assert by_text["Backups"].page == 2
        assert (
            by_text["Daily incremental backups are kept for 35 days."].kind == BlockKind.LIST_ITEM
        )
        table = next(b for b in parsed.blocks if b.kind == BlockKind.TABLE)
        assert "Logs | 90 days" in table.text

    def test_without_page_breaks_pages_are_unknown(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.docx"
        path.write_bytes(make_docx([[("h1", "Title"), ("p", "Only one page of text here.")]]))
        parsed = DocxParser().parse(path)
        assert parsed.page_count is None
        assert all(b.page is None for b in parsed.blocks)

    def test_invalid_archive(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.docx"
        path.write_bytes(b"PK\x03\x04 this is not really a zip")
        with pytest.raises(DocumentParsingError):
            DocxParser().parse(path)


class TestPdfParser:
    def test_pages_footer_removal_and_metadata(self, tmp_path: Path) -> None:
        pages = [
            ["# Overview", "The Vega-9 orbiter maps lunar water ice."],
            ["# Budget", "The budget was 312 million credits."],
            ["# Results", "Ice was found in 14 craters."],
        ]
        path = tmp_path / "orion.pdf"
        path.write_bytes(make_pdf(pages, title="Orion Report", footer="Orion Agency - Internal"))
        parsed = PdfParser().parse(path)
        assert parsed.page_count == 3
        assert parsed.title == "Orion Report"
        text = parsed.text
        assert "Internal" not in text  # running footer removed
        assert "Page 2" not in text
        budget = next(b for b in parsed.blocks if "312 million" in b.text)
        assert budget.page == 2

    def test_scanned_pdf_without_text(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(make_pdf([[], []]))
        with pytest.raises(EmptyDocumentError, match="OCR"):
            PdfParser().parse(path)

    def test_corrupted_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4\n garbage garbage")
        with pytest.raises(DocumentParsingError):
            PdfParser().parse(path)
