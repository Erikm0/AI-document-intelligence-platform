from __future__ import annotations

from app.db.models import FileType
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.pdf import PdfParser
from app.ingestion.parsers.txt import TxtParser

_PARSERS: dict[FileType, DocumentParser] = {
    FileType.PDF: PdfParser(),
    FileType.DOCX: DocxParser(),
    FileType.TXT: TxtParser(),
}

EXTENSION_TO_TYPE: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
}

CONTENT_TYPES: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileType.TXT: "text/plain",
}


def get_parser(file_type: FileType) -> DocumentParser:
    return _PARSERS[file_type]


__all__ = ["CONTENT_TYPES", "EXTENSION_TO_TYPE", "DocumentParser", "get_parser"]
