from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.db.models import FileType
from app.services.document_service import safe_filename
from app.services.storage import FileStorage, sniff_matches


class TestStorage:
    def test_receive_hashes_and_commits(self, tmp_path: Path) -> None:
        storage = FileStorage(tmp_path, max_bytes=1024)
        data = b"hello world"
        upload = storage.receive(io.BytesIO(data), FileType.TXT)
        assert upload.size_bytes == len(data)
        assert upload.sha256 == hashlib.sha256(data).hexdigest()
        import uuid

        final = storage.commit(upload, uuid.uuid4(), ".txt")
        assert final.read_bytes() == data
        assert not upload.temp_path.exists()

    def test_size_limit_enforced_while_streaming(self, tmp_path: Path) -> None:
        storage = FileStorage(tmp_path, max_bytes=10)
        with pytest.raises(FileTooLargeError):
            storage.receive(io.BytesIO(b"x" * 11), FileType.TXT)
        assert list(tmp_path.glob("*.part")) == []  # temp file cleaned up

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedFileTypeError, match="empty"):
            FileStorage(tmp_path, 100).receive(io.BytesIO(b""), FileType.TXT)

    def test_content_sniffing(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            FileStorage(tmp_path, 100).receive(io.BytesIO(b"plain text"), FileType.PDF)

    @pytest.mark.parametrize(
        ("file_type", "head", "ok"),
        [
            (FileType.PDF, b"%PDF-1.7 ...", True),
            (FileType.PDF, b"PK\x03\x04", False),
            (FileType.DOCX, b"PK\x03\x04...", True),
            (FileType.DOCX, b"%PDF-1.4", False),
            (FileType.TXT, b"plain text", True),
            (FileType.TXT, b"\x00\x01binary", False),
            (FileType.TXT, b"\xff\xfeh\x00i\x00", True),
        ],
    )
    def test_sniff(self, file_type: FileType, head: bytes, ok: bool) -> None:
        assert sniff_matches(file_type, head) is ok


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\Q3 report (final).docx", "Q3 report (final).docx"),
        ("weird<>|name?.txt", "weird_name_.txt"),
        ("", "document"),
        (None, "document"),
    ],
)
def test_safe_filename(raw: str | None, expected: str) -> None:
    assert safe_filename(raw) == expected


class TestSettings:
    def test_overlap_must_be_smaller_than_chunk(self) -> None:
        with pytest.raises(ValidationError):
            Settings(chunk_size_tokens=100, chunk_overlap_tokens=100)

    def test_extensions_are_normalised(self) -> None:
        assert Settings(allowed_extensions=["PDF", ".Txt"]).allowed_extensions == [".pdf", ".txt"]

    def test_rerank_candidates_capped(self) -> None:
        assert Settings(retrieval_candidates=10, rerank_candidates=50).rerank_candidates == 10
