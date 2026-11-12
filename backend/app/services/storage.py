"""Upload storage with streaming size enforcement, hashing and content sniffing."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.db.models import FileType

_CHUNK = 1024 * 1024


@dataclass(slots=True)
class StoredUpload:
    temp_path: Path
    size_bytes: int
    sha256: str


def sniff_matches(file_type: FileType, head: bytes) -> bool:
    """Cheap magic-byte check so a renamed binary can't masquerade as a document."""
    if file_type == FileType.PDF:
        return b"%PDF-" in head[:1024]
    if file_type == FileType.DOCX:
        return head.startswith(b"PK\x03\x04")
    if file_type == FileType.TXT:
        if head.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 BOM
            return True
        return b"\x00" not in head[:4096]
    return False


class FileStorage:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def receive(self, stream: BinaryIO, file_type: FileType) -> StoredUpload:
        """Copy an upload stream to a temp file, enforcing limits as we go."""
        digest = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(dir=self.root, prefix=".upload-", suffix=".part")
        tmp_path = Path(tmp_name)
        head = b""
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    block = stream.read(_CHUNK)
                    if not block:
                        break
                    if len(head) < 4096:
                        head += block[: 4096 - len(head)]
                    size += len(block)
                    if size > self.max_bytes:
                        raise FileTooLargeError(
                            f"File exceeds the {self.max_bytes // (1024 * 1024)} MB upload limit",
                            details={"max_bytes": self.max_bytes},
                        )
                    digest.update(block)
                    out.write(block)
            if size == 0:
                raise UnsupportedFileTypeError("The uploaded file is empty")
            if not sniff_matches(file_type, head):
                raise UnsupportedFileTypeError(
                    f"File content does not look like a valid {file_type.value.upper()} file"
                )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return StoredUpload(temp_path=tmp_path, size_bytes=size, sha256=digest.hexdigest())

    def commit(self, upload: StoredUpload, document_id: uuid.UUID, extension: str) -> Path:
        final = self.root / f"{document_id}{extension}"
        shutil.move(str(upload.temp_path), final)
        return final

    def discard(self, upload: StoredUpload) -> None:
        upload.temp_path.unlink(missing_ok=True)

    @staticmethod
    def delete(path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)
