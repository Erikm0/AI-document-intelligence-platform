"""Document use-cases: upload, list, inspect, delete, reprocess."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path, PurePath
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    UnsupportedFileTypeError,
)
from app.db.models import Document, DocumentStatus
from app.ingestion.parsers import CONTENT_TYPES, EXTENSION_TO_TYPE
from app.rag.lexical import BM25Index
from app.rag.vector_store import VectorStore
from app.repositories.documents import DocumentRepository
from app.services.storage import FileStorage

logger = logging.getLogger(__name__)

_UNSAFE_CHARS = re.compile(r"[^\w.\- ()\[\]]+", re.UNICODE)


def safe_filename(name: str | None) -> str:
    base = PurePath((name or "").replace("\\", "/")).name.strip()
    base = _UNSAFE_CHARS.sub("_", base)[:255]
    return base or "document"


class DocumentService:
    def __init__(
        self,
        session: Session,
        *,
        settings: Settings,
        storage: FileStorage,
        vector_store: VectorStore,
        lexical_index: BM25Index,
    ) -> None:
        self.session = session
        self.repo = DocumentRepository(session)
        self.settings = settings
        self.storage = storage
        self.vector_store = vector_store
        self.lexical_index = lexical_index

    def get(self, document_id: uuid.UUID) -> Document:
        document = self.repo.get(document_id)
        if document is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found", details={"document_id": str(document_id)}
            )
        return document

    def create_from_upload(self, filename: str | None, stream: BinaryIO) -> Document:
        name = safe_filename(filename)
        extension = Path(name).suffix.lower()
        if extension not in self.settings.allowed_extensions or extension not in EXTENSION_TO_TYPE:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension or 'none'}'. "
                f"Allowed: {', '.join(self.settings.allowed_extensions)}",
                details={"allowed": self.settings.allowed_extensions},
            )
        file_type = EXTENSION_TO_TYPE[extension]

        upload = self.storage.receive(stream, file_type)
        existing = self.repo.get_by_sha256(upload.sha256)
        if existing is not None:
            self.storage.discard(upload)
            raise DuplicateDocumentError(
                f"This file was already uploaded as '{existing.filename}'",
                details={"document_id": str(existing.id), "status": existing.status.value},
            )

        document_id = uuid.uuid4()
        path = self.storage.commit(upload, document_id, extension)
        document = Document(
            id=document_id,
            filename=name,
            file_type=file_type,
            content_type=CONTENT_TYPES[file_type],
            size_bytes=upload.size_bytes,
            sha256=upload.sha256,
            storage_path=str(path),
            status=DocumentStatus.PENDING,
            metadata_={},
        )
        try:
            self.repo.add(document)
        except Exception:
            self.storage.delete(path)
            raise
        logger.info("Accepted upload %s (%d bytes) as %s", name, upload.size_bytes, document_id)
        return document

    def delete(self, document_id: uuid.UUID) -> None:
        document = self.get(document_id)
        self.vector_store.delete_document(str(document_id))
        self.lexical_index.remove_document(str(document_id))
        storage_path = document.storage_path
        self.repo.delete(document)
        self.session.flush()
        self.storage.delete(storage_path)
        logger.info("Deleted document %s", document_id)

    def mark_for_reprocessing(self, document_id: uuid.UUID) -> Document:
        document = self.get(document_id)
        if document.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
            return document
        if not Path(document.storage_path).exists():
            raise DocumentNotFoundError("The original file is no longer available on disk")
        document.status = DocumentStatus.PENDING
        document.error_message = None
        self.session.flush()
        return document
