from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from app.db.models import DocumentStatus, FileType
from app.schemas.common import APIModel


class DocumentRead(APIModel):
    id: uuid.UUID
    filename: str
    file_type: FileType
    content_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    error_message: str | None = None
    title: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None


class DocumentStats(APIModel):
    total_documents: int
    ready_documents: int
    processing_documents: int
    failed_documents: int
    total_chunks: int
    total_pages: int
    total_size_bytes: int


class ChunkRead(APIModel):
    id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    token_count: int
