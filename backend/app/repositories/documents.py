"""Data-access layer for documents and chunks."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # --- documents -------------------------------------------------------------
    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.flush()
        return document

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_sha256(self, sha256: str) -> Document | None:
        return self.session.scalar(select(Document).where(Document.sha256 == sha256))

    def paginate(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: DocumentStatus | None = None,
        search: str | None = None,
    ) -> tuple[Sequence[Document], int]:
        stmt = select(Document)
        count_stmt = select(func.count()).select_from(Document)
        if status is not None:
            stmt = stmt.where(Document.status == status)
            count_stmt = count_stmt.where(Document.status == status)
        if search:
            pattern = f"%{search.lower()}%"
            stmt = stmt.where(func.lower(Document.filename).like(pattern))
            count_stmt = count_stmt.where(func.lower(Document.filename).like(pattern))
        total = self.session.scalar(count_stmt) or 0
        rows = self.session.scalars(
            stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return rows, total

    def get_many(self, ids: Iterable[uuid.UUID]) -> list[Document]:
        ids = list(ids)
        if not ids:
            return []
        return list(self.session.scalars(select(Document).where(Document.id.in_(ids))).all())

    def ready_ids(self) -> list[uuid.UUID]:
        return list(
            self.session.scalars(
                select(Document.id).where(Document.status == DocumentStatus.READY)
            ).all()
        )

    def delete(self, document: Document) -> None:
        self.session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        self.session.delete(document)
        self.session.flush()

    def stats(self) -> dict[str, int]:
        row = self.session.execute(
            select(
                func.count(Document.id),
                func.coalesce(func.sum(Document.chunk_count), 0),
                func.coalesce(func.sum(Document.page_count), 0),
                func.coalesce(func.sum(Document.size_bytes), 0),
            )
        ).one()
        by_status: dict[DocumentStatus, int] = {
            status: int(count or 0)
            for status, count in self.session.execute(
                select(Document.status, func.count(Document.id)).group_by(Document.status)
            ).all()
        }
        return {
            "total_documents": int(row[0] or 0),
            "total_chunks": int(row[1] or 0),
            "total_pages": int(row[2] or 0),
            "total_size_bytes": int(row[3] or 0),
            "ready_documents": by_status.get(DocumentStatus.READY, 0),
            "processing_documents": by_status.get(DocumentStatus.PROCESSING, 0)
            + by_status.get(DocumentStatus.PENDING, 0),
            "failed_documents": by_status.get(DocumentStatus.FAILED, 0),
        }

    # --- chunks ----------------------------------------------------------------
    def replace_chunks(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> None:
        self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.session.add_all(chunks)
        self.session.flush()

    def list_chunks(
        self, document_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[Sequence[Chunk], int]:
        total = (
            self.session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
            )
            or 0
        )
        rows = self.session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        ).all()
        return rows, total

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        rows = self.session.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
        return {c.id: c for c in rows}

    def iter_ready_chunks(self) -> Iterable[tuple[str, str, uuid.UUID]]:
        """(chunk_id, text, document_id) for every chunk of every READY document."""
        stmt = (
            select(Chunk.id, Chunk.text, Chunk.document_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY)
            .execution_options(yield_per=1000)
        )
        for row in self.session.execute(stmt):
            yield row[0], row[1], row[2]
