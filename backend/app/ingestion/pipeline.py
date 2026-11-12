"""Document ingestion: parse -> clean -> chunk -> embed -> index -> persist."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app.core.exceptions import AppError, EmptyDocumentError
from app.db.models import Chunk, DocumentStatus, utcnow
from app.db.session import Database
from app.ingestion.chunking import Chunker
from app.ingestion.models import build_index_text
from app.ingestion.parsers import get_parser
from app.rag.embeddings import Embedder
from app.rag.lexical import BM25Index
from app.rag.vector_store import VectorRecord, VectorStore
from app.repositories.documents import DocumentRepository

logger = logging.getLogger(__name__)


def chunk_id_for(document_id: uuid.UUID | str, index: int) -> str:
    return f"{document_id}:{index}"


class IngestionPipeline:
    def __init__(
        self,
        *,
        db: Database,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_index: BM25Index,
    ) -> None:
        self.db = db
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.lexical_index = lexical_index

    def process(self, document_id: uuid.UUID) -> None:
        """Ingest one document. Never raises: failures are recorded on the row."""
        started = time.perf_counter()
        with self.db.session() as session:
            repo = DocumentRepository(session)
            document = repo.get(document_id)
            if document is None:
                logger.warning("Document %s vanished before processing", document_id)
                return
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            path = Path(document.storage_path)
            file_type = document.file_type
            filename = document.filename

        logger.info("Ingesting %s (%s)", filename, document_id)
        try:
            parsed = get_parser(file_type).parse(path)
            chunks = self.chunker.chunk(parsed.blocks)
            if not chunks:
                raise EmptyDocumentError("No text could be extracted from the document")

            title = parsed.title or filename
            index_texts = [build_index_text(c.text, title=title, section=c.section) for c in chunks]
            embeddings = self.embedder.embed_documents(index_texts)

            doc_id = str(document_id)
            self.vector_store.delete_document(doc_id)  # idempotent re-processing
            self.vector_store.upsert(
                [
                    VectorRecord(
                        id=chunk_id_for(document_id, c.index),
                        embedding=emb,
                        text=c.text,
                        metadata={
                            "document_id": doc_id,
                            "chunk_index": c.index,
                            "filename": filename,
                            "page_start": c.page_start if c.page_start is not None else -1,
                            "page_end": c.page_end if c.page_end is not None else -1,
                            "section": (c.section or "")[:500],
                        },
                    )
                    for c, emb in zip(chunks, embeddings, strict=True)
                ]
            )

            with self.db.session() as session:
                repo = DocumentRepository(session)
                document = repo.get(document_id)
                if document is None:  # deleted while we were working
                    self.vector_store.delete_document(doc_id)
                    return
                repo.replace_chunks(
                    document_id,
                    [
                        Chunk(
                            id=chunk_id_for(document_id, c.index),
                            document_id=document_id,
                            chunk_index=c.index,
                            text=c.text,
                            page_start=c.page_start,
                            page_end=c.page_end,
                            section=c.section[:512] if c.section else None,
                            token_count=c.token_count,
                            char_start=c.char_start,
                            char_end=c.char_end,
                        )
                        for c in chunks
                    ],
                )
                document.status = DocumentStatus.READY
                document.title = title[:512]
                document.page_count = parsed.page_count
                document.chunk_count = len(chunks)
                document.char_count = sum(len(b.text) for b in parsed.blocks)
                document.metadata_ = {
                    **parsed.metadata,
                    "processing_ms": round((time.perf_counter() - started) * 1000, 1),
                    "embedding_model": self.embedder.name,
                    "avg_chunk_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1),
                }
                document.processed_at = utcnow()

            self.lexical_index.remove_document(doc_id)
            self.lexical_index.add_many(
                (chunk_id_for(document_id, c.index), text, doc_id)
                for c, text in zip(chunks, index_texts, strict=True)
            )
            logger.info(
                "Ingested %s: %d chunks, %s pages in %.0f ms",
                filename,
                len(chunks),
                parsed.page_count,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            message = exc.message if isinstance(exc, AppError) else f"Unexpected error: {exc}"
            if not isinstance(exc, AppError):
                logger.exception("Ingestion of %s failed", document_id)
            else:
                logger.warning("Ingestion of %s failed: %s", document_id, message)
            self._mark_failed(document_id, message)

    def _mark_failed(self, document_id: uuid.UUID, message: str) -> None:
        try:
            self.vector_store.delete_document(str(document_id))
        except Exception:
            logger.warning("Could not clean up vectors for %s", document_id)
        self.lexical_index.remove_document(str(document_id))
        with self.db.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = message[:2000]
                document.chunk_count = 0


class IngestionQueue:
    """Bounded worker pool for ingestion jobs.

    Embedding is CPU/GPU bound, so we cap concurrency instead of spawning a
    thread per upload. ``synchronous=True`` runs jobs inline (tests).
    For multi-node deployments replace with Celery/RQ/Arq behind this interface.
    """

    def __init__(
        self, pipeline: IngestionPipeline, *, workers: int = 2, synchronous: bool = False
    ) -> None:
        self.pipeline = pipeline
        self.synchronous = synchronous
        self._executor = (
            None
            if synchronous
            else ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        )

    def submit(self, document_id: uuid.UUID) -> Future[None] | None:
        if self._executor is None:
            self.pipeline.process(document_id)
            return None
        return self._executor.submit(self.pipeline.process, document_id)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
