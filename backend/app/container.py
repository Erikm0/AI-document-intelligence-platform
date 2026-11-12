"""Composition root: builds and wires every long-lived component from settings.

Tests build a container with in-process fakes (hashing embedder, in-memory
vector store, lexical reranker, fake LLM) through the same code path.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select

from app.core.config import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
    VectorStoreProvider,
)
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import Database
from app.ingestion.chunking import Chunker
from app.ingestion.pipeline import IngestionPipeline, IngestionQueue
from app.rag.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from app.rag.lexical import BM25Index
from app.rag.llm import FakeLLM, LLMClient, OllamaClient
from app.rag.pipeline import PipelineConfig, RAGPipeline
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import CrossEncoderReranker, LexicalReranker, Reranker
from app.rag.retriever import HybridRetriever, RetrieverConfig
from app.rag.types import ChunkInfo
from app.rag.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore
from app.repositories.documents import DocumentRepository
from app.services.storage import FileStorage

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    storage: FileStorage
    embedder: Embedder
    vector_store: VectorStore
    lexical_index: BM25Index
    reranker: Reranker
    llm: LLMClient
    retriever: HybridRetriever
    pipeline: RAGPipeline
    ingestion: IngestionPipeline
    queue: IngestionQueue

    # ------------------------------------------------------------------------------
    def startup(self) -> None:
        if self.settings.auto_create_schema:
            self.db.create_schema()
        self._recover_interrupted_jobs()
        self.rebuild_lexical_index()
        if self.settings.warmup_models:
            self.warmup()

    def warmup(self) -> None:
        """Load model weights now so the first user request is not the slow one."""
        for component in (self.embedder, self.reranker):
            warm = getattr(component, "warmup", None)
            if callable(warm):
                try:
                    warm()
                except Exception as exc:  # surfaced by /health/ready instead
                    logger.warning("Warm-up of %s failed: %s", component.name, exc)

    def shutdown(self) -> None:
        self.queue.shutdown()
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()
        self.db.dispose()

    def rebuild_lexical_index(self) -> int:
        from app.ingestion.models import build_index_text

        self.lexical_index.clear()
        with self.db.session() as session:
            rows = session.execute(
                select(Chunk.id, Chunk.text, Chunk.section, Chunk.document_id, Document.title)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.status == DocumentStatus.READY)
            ).all()
        for chunk_id, text, section, document_id, title in rows:
            self.lexical_index.add(
                chunk_id, build_index_text(text, title=title, section=section), str(document_id)
            )
        logger.info("BM25 index rebuilt with %d chunks", len(rows))
        return len(rows)

    def _recover_interrupted_jobs(self) -> None:
        with self.db.session() as session:
            stuck = session.scalars(
                select(Document).where(
                    Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING])
                )
            ).all()
            for document in stuck:
                document.status = DocumentStatus.FAILED
                document.error_message = (
                    "Processing was interrupted by a server restart. Use 'Reprocess' to retry."
                )
            if stuck:
                logger.warning("Marked %d interrupted ingestion jobs as failed", len(stuck))

    def chunk_lookup(self, chunk_ids: list[str]) -> dict[str, ChunkInfo]:
        if not chunk_ids:
            return {}
        with self.db.session() as session:
            rows = session.execute(
                select(Chunk, Document.filename, Document.title)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(chunk_ids), Document.status == DocumentStatus.READY)
            ).all()
        return {
            chunk.id: ChunkInfo(
                chunk_id=chunk.id,
                document_id=str(chunk.document_id),
                text=chunk.text,
                filename=filename,
                title=title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
            )
            for chunk, filename, title in rows
        }

    def ready_document_ids(self) -> list[uuid.UUID]:
        with self.db.session() as session:
            return DocumentRepository(session).ready_ids()


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == EmbeddingProvider.HASHING:
        return HashingEmbedder(settings.hashing_embedding_dim)
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == VectorStoreProvider.MEMORY:
        return InMemoryVectorStore()
    if settings.vector_store == VectorStoreProvider.CHROMA_PERSISTENT:
        return ChromaVectorStore.persistent(settings.chroma_persist_dir, settings.chroma_collection)
    return ChromaVectorStore.http(
        settings.chroma_host, settings.chroma_port, settings.chroma_collection
    )


def build_reranker(settings: Settings, lexical_index: BM25Index) -> Reranker:
    if settings.reranker_provider == RerankerProvider.LEXICAL:
        return LexicalReranker(idf=lexical_index.idf)
    return CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == LLMProvider.FAKE:
        return FakeLLM()
    return OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        timeout=settings.llm_timeout_seconds,
        think=settings.ollama_think,
    )


def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    synchronous_ingestion: bool = False,
) -> Container:
    db = Database(settings.database_url, echo=settings.database_echo)
    storage = FileStorage(settings.upload_dir, settings.max_upload_size_bytes)
    lexical_index = BM25Index()
    embedder = embedder or build_embedder(settings)
    vector_store = vector_store or build_vector_store(settings)
    reranker = reranker or build_reranker(settings, lexical_index)
    llm = llm or build_llm(settings)

    container: Container | None = None

    def lookup(ids: list[str]) -> dict[str, ChunkInfo]:
        assert container is not None
        return container.chunk_lookup(ids)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        reranker=reranker,
        chunk_lookup=lookup,
        config=RetrieverConfig(
            candidates=settings.retrieval_candidates,
            rerank_candidates=settings.rerank_candidates,
            semantic_weight=settings.hybrid_semantic_weight,
            lexical_weight=settings.hybrid_lexical_weight,
            rrf_k=settings.rrf_k,
        ),
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        rewriter=QueryRewriter(llm, enabled=settings.query_rewrite_enabled),
        config=PipelineConfig(
            top_k=settings.retrieval_top_k,
            min_relevance=(
                settings.min_relevance_score
                if settings.min_relevance_score is not None
                else reranker.default_min_relevance
            ),
            min_semantic=(
                settings.min_semantic_score
                if settings.min_semantic_score is not None
                else embedder.default_min_similarity
            ),
            max_context_chars=settings.max_context_chars,
            grounding_high=settings.grounding_high_threshold,
            grounding_medium=settings.grounding_medium_threshold,
            strict_grounding=settings.strict_grounding,
        ),
    )
    ingestion = IngestionPipeline(
        db=db,
        chunker=Chunker(
            settings.chunk_size_tokens, settings.chunk_overlap_tokens, settings.min_chunk_tokens
        ),
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
    )
    container = Container(
        settings=settings,
        db=db,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        reranker=reranker,
        llm=llm,
        retriever=retriever,
        pipeline=pipeline,
        ingestion=ingestion,
        queue=IngestionQueue(ingestion, synchronous=synchronous_ingestion),
    )
    return container
