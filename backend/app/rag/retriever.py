"""Hybrid retrieval: dense (Chroma) + sparse (BM25) -> RRF fusion -> rerank."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.rag.embeddings import Embedder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.lexical import BM25Index
from app.rag.reranker import Reranker
from app.rag.types import ChunkInfo, RetrievalResult, RetrievedChunk
from app.rag.vector_store import VectorStore
from app.schemas.chat import RetrievalMode

logger = logging.getLogger(__name__)

ChunkLookup = Callable[[list[str]], dict[str, ChunkInfo]]


@dataclass(slots=True)
class RetrieverConfig:
    candidates: int = 30
    rerank_candidates: int = 20
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0
    rrf_k: int = 60


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_index: BM25Index,
        reranker: Reranker,
        chunk_lookup: ChunkLookup,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.chunk_lookup = chunk_lookup
        self.config = config or RetrieverConfig()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        rerank: bool = True,
    ) -> RetrievalResult:
        cfg = self.config
        timings: dict[str, float] = {}

        semantic_ranking: list[str] = []
        semantic_scores: dict[str, float] = {}
        if mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
            t0 = time.perf_counter()
            query_vector = self.embedder.embed_query(query)
            timings["embed_ms"] = _ms(t0)
            t0 = time.perf_counter()
            for hit in self.vector_store.query(query_vector, cfg.candidates, document_ids):
                semantic_ranking.append(hit.id)
                semantic_scores[hit.id] = hit.score
            timings["semantic_ms"] = _ms(t0)

        lexical_ranking: list[str] = []
        lexical_scores: dict[str, float] = {}
        if mode in (RetrievalMode.LEXICAL, RetrievalMode.HYBRID):
            t0 = time.perf_counter()
            for lhit in self.lexical_index.search(query, cfg.candidates, document_ids):
                lexical_ranking.append(lhit.id)
                lexical_scores[lhit.id] = lhit.score
            timings["lexical_ms"] = _ms(t0)

        rankings, weights = [], []
        if semantic_ranking:
            rankings.append(semantic_ranking)
            weights.append(cfg.semantic_weight)
        if lexical_ranking:
            rankings.append(lexical_ranking)
            weights.append(cfg.lexical_weight)
        fused = reciprocal_rank_fusion(rankings, weights=weights, k=cfg.rrf_k) if rankings else []

        pool_size = cfg.rerank_candidates if rerank else top_k
        pool_ids = [cid for cid, _ in fused[: max(pool_size, top_k)]]
        t0 = time.perf_counter()
        infos = self.chunk_lookup(pool_ids)
        timings["lookup_ms"] = _ms(t0)

        sem_rank = {cid: i for i, cid in enumerate(semantic_ranking, start=1)}
        lex_rank = {cid: i for i, cid in enumerate(lexical_ranking, start=1)}
        fused_score = dict(fused)
        allowed = set(document_ids) if document_ids else None
        chunks: list[RetrievedChunk] = []
        for cid in pool_ids:
            info = infos.get(cid)
            if info is None:  # stale vector / document not ready - skip
                continue
            if allowed is not None and info.document_id not in allowed:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=info.document_id,
                    text=info.text,
                    filename=info.filename,
                    title=info.title,
                    page_start=info.page_start,
                    page_end=info.page_end,
                    section=info.section,
                    semantic_score=semantic_scores.get(cid),
                    semantic_rank=sem_rank.get(cid),
                    lexical_score=lexical_scores.get(cid),
                    lexical_rank=lex_rank.get(cid),
                    fusion_score=fused_score.get(cid, 0.0),
                )
            )

        reranker_name = "none"
        if rerank and chunks:
            t0 = time.perf_counter()
            chunks = self.reranker.rerank(query, chunks)
            timings["rerank_ms"] = _ms(t0)
            reranker_name = self.reranker.name

        logger.debug(
            "retrieval query=%r semantic=%d lexical=%d fused=%d returned=%d",
            query,
            len(semantic_ranking),
            len(lexical_ranking),
            len(fused),
            min(top_k, len(chunks)),
        )
        return RetrievalResult(
            query=query,
            chunks=chunks[:top_k],
            candidates=len(fused),
            reranker=reranker_name,
            timings_ms=timings,
        )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
