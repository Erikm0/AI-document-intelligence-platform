from __future__ import annotations

import math

import numpy as np
import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.lexical import BM25Index
from app.rag.reranker import CrossEncoderReranker, LexicalReranker
from app.rag.retriever import HybridRetriever, RetrieverConfig
from app.rag.types import ChunkInfo, RetrievedChunk
from app.rag.vector_store import InMemoryVectorStore, VectorRecord
from app.schemas.chat import RetrievalMode

DOCS = {
    "a:0": ("a", "Employees receive 26 days of paid annual leave per year."),
    "a:1": ("a", "Remote work is allowed for up to three days per week."),
    "b:0": ("b", "The reactor coolant outlet temperature is 520 degrees Celsius."),
    "b:1": ("b", "Clause 14.3 limits liability to twelve months of fees."),
}


class TestHashingEmbedder:
    def test_deterministic_and_normalised(self) -> None:
        emb = HashingEmbedder(256)
        v1, v2 = emb.embed_query("annual leave days"), emb.embed_query("annual leave days")
        assert v1 == v2
        assert math.isclose(float(np.linalg.norm(v1)), 1.0, rel_tol=1e-5)
        assert emb.dimension == 256

    def test_similar_texts_are_closer(self) -> None:
        emb = HashingEmbedder()
        q = np.array(emb.embed_query("paid annual leave"))
        near = np.array(emb.embed_documents([DOCS["a:0"][1]])[0])
        far = np.array(emb.embed_documents([DOCS["b:0"][1]])[0])
        assert q @ near > q @ far


class TestBM25:
    @pytest.fixture
    def index(self) -> BM25Index:
        index = BM25Index()
        index.add_many((cid, text, doc) for cid, (doc, text) in DOCS.items())
        return index

    def test_ranks_exact_terms_first(self, index: BM25Index) -> None:
        hits = index.search("clause 14.3 liability", top_k=3)
        assert hits[0].id == "b:1"

    def test_document_filter(self, index: BM25Index) -> None:
        hits = index.search("days", top_k=5, document_ids=["b"])
        assert all(h.document_id == "b" for h in hits)

    def test_remove_document(self, index: BM25Index) -> None:
        index.remove_document("a")
        assert len(index) == 2
        assert index.search("annual leave", top_k=5) == []

    def test_reindex_same_id_replaces(self, index: BM25Index) -> None:
        index.add("a:0", "completely different words", "a")
        assert len(index) == 4
        assert all(h.id != "a:0" for h in index.search("annual leave", top_k=5))

    def test_idf_rare_terms_weigh_more(self, index: BM25Index) -> None:
        assert index.idf("reactor") > index.idf("day")

    def test_empty_query(self, index: BM25Index) -> None:
        assert index.search("the and of", top_k=5) == []


class TestFusion:
    def test_rrf_scores(self) -> None:
        fused = dict(reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=60))
        assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["x"] == pytest.approx(1 / 61)
        assert max(fused, key=fused.__getitem__) == "y"

    def test_weights(self) -> None:
        fused = reciprocal_rank_fusion([["x"], ["z"]], weights=[1.0, 2.0])
        assert fused[0][0] == "z"

    def test_ties_keep_first_seen_order(self) -> None:
        assert [i for i, _ in reciprocal_rank_fusion([["p"], ["q"]])] == ["p", "q"]

    def test_weight_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], weights=[1.0, 1.0])


def _chunk(cid: str) -> RetrievedChunk:
    doc, text = DOCS[cid]
    return RetrievedChunk(chunk_id=cid, document_id=doc, text=text, filename=f"{doc}.txt")


class TestRerankers:
    def test_lexical_reranker_orders_by_coverage(self) -> None:
        reranker = LexicalReranker()
        ranked = reranker.rerank("annual leave days", [_chunk("b:0"), _chunk("a:0")])
        assert ranked[0].chunk_id == "a:0"
        assert all(0.0 <= (c.relevance or 0) <= 1.0 for c in ranked)
        assert ranked[1].relevance == 0.0

    def test_lexical_reranker_empty_query(self) -> None:
        ranked = LexicalReranker().rerank("the", [_chunk("a:0")])
        assert ranked[0].relevance == 0.0

    def test_cross_encoder_maps_logits_to_probabilities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reranker = CrossEncoderReranker("fake-model")
        monkeypatch.setattr(reranker, "_logits", lambda pairs: [-6.0, 3.0][: len(pairs)])
        ranked = reranker.rerank("q", [_chunk("a:0"), _chunk("b:0")])
        assert ranked[0].chunk_id == "b:0"
        assert ranked[0].relevance == pytest.approx(1 / (1 + math.exp(-3.0)))
        assert ranked[1].relevance == pytest.approx(1 / (1 + math.exp(6.0)))


class TestInMemoryVectorStore:
    def test_query_filter_delete(self) -> None:
        emb = HashingEmbedder()
        store = InMemoryVectorStore()
        store.upsert(
            [
                VectorRecord(
                    id=cid,
                    embedding=emb.embed_documents([t])[0],
                    text=t,
                    metadata={"document_id": d},
                )
                for cid, (d, t) in DOCS.items()
            ]
        )
        assert store.count() == 4
        hits = store.query(emb.embed_query("reactor coolant temperature"), top_k=2)
        assert hits[0].id == "b:0" and hits[0].score > hits[1].score
        assert all(
            h.metadata["document_id"] == "a" for h in store.query(emb.embed_query("x"), 5, ["a"])
        )
        store.delete_document("a")
        assert store.count() == 2


class TestHybridRetriever:
    @pytest.fixture
    def retriever(self) -> HybridRetriever:
        emb = HashingEmbedder()
        store = InMemoryVectorStore()
        index = BM25Index()
        for cid, (doc, text) in DOCS.items():
            store.upsert(
                [VectorRecord(cid, emb.embed_documents([text])[0], text, {"document_id": doc})]
            )
            index.add(cid, text, doc)

        def lookup(ids: list[str]) -> dict[str, ChunkInfo]:
            return {
                i: ChunkInfo(i, DOCS[i][0], DOCS[i][1], f"{DOCS[i][0]}.txt", None, 1, 1, None)
                for i in ids
                if i in DOCS and i != "b:1"  # b:1 simulates a stale vector
            }

        return HybridRetriever(
            embedder=emb,
            vector_store=store,
            lexical_index=index,
            reranker=LexicalReranker(idf=index.idf),
            chunk_lookup=lookup,
            config=RetrieverConfig(candidates=10, rerank_candidates=10),
        )

    def test_hybrid_rerank(self, retriever: HybridRetriever) -> None:
        result = retriever.retrieve("How many days of annual leave?", top_k=2)
        assert result.chunks[0].chunk_id == "a:0"
        assert (
            result.chunks[0].semantic_rank is not None and result.chunks[0].lexical_rank is not None
        )
        assert result.reranker == "lexical-coverage"
        assert {"embed_ms", "semantic_ms", "lexical_ms", "rerank_ms"} <= set(result.timings_ms)

    def test_modes_and_no_rerank(self, retriever: HybridRetriever) -> None:
        lex = retriever.retrieve(
            "reactor coolant", top_k=3, mode=RetrievalMode.LEXICAL, rerank=False
        )
        assert lex.chunks[0].chunk_id == "b:0"
        assert lex.chunks[0].semantic_rank is None and lex.chunks[0].relevance is None
        assert lex.reranker == "none"
        sem = retriever.retrieve("reactor coolant", top_k=3, mode=RetrievalMode.SEMANTIC)
        assert sem.chunks[0].lexical_rank is None

    def test_document_filter_and_stale_ids(self, retriever: HybridRetriever) -> None:
        result = retriever.retrieve("liability clause fees", top_k=5, document_ids=["b"])
        assert all(c.document_id == "b" for c in result.chunks)
        assert "b:1" not in {c.chunk_id for c in result.chunks}  # stale vector skipped
