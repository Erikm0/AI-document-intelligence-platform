"""ChromaVectorStore against a real (in-process) ChromaDB."""

from __future__ import annotations

import uuid

import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import ChromaVectorStore, VectorRecord

pytestmark = [pytest.mark.chroma, pytest.mark.integration]
pytest.importorskip("chromadb")

TEXTS = {
    "a:0": ("a", "Employees receive 26 days of annual leave."),
    "a:1": ("a", "Remote work is allowed three days per week."),
    "b:0": ("b", "The reactor coolant outlet is 520 degrees Celsius."),
}


@pytest.fixture
def store() -> ChromaVectorStore:
    return ChromaVectorStore.ephemeral(collection=f"test_{uuid.uuid4().hex[:8]}")


def _records(emb: HashingEmbedder) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=cid,
            embedding=emb.embed_documents([text])[0],
            text=text,
            metadata={"document_id": doc, "page_start": 1, "section": None},  # None is dropped
        )
        for cid, (doc, text) in TEXTS.items()
    ]


def test_upsert_query_filter_delete(store: ChromaVectorStore) -> None:
    emb = HashingEmbedder()
    store.upsert(_records(emb))
    assert store.count() == 3
    assert store.ping()

    hits = store.query(emb.embed_query("reactor coolant temperature"), top_k=2)
    assert hits[0].id == "b:0"
    assert -1.0 <= hits[1].score <= hits[0].score <= 1.0001
    assert hits[0].metadata["document_id"] == "b"

    only_a = store.query(emb.embed_query("days"), top_k=5, document_ids=["a"])
    assert {h.id for h in only_a} == {"a:0", "a:1"}
    both = store.query(emb.embed_query("days"), top_k=5, document_ids=["a", "b"])
    assert len(both) == 3

    store.upsert(_records(emb)[:1])  # idempotent upsert
    assert store.count() == 3

    store.delete_document("a")
    assert store.count() == 1
    assert store.query(emb.embed_query("leave"), top_k=5, document_ids=["a"]) == []


def test_unreachable_server_is_reported_not_fatal() -> None:
    from app.core.exceptions import VectorStoreUnavailableError

    store = ChromaVectorStore.http("127.0.0.1", 9, "unreachable")  # nothing listens on port 9
    with pytest.raises(VectorStoreUnavailableError):
        store.ping()
    with pytest.raises(VectorStoreUnavailableError):
        store.count()
