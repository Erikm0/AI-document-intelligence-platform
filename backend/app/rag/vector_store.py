"""Vector store abstraction with ChromaDB and in-memory implementations."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.core.exceptions import VectorStoreUnavailableError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorRecord:
    id: str
    embedding: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float  # cosine similarity in [-1, 1]
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    name: str

    def upsert(self, records: list[VectorRecord]) -> None: ...

    def query(
        self, embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[VectorHit]: ...

    def delete_document(self, document_id: str) -> None: ...

    def count(self) -> int: ...

    def ping(self) -> bool: ...


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool values (no None)."""
    return {k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)}


class ChromaVectorStore:
    """ChromaDB-backed store using cosine distance (HNSW).

    The client is created lazily: the HTTP client contacts the server on
    construction, and the API must be able to start (and report the outage via
    ``/health/ready``) before the Chroma container is up.
    """

    def __init__(
        self,
        client: Any | Callable[[], Any],
        collection_name: str,
        *,
        label: str = "chroma",
    ) -> None:
        self.name = label
        self._client_or_factory = client
        self._client_obj: Any = None if callable(client) else client
        self._collection_name = collection_name
        self._collection: Any = None
        self._lock = threading.Lock()

    @property
    def _client(self) -> Any:
        if self._client_obj is None:
            with self._lock:
                if self._client_obj is None:
                    try:
                        self._client_obj = self._client_or_factory()  # type: ignore[operator]
                    except Exception as exc:
                        raise VectorStoreUnavailableError(
                            f"Cannot connect to the vector store ({self.name}): {exc}"
                        ) from exc
        return self._client_obj

    @classmethod
    def http(cls, host: str, port: int, collection: str) -> ChromaVectorStore:
        def factory() -> Any:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            return chromadb.HttpClient(
                host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
            )

        return cls(factory, collection, label=f"chroma-http://{host}:{port}")

    @classmethod
    def persistent(cls, path: Path, collection: str) -> ChromaVectorStore:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        return cls(client, collection, label="chroma-persistent")

    @classmethod
    def ephemeral(cls, collection: str = "test") -> ChromaVectorStore:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        return cls(client, collection, label="chroma-ephemeral")

    @property
    def collection(self) -> Any:
        if self._collection is None:
            client = self._client
            with self._lock:
                if self._collection is None:
                    try:
                        self._collection = client.get_or_create_collection(
                            name=self._collection_name,
                            metadata={"hnsw:space": "cosine"},
                            embedding_function=None,
                        )
                    except Exception as exc:
                        raise VectorStoreUnavailableError(
                            f"Cannot reach the vector store: {exc}"
                        ) from exc
        return self._collection

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        batch = 500
        try:
            for i in range(0, len(records), batch):
                part = records[i : i + batch]
                self.collection.upsert(
                    ids=[r.id for r in part],
                    embeddings=[r.embedding for r in part],
                    documents=[r.text for r in part],
                    metadatas=[_clean_metadata(r.metadata) for r in part],
                )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Vector upsert failed: {exc}") from exc

    def query(
        self, embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[VectorHit]:
        where: dict[str, Any] | None = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )
        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Vector query failed: {exc}") from exc

        hits: list[VectorHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            hits.append(
                VectorHit(
                    id=chunk_id,
                    score=1.0 - float(dists[i]),  # cosine distance -> similarity
                    text=docs[i] if docs else "",
                    metadata=dict(metas[i] or {}) if metas else {},
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Vector delete failed: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(str(exc)) from exc

    def ping(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            raise VectorStoreUnavailableError(f"Chroma heartbeat failed: {exc}") from exc


class InMemoryVectorStore:
    """Brute-force cosine search over a NumPy matrix. Thread-safe."""

    def __init__(self) -> None:
        self.name = "memory"
        self._records: dict[str, VectorRecord] = {}
        self._lock = threading.RLock()

    def upsert(self, records: list[VectorRecord]) -> None:
        with self._lock:
            for record in records:
                self._records[record.id] = record

    def query(
        self, embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[VectorHit]:
        with self._lock:
            candidates = [
                r
                for r in self._records.values()
                if not document_ids or r.metadata.get("document_id") in set(document_ids)
            ]
        if not candidates:
            return []
        matrix = np.asarray([r.embedding for r in candidates], dtype=np.float32)
        query = np.asarray(embedding, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        norms[norms == 0] = 1.0
        scores = matrix @ query / norms
        order = np.argsort(-scores)[:top_k]
        return [
            VectorHit(
                id=candidates[i].id,
                score=float(scores[i]),
                text=candidates[i].text,
                metadata=dict(candidates[i].metadata),
            )
            for i in order
        ]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            for key in [
                k for k, r in self._records.items() if r.metadata.get("document_id") == document_id
            ]:
                del self._records[key]

    def count(self) -> int:
        return len(self._records)

    def ping(self) -> bool:
        return True
