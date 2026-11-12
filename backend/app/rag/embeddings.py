"""Embedding models.

``SentenceTransformerEmbedder`` is the production implementation (local, open
source, CPU/GPU). ``HashingEmbedder`` is a dependency-free feature-hashing model
used by the test-suite and CI, so the whole pipeline can be exercised without
downloading model weights.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

from app.core.text import content_terms

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    # Cosine similarity above which a chunk counts as evidence (None = unused).
    default_min_similarity: float | None

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` bi-encoder (default: BAAI/bge-small-en-v1.5).

    The model is loaded lazily and shared across threads; ``encode`` is thread-safe
    for inference.
    """

    # Calibrated on evaluation/dataset.json for BAAI/bge-small-en-v1.5: answerable
    # evidence scores >= 0.54, off-topic questions <= 0.42.
    default_min_similarity: float | None = 0.5

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.name)
                    self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._load()
            getter = (
                getattr(model, "get_embedding_dimension", None)
                or model.get_sentence_embedding_dimension
            )
            self._dimension = int(getter())
        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]

    def warmup(self) -> None:
        self._load()


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams of stemmed terms).

    It is *lexical*, not semantic - but it is stable, fast and good enough to test
    ranking, filtering and fusion logic end-to-end.
    """

    # Hashing similarities are lexical and uncalibrated - rely on the reranker.
    default_min_similarity: float | None = None

    def __init__(self, dimension: int = 512) -> None:
        self.name = f"hashing-{dimension}"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        terms = content_terms(text)
        features = terms + [f"{a}_{b}" for a, b in pairwise(terms)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 0.5 if "_" in feature else 1.0
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
