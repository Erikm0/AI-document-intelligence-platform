"""Second-stage rerankers.

Bi-encoders embed query and passage independently; a cross-encoder reads them
*together* and is far more precise, at a higher cost - so it only sees the top
fused candidates. Its calibrated score doubles as the evidence signal for the
"refuse when unsupported" guardrail.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Protocol

from app.core.text import content_terms
from app.ingestion.models import build_index_text
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str
    # Calibrated evidence threshold for this reranker's score scale.
    default_min_relevance: float

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _passage(chunk: RetrievedChunk) -> str:
    return build_index_text(chunk.text, title=chunk.title, section=chunk.section)


class CrossEncoderReranker:
    # sigmoid(logit) for MS-MARCO cross-encoders: clearly relevant passages score
    # > 0.5, unrelated ones < 0.01. 0.1 (logit ~ -2.2) keeps borderline evidence
    # for the LLM to judge while rejecting off-topic questions outright.
    default_min_relevance = 0.1

    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 16) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading cross-encoder %s", self.name)
                    self._model = CrossEncoder(self.name, device=self.device, max_length=512)
        return self._model

    def warmup(self) -> None:
        self._load()

    def _logits(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        try:
            import torch

            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                activation_fn=torch.nn.Identity(),
                show_progress_bar=False,
            )
        except TypeError:  # older sentence-transformers without activation_fn
            scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        logits = self._logits([(query, _passage(c)) for c in chunks])
        for chunk, logit in zip(chunks, logits, strict=True):
            # MS-MARCO cross-encoders emit logits; sigmoid gives a 0..1 relevance.
            chunk.relevance = _sigmoid(logit)
        return sorted(chunks, key=lambda c: c.relevance or 0.0, reverse=True)


class LexicalReranker:
    """IDF-weighted query-term coverage with a phrase (bigram) bonus.

    A cheap, dependency-free reranker used when the cross-encoder is disabled and
    in tests. Scores are in [0, 1] and roughly comparable to cross-encoder
    probabilities for thresholding purposes.
    """

    name = "lexical-coverage"
    # Requires that a good share of the *informative* (high-IDF) query terms occur.
    default_min_relevance = 0.35

    def __init__(self, idf: Callable[[str], float] | None = None) -> None:
        self._idf = idf

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q_terms = list(dict.fromkeys(content_terms(query)))
        if not q_terms:
            for chunk in chunks:
                chunk.relevance = 0.0
            return chunks
        weights = {t: max(self._idf(t), 0.05) if self._idf else 1.0 for t in q_terms}
        total = sum(weights.values())
        q_bigrams = set(pairwise(q_terms))

        for chunk in chunks:
            terms = content_terms(_passage(chunk))
            term_set = set(terms)
            coverage = sum(w for t, w in weights.items() if t in term_set) / total
            bonus = 0.0
            if q_bigrams:
                c_bigrams = set(pairwise(terms))
                bonus = 0.15 * len(q_bigrams & c_bigrams) / len(q_bigrams)
            chunk.relevance = min(1.0, coverage * 0.9 + bonus)
        # Stable sort keeps the fusion order for ties.
        return sorted(chunks, key=lambda c: c.relevance or 0.0, reverse=True)
