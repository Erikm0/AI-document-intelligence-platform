"""In-process BM25 (Okapi) index for the lexical half of hybrid search.

Dense embeddings are weak on exact identifiers, numbers, acronyms and rare
terms ("clause 14.3", "ISO-27001", "Q3 revenue"); BM25 is strong there. The index
is rebuilt from PostgreSQL on start-up and updated incrementally when documents
are ingested or deleted. For very large corpora swap this for PostgreSQL
full-text search or OpenSearch behind the same interface.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.text import content_terms


@dataclass(slots=True)
class LexicalHit:
    id: str
    score: float
    document_id: str


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._doc_of: dict[str, str] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._tf)

    def add(self, chunk_id: str, text: str, document_id: str) -> None:
        terms = content_terms(text)
        with self._lock:
            if chunk_id in self._tf:
                self._remove(chunk_id)
            tf = Counter(terms)
            self._tf[chunk_id] = tf
            self._len[chunk_id] = len(terms)
            self._doc_of[chunk_id] = document_id
            self._total_len += len(terms)
            for term in tf:
                self._postings[term].add(chunk_id)

    def add_many(self, items: Iterable[tuple[str, str, str]]) -> None:
        for chunk_id, text, document_id in items:
            self.add(chunk_id, text, document_id)

    def _remove(self, chunk_id: str) -> None:
        tf = self._tf.pop(chunk_id, None)
        if tf is None:
            return
        self._total_len -= self._len.pop(chunk_id, 0)
        self._doc_of.pop(chunk_id, None)
        for term in tf:
            postings = self._postings.get(term)
            if postings is not None:
                postings.discard(chunk_id)
                if not postings:
                    del self._postings[term]

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            for chunk_id in [c for c, d in self._doc_of.items() if d == document_id]:
                self._remove(chunk_id)

    def clear(self) -> None:
        with self._lock:
            self._tf.clear()
            self._len.clear()
            self._doc_of.clear()
            self._postings.clear()
            self._total_len = 0

    def idf(self, term: str) -> float:
        n = len(self._tf)
        df = len(self._postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, top_k: int, document_ids: list[str] | None = None
    ) -> list[LexicalHit]:
        q_terms = list(dict.fromkeys(content_terms(query)))
        if not q_terms:
            return []
        allowed = set(document_ids) if document_ids else None
        with self._lock:
            n = len(self._tf)
            if n == 0:
                return []
            avg_len = self._total_len / n if n else 1.0
            scores: dict[str, float] = defaultdict(float)
            for term in q_terms:
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = self.idf(term)
                for chunk_id in postings:
                    if allowed is not None and self._doc_of[chunk_id] not in allowed:
                        continue
                    freq = self._tf[chunk_id][term]
                    length = self._len[chunk_id] or 1
                    denom = freq + self.k1 * (1 - self.b + self.b * length / avg_len)
                    scores[chunk_id] += idf * freq * (self.k1 + 1) / denom
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [LexicalHit(id=c, score=s, document_id=self._doc_of[c]) for c, s in ranked]
