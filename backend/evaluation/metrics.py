"""Information-retrieval metrics (binary relevance)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    """1.0 if any relevant item is in the top-k (single-evidence questions)."""
    return 1.0 if any(relevant_flags[:k]) else 0.0


def reciprocal_rank(relevant_flags: Sequence[bool]) -> float:
    for rank, flag in enumerate(relevant_flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, f in enumerate(relevant_flags[:k], start=1) if f)
    n_rel = min(sum(1 for f in relevant_flags if f), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_rel + 1))
    return dcg / idcg


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
