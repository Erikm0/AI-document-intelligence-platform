"""Rank fusion for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion (Cormack et al., 2009).

    ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks. RRF needs no
    score calibration between retrievers, which is exactly the problem when mixing
    cosine similarities with unbounded BM25 scores.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    fused: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    order = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] += weight / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
