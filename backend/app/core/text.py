"""Light-weight, dependency-free text utilities shared by BM25, reranking,
token counting and grounding checks."""

from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.'][A-Za-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do
    does doesn't doing don't down during each few for from further had hadn't has hasn't
    have haven't having he he'd he'll he's her here here's hers herself him himself his how
    how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she she'd she'll she's should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while who who's
    whom why why's will with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves also may might must shall us via per within without upon according
    tell explain describe document documents please give list show say said
    """.split()  # noqa: SIM905
)


@lru_cache(maxsize=50_000)
def stem(token: str) -> str:
    """A tiny suffix-stripping stemmer (Porter-lite).

    Good enough to make "policies"/"policy" and "running"/"run" match for BM25
    and grounding, without pulling in NLTK.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, replacement in (
        ("ational", "ate"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("iveness", "ive"),
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            stemmed = token[: -len(suffix)] + replacement
            # "running" -> "runn" -> "run"
            if len(stemmed) > 3 and stemmed[-1] == stemmed[-2] and stemmed[-1] not in "lsz":
                stemmed = stemmed[:-1]
            return stemmed
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (keeps numbers, internal dots/apostrophes)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stop-words removed - the unit of lexical matching."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def count_tokens(text: str) -> int:
    """Approximate LLM/embedding token count.

    Sub-word tokenizers produce ~1.3 tokens per English word; we count words and
    punctuation runs and scale, which is accurate to ~10% and has no dependencies.
    """
    words = len(text.split())
    return int(words * 1.3) + 1 if words else 0


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]
