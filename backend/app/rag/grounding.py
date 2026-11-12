"""Post-generation guardrails: refusal detection, citation validation/repair and a
lexical grounding (faithfulness) score.

These checks are deliberately model-free so they are cheap, deterministic and
testable. They complement - not replace - the retrieval-level evidence gate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.text import content_terms, split_sentences, stem
from app.rag.prompts import REFUSAL_SENTINEL
from app.rag.types import RetrievedChunk

_CITATION_GROUP_RE = re.compile(r"\[(\s*\d+(?:\s*[,;]\s*\d+)*\s*)\]")
_SOURCE_WORD_RE = re.compile(r"[\[(]\s*(?:source|src|doc(?:ument)?)\s*#?\s*(\d+)\s*[\])]", re.I)
_SINGLE_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_REFUSAL_PHRASES = (
    "insufficient evidence",
    "insufficient_evidence",
    "do not contain",
    "does not contain",
    "don't contain",
    "doesn't contain",
    "not mentioned",
    "no information",
    "not provided in",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not able to answer",
    "cannot find",
    "could not find",
    "couldn't find",
    "no mention",
    "no basis",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "not possible to determine",
    "not specified",
    "not stated",
)
# Words that attribute rather than claim ("according to the provided source [1]").
_ATTRIBUTION_TERMS = frozenset(
    stem(w)
    for w in [
        "source",
        "sources",
        "provided",
        "context",
        "passage",
        "passages",
        "excerpt",
        "excerpts",
        "states",
        "stated",
        "mentions",
        "mentioned",
        "based",
        "therefore",
        "indicates",
        "indicated",
        "specifies",
        "specified",
        "notes",
        "noted",
        "reports",
        "reported",
    ]
)
_LEADING_VERDICT_RE = re.compile(r"^[\W_]*insufficient[ _]evidence\b", re.I)
# Models sometimes append a bibliography that echoes the source headers:
#   "[1] handbook.txt | p. 1"   /   "Sources:"
_SOURCE_LIST_HEADING_RE = re.compile(r"^[\W_]*(?:sources?|references?|citations?)[\W_]*$", re.I)
_SOURCE_LIST_ITEM_RE = re.compile(r"^[-*\s]*\[\d+\]\s*(.+)$")
_FILENAME_RE = re.compile(r"\b[\w.\- ()]+\.(?:pdf|docx|txt)\b", re.I)
# "The documents do not specify ...", "the report doesn't mention ..."
_META_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|don't|doesn't|did\s+not)\s+"
    r"(?:mention|specify|state|cover|contain|include|say|provide|discuss|address|describe)",
    re.I,
)


def _mentions_missing_info(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _REFUSAL_PHRASES) or bool(_META_RE.search(text))


@dataclass(slots=True)
class GroundingReport:
    answer: str
    cited_indices: list[int]
    invalid_citations: list[int] = field(default_factory=list)
    repaired: bool = False
    grounding_score: float = 0.0
    unsupported_numbers: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


def normalize_citations(text: str) -> str:
    """``[Source 2]`` -> ``[2]``, ``[1, 3]`` -> ``[1][3]``."""
    text = _SOURCE_WORD_RE.sub(lambda m: f"[{m.group(1)}]", text)

    def _split(m: re.Match[str]) -> str:
        nums = re.split(r"\s*[,;]\s*", m.group(1).strip())
        return "".join(f"[{n}]" for n in nums if n)

    return _CITATION_GROUP_RE.sub(_split, text)


def extract_citations(text: str) -> list[int]:
    seen: dict[int, None] = {}
    for m in _SINGLE_CITATION_RE.finditer(normalize_citations(text)):
        seen[int(m.group(1))] = None
    return list(seen)


def strip_invalid_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    invalid: list[int] = []

    def _check(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            return m.group(0)
        invalid.append(n)
        return ""

    cleaned = _SINGLE_CITATION_RE.sub(_check, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), sorted(set(invalid))


def strip_source_list(answer: str, filenames: Sequence[str] = ()) -> str:
    """Remove an echoed bibliography ("Sources:" / "[1] report.pdf | p. 3") lines.

    The source panel already shows the sources; echoed headers would also count
    as unsupported words in the grounding score.
    """
    names = [n.lower() for n in filenames if n]
    kept: list[str] = []
    for line in answer.split("\n"):
        stripped = line.strip()
        if _SOURCE_LIST_HEADING_RE.match(stripped):
            continue
        item = _SOURCE_LIST_ITEM_RE.match(stripped)
        if item and len(item.group(1)) < 200:
            rest = item.group(1).lower()
            if any(n in rest for n in names) or _FILENAME_RE.match(rest):
                continue
        kept.append(line)
    return "\n".join(kept).strip()


def tidy_answer(answer: str, sources: Sequence[RetrievedChunk]) -> str:
    """Remove prompt artefacts small models copy into their answers.

    Observed with llama3.2:3b:
      * a literal ``[n]`` placeholder copied from the instructions;
      * source-header fragments: "[1] report.pdf", "[1] Section: 4. Expenses";
      * citations stranded after the full stop: "... per day. [1]".
    """
    answer = re.sub(r"\s*\[n\]", "", answer)
    answer = strip_source_list(answer, [s.filename for s in sources])
    for source in sources:
        # "[1] report.pdf, ..." / "[1] (report.pdf) ..." -> "[1] ..."
        answer = re.sub(
            rf"(\[\d+\])[ \t]*\(?{re.escape(source.filename)}\)?,?", r"\1", answer, flags=re.I
        )
        if source.section:
            answer = re.sub(
                rf"[ \t]*\(?Section:[ \t]*{re.escape(source.section)}\)?\.?", "", answer, flags=re.I
            )
    answer = re.sub(r"([.!?])[ \t]+((?:\[\d+\])+)[ \t]*\.?(?=[ \t]|\n|$)", r" \2\1", answer)
    answer = re.sub(r"(\[\d+\])(?:[ \t]*\1)+", r"\1", answer)  # "[1] [1]" -> "[1]"
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"[ \t]+([.,;:])", r"\1", answer)
    return answer.strip()


def is_refusal(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # "Insufficient evidence. The source [1] mentions X but not Y..." - the model
    # leads with its verdict; the citation only explains what is missing.
    if _LEADING_VERDICT_RE.match(stripped):
        return True
    upper = stripped.upper()
    if REFUSAL_SENTINEL in upper or "INSUFFICIENT EVIDENCE" in upper:
        # A sentinel tacked onto the end of a real, cited answer is still an answer.
        remainder = upper.replace(REFUSAL_SENTINEL, "").replace("INSUFFICIENT EVIDENCE", "")
        return not (extract_citations(stripped) and len(remainder.strip()) > 80)
    # Every sentence only describes what the documents do *not* say.
    sentences = _sentence_spans(stripped)
    if sentences and all(_is_meta_sentence(s) for s in sentences):
        return True
    if extract_citations(stripped):
        return False
    return len(stripped) < 400 and _mentions_missing_info(stripped)


def _sentence_spans(answer: str) -> list[str]:
    spans: list[str] = []
    for paragraph in answer.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            spans.extend(split_sentences(paragraph) or [paragraph])
    return spans


def _is_meta_sentence(sentence: str) -> bool:
    """Statements about what the documents do *not* say are not factual claims."""
    return _mentions_missing_info(sentence)


def _numbers(text: str) -> set[str]:
    """Normalised numbers in ``text`` ("18,400" -> "18400", "6%" -> "6")."""
    return {
        n.rstrip("%").replace(",", "")
        for n in _NUMBER_RE.findall(_SINGLE_CITATION_RE.sub("", text))
    } - {""}


def repair_citations(
    answer: str, sources: Sequence[RetrievedChunk], *, min_overlap: float = 0.5
) -> tuple[str, bool]:
    """Attach a citation to uncited factual sentences that clearly come from a source.

    Small local models frequently forget the [n] markers even when the answer is
    faithful; rather than throwing such answers away, attribute each sentence to
    the source it overlaps most - but only when the overlap is strong *and* every
    number in the sentence literally appears in that source (a sentence with an
    invented figure must never be laundered into a cited claim).
    """
    source_terms = [set(content_terms(s.text)) for s in sources]
    source_numbers = [_numbers(s.text) for s in sources]
    repaired = False
    out_lines: list[str] = []
    for line in answer.split("\n"):
        sentences = split_sentences(line) if line.strip() else []
        if not sentences:
            out_lines.append(line)
            continue
        new_sentences: list[str] = []
        for sentence in sentences:
            if _SINGLE_CITATION_RE.search(sentence) or _is_meta_sentence(sentence):
                new_sentences.append(sentence)
                continue
            terms = set(content_terms(sentence))
            if len(terms) < 3:
                new_sentences.append(sentence)
                continue
            best_idx, best = -1, 0.0
            for i, s_terms in enumerate(source_terms):
                overlap = len(terms & s_terms) / len(terms)
                if overlap > best:
                    best_idx, best = i, overlap
            if (
                best_idx >= 0
                and best >= min_overlap
                and _numbers(sentence) <= source_numbers[best_idx]
            ):
                stripped = sentence.rstrip()
                if stripped and stripped[-1] in ".!?":
                    sentence = f"{stripped[:-1]} [{best_idx + 1}]{stripped[-1]}"
                else:
                    sentence = f"{stripped} [{best_idx + 1}]"
                repaired = True
            new_sentences.append(sentence)
        indent = line[: len(line) - len(line.lstrip())]
        out_lines.append(indent + " ".join(new_sentences))
    return "\n".join(out_lines), repaired


def grounding_score(answer: str, sources: Sequence[RetrievedChunk]) -> tuple[float, list[str]]:
    """Share of the answer's content words that are supported by the sources it cites.

    Each sentence is compared against the sources it cites (or, if it cites none,
    against all sources). Returns ``(score, unsupported_numbers)``; numbers are
    checked exactly because invented figures are the most damaging hallucination.
    """
    if not sources:
        return 0.0, []
    all_terms = [set(content_terms(s.text)) for s in sources]
    all_numbers = [_numbers(s.text) for s in sources]

    supported = 0
    total = 0
    unsupported_numbers: list[str] = []
    for sentence in _sentence_spans(answer):
        if _is_meta_sentence(sentence):
            continue
        cited = [int(n) for n in _SINGLE_CITATION_RE.findall(sentence)]
        body = _SINGLE_CITATION_RE.sub("", sentence)
        terms = [t for t in content_terms(body) if t not in _ATTRIBUTION_TERMS]
        if not terms:
            continue
        pool: set[str] = set()
        numbers: set[str] = set()
        valid = [c for c in cited if 1 <= c <= len(sources)]
        for c in valid or range(1, len(sources) + 1):
            pool |= all_terms[c - 1]
            numbers |= all_numbers[c - 1]
        supported += sum(1 for t in terms if t in pool)
        total += len(terms)
        # Figures must appear in the very sources the sentence cites.
        for number in _NUMBER_RE.findall(body):
            normalized = number.rstrip("%").replace(",", "")
            if normalized and normalized not in numbers and len(normalized) > 1:
                unsupported_numbers.append(number)
    if total == 0:
        return 0.0, unsupported_numbers
    return round(supported / total, 4), unsupported_numbers


def evaluate_answer(raw_answer: str, sources: Sequence[RetrievedChunk]) -> GroundingReport:
    """Run all post-generation guardrails on a raw LLM answer."""
    answer = normalize_citations(raw_answer.strip())
    answer = tidy_answer(answer, sources)
    if is_refusal(answer):
        return GroundingReport(
            answer=answer,
            cited_indices=[],
            refused=True,
            refusal_reason="model_insufficient_evidence",
        )
    # Remove a trailing sentinel the model may have appended to a real answer.
    answer = re.sub(rf"\s*{REFUSAL_SENTINEL}\s*$", "", answer).strip()

    answer, invalid = strip_invalid_citations(answer, len(sources))
    repaired = False
    # Repair only answers where the model *forgot* citations. A model that cited a
    # source that does not exist was not reading the sources - never rescue that.
    if not extract_citations(answer) and not invalid:
        answer, repaired = repair_citations(answer, sources)
    cited = extract_citations(answer)
    score, unsupported_numbers = grounding_score(answer, sources)
    report = GroundingReport(
        answer=answer,
        cited_indices=cited,
        invalid_citations=invalid,
        repaired=repaired,
        grounding_score=score,
        unsupported_numbers=unsupported_numbers,
    )
    if not cited:
        report.refused = True
        report.refusal_reason = "no_supporting_citations"
    return report
