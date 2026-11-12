from __future__ import annotations

import pytest

from app.rag.grounding import (
    evaluate_answer,
    extract_citations,
    grounding_score,
    is_refusal,
    normalize_citations,
    repair_citations,
    strip_invalid_citations,
    tidy_answer,
)
from app.rag.types import RetrievedChunk

SOURCES = [
    RetrievedChunk(
        chunk_id="d:0",
        document_id="d",
        filename="handbook.pdf",
        text="Employees receive 26 days of paid annual leave per calendar year.",
    ),
    RetrievedChunk(
        chunk_id="d:1",
        document_id="d",
        filename="handbook.pdf",
        text="Remote work is allowed for up to three days per week with manager approval.",
    ),
]


class TestCitationParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Fact [1].", "Fact [1]."),
            ("Fact [1, 2].", "Fact [1][2]."),
            ("Fact [Source 2].", "Fact [2]."),
            ("Fact (source 1).", "Fact [1]."),
            ("Fact [1; 3]", "Fact [1][3]"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_citations(raw) == expected

    def test_extract_deduplicates_in_order(self) -> None:
        assert extract_citations("A [2]. B [1][2]. C [Source 3].") == [2, 1, 3]

    def test_strip_invalid(self) -> None:
        cleaned, invalid = strip_invalid_citations("A [1]. B [7]. C [0].", n_sources=2)
        assert cleaned == "A [1]. B. C."
        assert invalid == [0, 7]


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "INSUFFICIENT_EVIDENCE",
            "insufficient_evidence",
            "",
            "The provided sources do not contain information about salaries.",
            "I cannot find this in the documents.",
        ],
    )
    def test_refusals(self, text: str) -> None:
        assert is_refusal(text)

    def test_answer_with_citations_is_not_refusal(self) -> None:
        assert not is_refusal("Employees get 26 days [1]. The documents do not cover sick leave.")

    def test_long_cited_answer_with_trailing_sentinel_is_not_refusal(self) -> None:
        text = "Employees receive 26 days of paid annual leave per year as stated [1]. " * 2
        assert not is_refusal(text + "INSUFFICIENT_EVIDENCE")


class TestRepairAndGrounding:
    def test_repair_attaches_best_source(self) -> None:
        repaired, changed = repair_citations(
            "Employees receive 26 days of paid annual leave.", SOURCES
        )
        assert changed and repaired == "Employees receive 26 days of paid annual leave [1]."

    def test_repair_ignores_unsupported_sentences(self) -> None:
        repaired, changed = repair_citations("The CEO earns a large bonus every quarter.", SOURCES)
        assert not changed and "[" not in repaired

    def test_grounded_answer_scores_high(self) -> None:
        score, numbers = grounding_score(
            "Employees receive 26 days of paid annual leave [1].", SOURCES
        )
        assert score > 0.9 and numbers == []

    def test_fabricated_answer_scores_low_and_flags_numbers(self) -> None:
        score, numbers = grounding_score(
            "Executives receive 40 days of sabbatical and a 15% bonus [1].", SOURCES
        )
        assert score < 0.5
        assert "40" in numbers and "15%" in numbers

    def test_meta_sentences_are_ignored(self) -> None:
        score, _ = grounding_score(
            "Employees receive 26 days of paid annual leave [1]. "
            "The documents do not mention sick leave or bonuses.",
            SOURCES,
        )
        assert score > 0.9


class TestEvaluateAnswer:
    def test_valid_answer(self) -> None:
        report = evaluate_answer("Up to three days per week [2].", SOURCES)
        assert not report.refused and report.cited_indices == [2]

    def test_sentinel_refusal(self) -> None:
        report = evaluate_answer("INSUFFICIENT_EVIDENCE", SOURCES)
        assert report.refused and report.refusal_reason == "model_insufficient_evidence"

    def test_only_invalid_citations_is_refused(self) -> None:
        report = evaluate_answer("The CEO earns 2 million euros [5].", SOURCES)
        assert report.refused
        assert report.refusal_reason == "no_supporting_citations"
        assert report.invalid_citations == [5]

    def test_uncited_but_faithful_answer_is_repaired(self) -> None:
        report = evaluate_answer("Remote work is allowed for up to three days per week.", SOURCES)
        assert not report.refused and report.repaired and report.cited_indices == [2]

    def test_uncited_world_knowledge_answer_is_refused(self) -> None:
        report = evaluate_answer("Paris is the capital of France.", SOURCES)
        assert report.refused


class TestRealModelOutputs:
    """Shapes observed from small local models (LFM3 1.7B via Ollama)."""

    def test_leading_verdict_with_explanatory_citation_is_refusal(self) -> None:
        text = (
            "Insufficient evidence. The provided source [1] mentions that full-time employees "
            "receive 26 days of paid annual leave per calendar year, but it does not contain any "
            "information about the CEO salary. Therefore, there is no basis to determine the CEO "
            "salary from the given source."
        )
        assert is_refusal(text)
        report = evaluate_answer(text, SOURCES)
        assert report.refused and report.refusal_reason == "model_insufficient_evidence"

    def test_only_meta_sentences_is_refusal(self) -> None:
        assert is_refusal("Source [1] does not mention bonuses. The salary is not specified [2].")

    def test_partial_answer_is_not_refusal(self) -> None:
        text = (
            "Employees receive 26 days of paid annual leave [1]. "
            "The documents do not cover sick leave."
        )
        assert not is_refusal(text)

    def test_echoed_source_list_is_stripped(self) -> None:
        raw = (
            "According to the source [1], full-time employees receive 26 days of paid annual leave "
            "per calendar year.\n\nSources:\n[1] handbook.pdf | p. 1"
        )
        report = evaluate_answer(raw, SOURCES)
        assert not report.refused
        assert "handbook.pdf" not in report.answer and "Sources:" not in report.answer
        assert report.answer.endswith("per calendar year.")
        assert report.grounding_score > 0.8


class TestTidyAnswer:
    """Verbatim llama3.2:3b outputs from the Docker smoke test."""

    sources = [
        RetrievedChunk(
            chunk_id="a:3",
            document_id="a",
            filename="aurora_employee_handbook.txt",
            section="4. Travel and Expenses",
            text="Meal expenses during business travel are reimbursed up to 45 euros per day.",
        )
    ]

    def test_placeholder_marker_is_removed(self) -> None:
        raw = "The Phase II cost was 48.5 million euros [1].  [n]"
        assert tidy_answer(raw, self.sources) == "The Phase II cost was 48.5 million euros [1]."

    def test_inline_header_echo_is_removed(self) -> None:
        raw = (
            "According to [1] aurora_employee_handbook.txt, the daily limit for meal expenses at "
            "Aurora Robotics is 45 euros per day. [1] Section: 4. Travel and Expenses."
        )
        assert tidy_answer(raw, self.sources) == (
            "According to [1] the daily limit for meal expenses at Aurora Robotics is "
            "45 euros per day [1]."
        )

    def test_stranded_citation_moves_before_full_stop(self) -> None:
        raw = "Availability is 99.5 percent. [1] Credits apply below that. [1][2]"
        assert tidy_answer(raw, self.sources) == (
            "Availability is 99.5 percent [1]. Credits apply below that [1][2]."
        )

    def test_clean_answer_is_unchanged(self) -> None:
        raw = (
            "Meal expenses are reimbursed up to 45 euros per day [1].\n\n- Alcohol is excluded [1]."
        )
        assert tidy_answer(raw, self.sources) == raw
