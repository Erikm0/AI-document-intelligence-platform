from __future__ import annotations

import pytest

from app.core.text import content_terms, count_tokens, split_sentences, stem, tokenize
from app.ingestion.cleaning import (
    is_page_number_line,
    normalize_text,
    remove_repeated_headers_footers,
)


class TestText:
    def test_tokenize_keeps_numbers_and_decimals(self) -> None:
        assert tokenize("Q-factor was 0.73 in Phase II!") == [
            "q",
            "factor",
            "was",
            "0.73",
            "in",
            "phase",
            "ii",
        ]

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("policies", "policy"),
            ("running", "run"),
            ("encrypted", "encrypt"),
            ("class", "class"),
            ("is", "is"),
        ],
    )
    def test_stem(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    def test_content_terms_removes_stopwords_and_stems(self) -> None:
        assert content_terms("What are the retention policies?") == ["retention", "policy"]

    def test_split_sentences(self) -> None:
        text = "First sentence. Second one! Is this third? Yes. e.g. not split mid-word"
        assert split_sentences(text)[:3] == ["First sentence.", "Second one!", "Is this third?"]

    def test_split_sentences_does_not_break_decimals(self) -> None:
        assert split_sentences("The value is 4.2 seconds. Next.") == [
            "The value is 4.2 seconds.",
            "Next.",
        ]

    def test_count_tokens(self) -> None:
        assert count_tokens("") == 0
        assert 10 <= count_tokens("one two three four five six seven eight nine ten") <= 15


class TestCleaning:
    def test_normalize_ligatures_quotes_and_whitespace(self) -> None:
        raw = "The ﬁnal “report”   is here.\r\n\r\n\r\n\r\nNext­ para"
        assert normalize_text(raw) == 'The final "report" is here.\n\nNext para'

    def test_normalize_dehyphenates_line_breaks(self) -> None:
        assert normalize_text("infor-\nmation retrieval") == "information retrieval"

    def test_normalize_keeps_real_hyphens(self) -> None:
        assert normalize_text("state-of-the-art\nModel") == "state-of-the-art\nModel"

    def test_normalize_removes_control_characters(self) -> None:
        assert normalize_text("abc\x00\x07def​") == "abcdef"

    @pytest.mark.parametrize("line", ["3", "Page 4", "- 12 -", "page 2 of 10", "  7  "])
    def test_page_number_lines(self, line: str) -> None:
        assert is_page_number_line(line)

    @pytest.mark.parametrize("line", ["Section 3", "3 apples were eaten", "Chapter Seven"])
    def test_not_page_number_lines(self, line: str) -> None:
        assert not is_page_number_line(line)

    def test_removes_running_headers_and_footers(self) -> None:
        topics = ["alpha", "beta", "gamma", "delta"]
        pages = [
            f"ACME Corp - Confidential\nBody text about {t} with facts.\n"
            f"More on {t}.\nPage {i} of 4"
            for i, t in enumerate(topics, start=1)
        ]
        cleaned = remove_repeated_headers_footers(pages)
        for page, topic in zip(cleaned, topics, strict=True):
            assert "Confidential" not in page
            assert "Page" not in page
            assert f"Body text about {topic}" in page and f"More on {topic}." in page

    def test_keeps_numbers_in_page_body(self) -> None:
        pages = ["Header\nTable:\n42\nvalue\nmore\nrows\nfooter text"]
        assert "42" in remove_repeated_headers_footers(pages)[0]
