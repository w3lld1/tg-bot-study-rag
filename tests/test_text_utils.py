from dataclasses import dataclass

import pytest

from ragbot.text_utils import (
    clamp_text,
    contains_fake_pages,
    coverage_tokens,
    dedup_docs,
    diversify_docs,
    doc_key,
    extract_numbers_from_text,
    fix_broken_numbers,
    has_number,
    normalize_query,
    safe_page_range,
    word_hit_ratio,
)


@dataclass
class DummyDoc:
    page_content: str
    metadata: dict


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  hello   world  ", "hello world"),
        ("", ""),
        ("\n\tмного\t\tпробелов\n", "много пробелов"),
    ],
)
def test_normalize_query(raw, expected):
    assert normalize_query(raw) == expected


def test_clamp_text_whitespace_and_len():
    assert clamp_text("  a   b   c  ", 100) == "a b c"
    assert clamp_text("abcdef", 3) == "abc"


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({"page": 0}, "1"),
        ({"page_start": 2, "page_end": 4}, "3-5"),
        ({"page_start": 2, "page_end": 2}, "3"),
        ({}, "?"),
        ({"page_start": "x", "page_end": "y"}, "x-y"),
    ],
)
def test_safe_page_range(meta, expected):
    assert safe_page_range(meta) == expected


def test_doc_key_uses_metadata_fields():
    d = DummyDoc("txt", {"source": "a", "page_start": 1, "page_end": 2, "chunk_id": 7})
    assert doc_key(d) == ("a", 1, 2, 7)


def test_dedup_docs_keeps_order_and_limits():
    docs = [
        DummyDoc("a", {"source": "s", "page": 0, "chunk_id": 1}),
        DummyDoc("a-dup", {"source": "s", "page": 0, "chunk_id": 1}),
        DummyDoc("b", {"source": "s", "page": 1, "chunk_id": 2}),
    ]
    out = dedup_docs(docs, max_total=10)
    assert [d.page_content for d in out] == ["a", "b"]


def test_dedup_docs_respects_max_total():
    docs = [DummyDoc(str(i), {"source": "s", "page": i, "chunk_id": i}) for i in range(5)]
    out = dedup_docs(docs, max_total=3)
    assert len(out) == 3


def test_coverage_tokens_filters_short_and_stopwords_and_limit():
    out = coverage_tokens("и в на это очень полезная проверка для сознания животных и экспериментов в нейронауке")
    assert "очень" in out
    assert "проверка" in out
    assert "для" not in out
    assert "и" not in out
    assert len(out) <= 10


def test_word_hit_ratio():
    tokens = ["сознание", "животных", "мозг"]
    text = "Сознание животных изучают. Про мозг тоже говорят."
    assert word_hit_ratio(tokens, text) == pytest.approx(1.0)
    assert word_hit_ratio(["квант"], text) == 0.0


def test_diversify_docs_limits_per_group():
    docs = [
        DummyDoc("a1", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 1}),
        DummyDoc("a2", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 2}),
        DummyDoc("a3", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 3}),
        DummyDoc("b1", {"source": "s", "page": 1, "section_title": "x", "chunk_id": 4}),
    ]
    out = diversify_docs(docs, max_per_group=2)
    assert [d.page_content for d in out] == ["a1", "a2", "b1"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("без цифр", False),
        ("в 2026 году", True),
        ("цена 12 500 ₽", True),
        ("1,25 литра", True),
        ("7\u00A0500 штук", True),
    ],
)
def test_has_number(text, expected):
    assert has_number(text) is expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1 234", "1234"),
        ("12 345 678", "12345678"),
        ("12\n345", "12345"),
        ("1 , 5 %", "1,5%"),
        ("12 \n , \n 5", "12,5"),
    ],
)
def test_fix_broken_numbers(raw, expected):
    assert fix_broken_numbers(raw) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("стр. xx", True),
        ("(стр. ??)", True),
        ("стр. 10", False),
        ("page 12", False),
    ],
)
def test_contains_fake_pages(text, expected):
    assert contains_fake_pages(text) is expected


def test_extract_numbers_from_text():
    out = extract_numbers_from_text("Цена 12 500 ₽, скидка 7.5%, год 2026")
    assert "12500" in out
    assert "7.5%" in out
    assert "2026" in out


def test_extract_numbers_deduplicates_and_handles_minus_symbol():
    out = extract_numbers_from_text("−10 и -10 и 10 и 10")
    assert "-10" in out
    assert "10" in out
    assert out.count("10") == 1
