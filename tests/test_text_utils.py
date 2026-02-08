import pytest

from ragbot.text_utils import (
    clamp_text,
    contains_fake_pages,
    coverage_tokens,
    extract_numbers_from_text,
    fix_broken_numbers,
    has_number,
    normalize_query,
    safe_page_range,
    word_hit_ratio,
)


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
    ],
)
def test_safe_page_range(meta, expected):
    assert safe_page_range(meta) == expected


def test_coverage_tokens_filters_short_and_stopwords():
    out = coverage_tokens("и в на это очень полезная проверка для сознания животных")
    assert "очень" in out
    assert "проверка" in out
    assert "для" not in out
    assert "и" not in out


def test_word_hit_ratio():
    tokens = ["сознание", "животных", "мозг"]
    text = "Сознание животных изучают. Про мозг тоже говорят."
    assert word_hit_ratio(tokens, text) == pytest.approx(1.0)
    assert word_hit_ratio(["квант"], text) == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("без цифр", False),
        ("в 2026 году", True),
        ("цена 12 500 ₽", True),
        ("1,25 литра", True),
    ],
)
def test_has_number(text, expected):
    assert has_number(text) is expected


def test_fix_broken_numbers():
    assert fix_broken_numbers("1 234") == "1234"
    assert fix_broken_numbers("12 345 678") == "12345678"


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
