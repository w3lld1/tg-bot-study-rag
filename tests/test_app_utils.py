import ast
import re
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_functions(*names):
    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    const_names = {"_NBSP", "_NNBSP", "NUM_RE", "BAD_PAGE_PAT", "EXPLICIT_BAD_PAGE"}

    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in const_names for t in targets):
                selected.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in const_names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(APP_PATH), mode="exec")

    ns = {
        "re": re,
        "Dict": dict,
        "Any": object,
        "List": list,
    }
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


FUNCS = load_functions(
    "normalize_query",
    "clamp_text",
    "safe_page_range",
    "coverage_tokens",
    "_word_hit_ratio",
    "has_number",
    "fix_broken_numbers",
    "contains_fake_pages",
    "extract_numbers_from_text",
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
    assert FUNCS["normalize_query"](raw) == expected


def test_clamp_text_whitespace_and_len():
    fn = FUNCS["clamp_text"]
    assert fn("  a   b   c  ", 100) == "a b c"
    assert fn("abcdef", 3) == "abc"


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
    assert FUNCS["safe_page_range"](meta) == expected


def test_coverage_tokens_filters_short_and_stopwords():
    fn = FUNCS["coverage_tokens"]
    out = fn("и в на это очень полезная проверка для сознания животных")
    # stop-words removed, words length >=4
    assert "очень" in out
    assert "проверка" in out
    assert "для" not in out
    assert "и" not in out


def test_word_hit_ratio():
    fn = FUNCS["_word_hit_ratio"]
    tokens = ["сознание", "животных", "мозг"]
    text = "Сознание животных изучают. Про мозг тоже говорят."
    assert fn(tokens, text) == pytest.approx(1.0)
    assert fn(["квант"], text) == 0.0


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
    assert FUNCS["has_number"](text) is expected


def test_fix_broken_numbers():
    fn = FUNCS["fix_broken_numbers"]
    assert fn("1 234") == "1234"
    assert fn("12 345 678") == "12345678"


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
    assert FUNCS["contains_fake_pages"](text) is expected


def test_extract_numbers_from_text():
    fn = FUNCS["extract_numbers_from_text"]
    out = fn("Цена 12 500 ₽, скидка 7.5%, год 2026")
    assert "12500" in out
    assert "7.5%" in out
    assert "2026" in out
