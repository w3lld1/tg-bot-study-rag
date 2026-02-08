import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ragbot.text_utils import (
    DATE_RE,
    clamp_text,
    coverage_tokens,
    doc_key,
    extract_numbers_from_text,
    has_number,
    safe_page_range,
    word_hit_ratio,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@dataclass
class DummyDoc:
    page_content: str
    metadata: dict


def load_app_symbols(function_names):
    src = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    const_names = {"_CITATION_ONLY_STRONG", "_NOT_FOUND_PAT"}
    selected = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in const_names for t in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected.append(node)

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(APP_PATH), mode="exec")

    ns = {
        "re": re,
        "json": json,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Document": DummyDoc,
        "DATE_RE": DATE_RE,
        "safe_page_range": safe_page_range,
        "clamp_text": clamp_text,
        "coverage_tokens": coverage_tokens,
        "_word_hit_ratio": word_hit_ratio,
        "has_number": has_number,
        "extract_numbers_from_text": extract_numbers_from_text,
        "doc_key": doc_key,
    }
    exec(code, ns, ns)
    return {name: ns[name] for name in function_names}


APP = load_app_symbols(
    {
        "coverage_score",
        "_extract_first_json_object",
        "invoke_json_robust",
        "detect_intent_fast",
        "format_context",
        "is_not_found_answer",
        "add_neighbors_from_parent_map",
        "_num_rerank_score",
        "rerank_numbers_heuristic",
        "answer_numbers_not_in_context",
    }
)


def test_extract_first_json_object():
    fn = APP["_extract_first_json_object"]
    assert fn('{"a": 1}') == {"a": 1}
    assert fn('prefix {"a": 2} suffix') == {"a": 2}
    assert fn("no json here") is None


def test_invoke_json_robust_uses_json_chain_first():
    class Chain:
        def invoke(self, _):
            return {"ok": True}

    fn = APP["invoke_json_robust"]
    out = fn(Chain(), Chain(), {"x": 1}, {"ok": False}, retries=1)
    assert out == {"ok": True}


def test_invoke_json_robust_falls_back_to_string_chain():
    class Broken:
        def invoke(self, _):
            raise RuntimeError("boom")

    class StringChain:
        def invoke(self, _):
            return 'text {"intent": "default", "confidence": 0.5}'

    fn = APP["invoke_json_robust"]
    out = fn(Broken(), StringChain(), {"x": 1}, {"intent": "fallback"}, retries=1)
    assert out["intent"] == "default"


@pytest.mark.parametrize(
    "q, expected",
    [
        ("процитируй дословно", "citation_only"),
        ("сделай сводку", "summary"),
        ("сравни 2021 и 2022", "compare"),
        ("какие требования стандарта", "requirements"),
        ("как сделать шаги процедуры", "procedure"),
        ("что такое дефолт", "definition"),
        ("сколько процентов в 2026", "numbers_and_dates"),
        ("просто расскажи", "default"),
    ],
)
def test_detect_intent_fast(q, expected):
    fn = APP["detect_intent_fast"]
    intent, _ = fn(q)
    assert intent == expected


def test_coverage_score_numbers_boost():
    fn = APP["coverage_score"]
    docs = [DummyDoc("Выручка 12 500 и рост 7%", {"page": 0})]
    score = fn("сколько рост", docs, "numbers_and_dates")
    assert score > 0.4


def test_format_context_limits_and_truncates():
    fn = APP["format_context"]
    docs = [
        DummyDoc("A" * 300, {"page": 0, "section_title": "s1"}),
        DummyDoc("B" * 300, {"page": 1, "section_title": "s2"}),
    ]
    out = fn(docs, limit=2, max_chars=350)
    assert "[стр. 1 | s1]" in out
    assert len(out) <= 350


def test_is_not_found_answer():
    fn = APP["is_not_found_answer"]
    assert fn("В документе не найдено") is True
    assert fn("не найдено.") is True
    assert fn("В документе не найдено по этому вопросу") is True
    assert fn("Нашёл ответ на стр. 10") is False


def test_add_neighbors_from_parent_map_adds_window_chunks():
    fn = APP["add_neighbors_from_parent_map"]
    base = DummyDoc("c2", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 2})
    n1 = DummyDoc("c1", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 1})
    n3 = DummyDoc("c3", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 3})
    far = DummyDoc("c9", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 9})

    parent_key = ("s", 0, 0, "x")
    parent_map = {parent_key: [n1, base, n3, far]}

    out = fn(parent_map, [base], window=1, max_total=10)
    contents = [d.page_content for d in out]
    assert "c1" in contents and "c3" in contents
    assert "c9" not in contents


def test_rerank_numbers_heuristic_prefers_relevant_docs():
    fn = APP["rerank_numbers_heuristic"]
    docs = [
        DummyDoc("просто текст", {"page": 0}),
        DummyDoc("рост 7% в 2026 году", {"page": 1}),
        DummyDoc("метрика 123", {"page": 2}),
    ]
    out = fn("сколько рост процентов 2026", docs, keep=2)
    assert out[0].page_content == "рост 7% в 2026 году"


def test_answer_numbers_not_in_context():
    fn = APP["answer_numbers_not_in_context"]
    context = "Выручка 12500, рост 7.5%, год 2026"
    assert fn("Выручка 12500, рост 7.5%", context) is False
    # 2 из 2 чисел отсутствуют в контексте => True
    assert fn("Выручка 99999, рост 99%", context) is True
