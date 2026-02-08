from dataclasses import dataclass

import pytest

from ragbot.core_logic import (
    _extract_first_json_object,
    add_neighbors_from_parent_map,
    answer_numbers_not_in_context,
    coverage_score,
    detect_intent_fast,
    format_context,
    invoke_json_robust,
    is_not_found_answer,
    rerank_numbers_heuristic,
)


@dataclass
class DummyDoc:
    page_content: str
    metadata: dict


def test_extract_first_json_object():
    assert _extract_first_json_object('{"a": 1}') == {"a": 1}
    assert _extract_first_json_object('prefix {"a": 2} suffix') == {"a": 2}
    assert _extract_first_json_object("no json here") is None


def test_invoke_json_robust_uses_json_chain_first():
    class Chain:
        def invoke(self, _):
            return {"ok": True}

    out = invoke_json_robust(Chain(), Chain(), {"x": 1}, {"ok": False}, retries=1)
    assert out == {"ok": True}


def test_invoke_json_robust_falls_back_to_string_chain():
    class Broken:
        def invoke(self, _):
            raise RuntimeError("boom")

    class StringChain:
        def invoke(self, _):
            return 'text {"intent": "default", "confidence": 0.5}'

    out = invoke_json_robust(Broken(), StringChain(), {"x": 1}, {"intent": "fallback"}, retries=1)
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
    intent, _ = detect_intent_fast(q)
    assert intent == expected


def test_coverage_score_numbers_boost():
    docs = [DummyDoc("Выручка 12 500 и рост 7%", {"page": 0})]
    score = coverage_score("сколько рост", docs, "numbers_and_dates")
    assert score > 0.4


def test_format_context_limits_and_truncates():
    docs = [
        DummyDoc("A" * 300, {"page": 0, "section_title": "s1"}),
        DummyDoc("B" * 300, {"page": 1, "section_title": "s2"}),
    ]
    out = format_context(docs, limit=2, max_chars=350)
    assert "[стр. 1 | s1]" in out
    assert len(out) <= 350


def test_is_not_found_answer():
    assert is_not_found_answer("В документе не найдено") is True
    assert is_not_found_answer("не найдено.") is True
    assert is_not_found_answer("В документе не найдено по этому вопросу") is True
    assert is_not_found_answer("Нашёл ответ на стр. 10") is False


def test_add_neighbors_from_parent_map_adds_window_chunks():
    base = DummyDoc("c2", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 2})
    n1 = DummyDoc("c1", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 1})
    n3 = DummyDoc("c3", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 3})
    far = DummyDoc("c9", {"source": "s", "page": 0, "section_title": "x", "chunk_id": 9})

    parent_key = ("s", 0, 0, "x")
    parent_map = {parent_key: [n1, base, n3, far]}

    out = add_neighbors_from_parent_map(parent_map, [base], window=1, max_total=10)
    contents = [d.page_content for d in out]
    assert "c1" in contents and "c3" in contents
    assert "c9" not in contents


def test_rerank_numbers_heuristic_prefers_relevant_docs():
    docs = [
        DummyDoc("просто текст", {"page": 0}),
        DummyDoc("рост 7% в 2026 году", {"page": 1}),
        DummyDoc("метрика 123", {"page": 2}),
    ]
    out = rerank_numbers_heuristic("сколько рост процентов 2026", docs, keep=2)
    assert out[0].page_content == "рост 7% в 2026 году"


def test_answer_numbers_not_in_context():
    context = "Выручка 12500, рост 7.5%, год 2026"
    assert answer_numbers_not_in_context("Выручка 12500, рост 7.5%", context) is False
    # 2 из 2 чисел отсутствуют в контексте => True
    assert answer_numbers_not_in_context("Выручка 99999, рост 99%", context) is True
