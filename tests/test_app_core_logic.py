from dataclasses import dataclass

import pytest

from ragbot.core_logic import (
    _extract_first_json_object,
    add_neighbors_from_parent_map,
    answer_numbers_not_in_context,
    build_extractive_plan,
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
        ("как подключить модуль", "procedure"),
        ("как войти в систему", "procedure"),
        ("как проверить статус", "procedure"),
        ("как открыть форму заявки", "procedure"),
        ("как называется подсистема", "default"),
        ("как устроена подсистема", "default"),
        ("как формулируется миссия", "default"),
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


def test_build_extractive_plan_collects_quotes_and_lexical_coverage():
    context = (
        "[стр. 5] Выручка компании за 2026 год составила 12500 млн руб. Рост к прошлому году — 7.5%."
        "\n\n[стр. 6] EBITDA достигла 2100 млн руб, маржа 16.8%."
    )
    plan = build_extractive_plan("Какая выручка и рост в 2026?", context, intent="numbers_and_dates", max_items=4)
    assert len(plan["evidence"]) >= 1
    assert "стр. 5" in plan["evidence_text"]
    assert "выручка" in plan["lexical_report"]["covered_tokens"]
    assert "2026" in plan["lexical_report"]["covered_numbers"]


def test_build_extractive_plan_synthesis_context_contains_layers():
    context = "[стр. 2] Требуется оформить заявку в течение 3 дней."
    plan = build_extractive_plan("Какие требования и сроки?", context, intent="requirements", max_items=3)
    synth = plan["synthesis_context"]
    assert "EXTRACTIVE-PLAN" in synth
    assert "LEXICAL-CONSTRAINTS" in synth
    assert "HIERARCHICAL-CONTEXT" in synth
    assert "SYNTHESIS-RULES" in synth


def test_build_extractive_plan_builds_hierarchical_context_report():
    context = (
        "[стр. 1] Общая справка по компании и вводная информация."
        "\n\n[стр. 9] Выручка за 2026 год составила 12500 млн руб. Рост 7.5% к 2025 году."
        "\n\n[стр. 11] Приложение и нерелевантные технические детали."
    )
    plan = build_extractive_plan(
        "Какая выручка и рост в 2026?",
        context,
        intent="numbers_and_dates",
        hierarchical_max_sections=1,
        hierarchical_max_sentences_per_section=2,
    )
    hctx = plan["hierarchical_context"]
    assert hctx["report"]["sections_selected"] == 1
    assert "стр. 9" in hctx["context"]


def test_build_extractive_plan_hierarchical_context_fallback_to_raw_blocks():
    context = "[стр. 3] Кратко.\n\n[стр. 4] Ещё текст."
    plan = build_extractive_plan(
        "Что известно?",
        context,
        intent="default",
        hierarchical_max_sections=1,
        hierarchical_max_sentences_per_section=1,
    )
    hctx = plan["hierarchical_context"]
    assert hctx["report"]["fallback_used"] is True
    assert "стр. 3" in hctx["context"] or "стр. 4" in hctx["context"]


def test_build_extractive_plan_filters_noisy_context_with_threshold():
    context = (
        "[стр. 1] Купить крипту сейчас! Лучшие условия и акции каждый день."
        "\n\n[стр. 7] Выручка за 2026 год составила 12500 млн руб., рост 7.5% к 2025 году."
        "\n\n[стр. 8] Реклама, скидки, подписки, бонусы без отношения к вопросу."
    )
    plan = build_extractive_plan(
        "Какая выручка и рост в 2026?",
        context,
        intent="numbers_and_dates",
        max_items=4,
        min_score_threshold=0.2,
    )
    assert len(plan["evidence"]) >= 1
    assert all(item["page"] == "7" for item in plan["evidence"])


def test_build_extractive_plan_limits_evidence_per_page_for_diversity():
    context = (
        "[стр. 10] Шаг 1: подайте заявление в личном кабинете. Шаг 2: приложите документы. Шаг 3: дождитесь подтверждения."
        "\n\n[стр. 11] Шаг 4: получите уведомление о результате и сроки исполнения."
    )
    plan = build_extractive_plan(
        "Опиши процедуру подачи заявления",
        context,
        intent="procedure",
        max_items=5,
        max_per_page=1,
        min_score_threshold=0.1,
    )
    pages = [x["page"] for x in plan["evidence"]]
    assert pages.count("10") <= 1
    assert pages.count("11") <= 1


def test_build_extractive_plan_handles_paraphrase_lexically():
    context = "[стр. 4] Компания подняла выручку до 12 500 млн руб. в 2026 году, прибавив 7.5 процента год к году."
    plan = build_extractive_plan("Сколько составил рост выручки в 2026?", context, intent="numbers_and_dates", max_items=3)
    assert plan["evidence"]
    assert "2026" in plan["lexical_report"]["covered_numbers"]
