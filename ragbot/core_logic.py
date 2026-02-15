import json
import re
from typing import Any, Dict, List, Optional, Tuple

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


INTENTS = {"definition", "procedure", "requirements", "numbers_and_dates", "compare", "summary", "citation_only", "default"}

_CITATION_ONLY_STRONG = [
    "только цитаты", "только цитат",
    "без пересказа", "без объяснений",
    "дословно", "дословная",
    "приведи цитату", "приведи цитаты",
    "процитируй", "цитируй",
    "покажи фрагмент", "покажи отрывок",
    "quote only", "verbatim",
]

_NOT_FOUND_PAT = re.compile(r"^\s*(в документе не найдено|не найдено)\s*\.?\s*$", re.IGNORECASE)


def coverage_score(question: str, docs: List[Any], intent: str) -> float:
    """
    Функция `coverage_score` модуля `core_logic`.
    """
    if not docs:
        return 0.0
    toks = coverage_tokens(question)
    text = " ".join((d.page_content or "")[:900] for d in docs[:6])
    token_part = word_hit_ratio(toks, text)
    if intent == "numbers_and_dates":
        num_part = 1.0 if has_number(text) or ("%" in text) else 0.0
        return 0.55 * token_part + 0.45 * num_part
    return token_part


def _extract_first_json_object(s: str) -> Optional[dict]:
    """
    Внутренний helper `_extract_first_json_object` для инкапсуляции локальной логики модуля.
    """
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def invoke_json_robust(chain_json, chain_str, inputs: dict, fallback: dict, retries: int = 1) -> dict:
    """
    Функция `invoke_json_robust` модуля `core_logic`.
    """
    for _ in range(max(1, retries + 1)):
        try:
            out = chain_json.invoke(inputs)
            if isinstance(out, dict):
                return out
        except Exception:
            try:
                raw = chain_str.invoke(inputs)
                if isinstance(raw, str):
                    parsed = _extract_first_json_object(raw)
                    if isinstance(parsed, dict):
                        return parsed
            except Exception:
                pass
    return dict(fallback)


def detect_intent_fast(user_question: str):
    """
    Определяет тип/структуру входных данных по эвристическим правилам.
    """
    q = (user_question or "").lower()
    if any(k in q for k in _CITATION_ONLY_STRONG):
        return "citation_only", 0.90
    if any(k in q for k in ["сводк", "резюм", "кратко", "обзор", "выжимк", "самое важное", "коротко: о чём"]):
        return "summary", 0.85
    if any(k in q for k in ["сравни", "сравнить", "отлич", "разниц", "vs", "против"]):
        return "compare", 0.80
    if any(k in q for k in ["требован", "обязан", "должен", "услови", "необходимо", "запрещ", "разреш", "стандарт", "регламент"]):
        return "requirements", 0.78
    how_to_procedure = (
        (
            "как " in q
            and any(
                v in q
                for v in [
                    "сдел",
                    "выполн",
                    "оформ",
                    "подат",
                    "заполн",
                    "пройти",
                    "настро",
                    "созда",
                    "получ",
                    "подключ",
                    "откры",
                    "войти",
                    "провер",
                    "включ",
                    "отключ",
                    "измен",
                ]
            )
        )
        or any(k in q for k in ["шаг", "процедур", "инструкц", "порядок", "алгоритм"])
    )
    if how_to_procedure:
        return "procedure", 0.70
    if any(k in q for k in ["что такое", "определени", "термин", "означает"]):
        return "definition", 0.70
    if re.search(r"\d", q) or any(k in q for k in ["сколько", "когда", "дата", "процент", "сумм", "руб", "млн", "млрд", "показател", "метрик", "доля", "mau", "dau"]):
        return "numbers_and_dates", 0.72
    return "default", 0.55


def format_context(docs: List[Any], limit: int, max_chars: int) -> str:
    """
    Форматирует данные в текст, пригодный для ответа пользователю или LLM-контекста.
    """
    out = []
    total = 0
    used = 0
    for d in docs:
        if used >= limit:
            break
        pages = safe_page_range(d.metadata)
        title = (d.metadata or {}).get("section_title", "")
        title_part = f" | {title}" if title else ""
        block = f"[стр. {pages}{title_part}] {d.page_content}".strip()
        if not block:
            continue
        if total + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - total - 2)
            if remaining > 200:
                out.append(block[:remaining])
            break
        out.append(block)
        total += len(block) + 2
        used += 1
    return "\n\n".join(out)


def is_not_found_answer(ans: str) -> bool:
    """
    Проверяет условие/признак и возвращает булево значение.
    """
    if not ans:
        return True
    s = ans.strip().lower()
    if _NOT_FOUND_PAT.match(s):
        return True
    return "в документе не найдено" in s and len(s) < 140


def add_neighbors_from_parent_map(parent_map: dict, docs: List[Any], window: int, max_total: int = 320) -> List[Any]:
    """
    Функция `add_neighbors_from_parent_map` модуля `core_logic`.
    """
    if not docs or window <= 0:
        return docs
    out = list(docs)
    seen = set(doc_key(d) for d in out)

    for d in docs:
        m = d.metadata or {}
        parent_key = (
            m.get("source", ""),
            m.get("page_start", m.get("page", None)),
            m.get("page_end", m.get("page", None)),
            m.get("section_title", ""),
        )
        cid = m.get("chunk_id", None)
        if cid is None:
            continue
        try:
            cid = int(cid)
        except Exception:
            continue

        pool = parent_map.get(parent_key) or []
        if not pool:
            continue

        for nd in pool:
            nm = nd.metadata or {}
            try:
                ncid = int(nm.get("chunk_id", -999999))
            except Exception:
                continue
            if abs(ncid - cid) <= window:
                k = doc_key(nd)
                if k not in seen:
                    out.append(nd)
                    seen.add(k)
                    if len(out) >= max_total:
                        return out
    return out


def _num_rerank_score(question: str, doc: Any) -> float:
    """
    Внутренний helper `_num_rerank_score` для инкапсуляции локальной логики модуля.
    """
    q = (question or "").lower()
    text = (doc.page_content or "").lower()
    toks = coverage_tokens(q)
    hit = word_hit_ratio(toks, text)
    has_num = 1.0 if has_number(text) else 0.0
    has_pct = 1.0 if "%" in text or "процент" in text else 0.0
    has_date = 1.0 if DATE_RE.search(text) else 0.0
    return 0.55 * hit + 0.20 * has_num + 0.15 * has_pct + 0.10 * has_date


def rerank_numbers_heuristic(question: str, docs: List[Any], keep: int) -> List[Any]:
    """
    Функция `rerank_numbers_heuristic` модуля `core_logic`.
    """
    if not docs:
        return docs
    scored = [(_num_rerank_score(question, d), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep]]


def answer_numbers_not_in_context(answer: str, context: str) -> bool:
    """
    Функция `answer_numbers_not_in_context` модуля `core_logic`.
    """
    if not answer or not context:
        return False
    ans_nums = extract_numbers_from_text(answer)
    if not ans_nums:
        return False
    ctx_norm = re.sub(r"[\s\u00A0\u202F]+", "", context).replace("−", "-")
    misses = 0
    checks = 0
    for n in ans_nums:
        if len(re.sub(r"\D", "", n)) <= 1:
            continue
        checks += 1
        if n not in ctx_norm:
            misses += 1
    return checks >= 2 and (misses / max(1, checks)) >= 0.5


def build_extractive_evidence(question: str, context: str, max_items: int = 6) -> str:
    """
    Собирает и возвращает компонент/цепочку `extractive_evidence` для этапов RAG-пайплайна.
    """
    if not context:
        return ""
    q_toks = coverage_tokens(question)
    blocks = [b.strip() for b in (context or "").split("\n\n") if b.strip()]
    cand = []
    for b in blocks:
        m = re.match(r"^\[стр\.\s*([^\]]+)\]\s*(.*)$", b, flags=re.IGNORECASE | re.DOTALL)
        page = m.group(1).strip() if m else "?"
        text = (m.group(2) if m else b).strip()
        parts = re.split(r"(?<=[\.!?])\s+", text)
        for s in parts:
            s = s.strip().strip('"')
            if len(s) < 40:
                continue
            hit = word_hit_ratio(q_toks, s)
            numeric = 0.15 if has_number(s) else 0.0
            score = hit + numeric
            if score <= 0.05:
                continue
            cand.append((score, page, s))

    if not cand:
        return ""
    cand.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, page, s in cand:
        key = re.sub(r"\W+", "", s.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        clip = clamp_text(s, 220)
        out.append(f"- (стр. {page}) \"{clip}\"")
        if len(out) >= max_items:
            break
    return "\n".join(out)


def _context_blocks(context: str) -> List[Tuple[str, str]]:
    """
    Внутренний helper `_context_blocks` для инкапсуляции локальной логики модуля.
    """
    out: List[Tuple[str, str]] = []
    for b in [x.strip() for x in (context or "").split("\n\n") if x.strip()]:
        m = re.match(r"^\[стр\.\s*([^\]]+)\]\s*(.*)$", b, flags=re.IGNORECASE | re.DOTALL)
        page = (m.group(1).strip() if m else "?")
        text = (m.group(2) if m else b).strip()
        if text:
            out.append((page, text))
    return out


def _question_lexical_constraints(question: str) -> Dict[str, List[str]]:
    """
    Внутренний helper `_question_lexical_constraints` для инкапсуляции локальной логики модуля.
    """
    tokens = coverage_tokens(question)
    numbers = extract_numbers_from_text(question)
    return {
        "tokens": tokens[:8],
        "numbers": numbers[:6],
    }


def _intent_phrase_bonus(intent: str, sentence: str) -> float:
    """
    Внутренний helper `_intent_phrase_bonus` для инкапсуляции локальной логики модуля.
    """
    if intent == "requirements" and re.search(r"долж|обязан|необходимо|запрещ", sentence, flags=re.IGNORECASE):
        return 0.12
    if intent == "procedure" and re.search(r"шаг|этап|сначала|далее|затем", sentence, flags=re.IGNORECASE):
        return 0.12
    if intent == "compare" and re.search(r"в отличие|по сравнению|против|больше|меньше", sentence, flags=re.IGNORECASE):
        return 0.08
    return 0.0


def _score_sentence(question_tokens: List[str], sentence: str, intent: str) -> float:
    """
    Внутренний helper `_score_sentence` для инкапсуляции локальной логики модуля.
    """
    hit = word_hit_ratio(question_tokens, sentence)
    numeric = 0.16 if has_number(sentence) else 0.0
    date_bonus = 0.08 if DATE_RE.search(sentence) else 0.0
    intent_bonus = _intent_phrase_bonus(intent, sentence)
    return hit + numeric + date_bonus + intent_bonus


def _build_hierarchical_context(
    question: str,
    context: str,
    intent: str,
    max_sections: int,
    max_sentences_per_section: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Внутренний helper `_build_hierarchical_context` для инкапсуляции локальной логики модуля.
    """
    blocks = _context_blocks(context)
    q_toks = coverage_tokens(question)
    section_scored: List[Tuple[float, str, List[Tuple[float, str]]]] = []

    for page, text in blocks:
        parts = [p.strip().strip('"') for p in re.split(r"(?<=[\.!?])\s+", text) if p.strip()]
        scored_parts: List[Tuple[float, str]] = []
        for s in parts:
            if len(s) < 30:
                continue
            scored_parts.append((_score_sentence(q_toks, s, intent), s))

        if not scored_parts:
            continue
        scored_parts.sort(key=lambda x: x[0], reverse=True)
        top = scored_parts[: max(1, max_sentences_per_section)]
        section_score = sum(x[0] for x in top) / len(top)
        section_scored.append((section_score, page, top))

    section_scored.sort(key=lambda x: x[0], reverse=True)
    chosen = section_scored[: max(1, max_sections)]

    packed_blocks = []
    selected_pages = []
    for _, page, sents in chosen:
        selected_pages.append(page)
        joined = " ".join([clamp_text(s, 260) for _, s in sents])
        packed_blocks.append(f"[стр. {page}] {joined}".strip())

    fallback_used = False
    if not packed_blocks and blocks:
        fallback_used = True
        for page, text in blocks[: max(1, max_sections)]:
            selected_pages.append(page)
            packed_blocks.append(f"[стр. {page}] {clamp_text(text, 420)}")

    packed_context = "\n\n".join(packed_blocks)
    report = {
        "selected_pages": selected_pages,
        "sections_total": len(section_scored),
        "sections_selected": len(chosen),
        "max_sections": int(max_sections),
        "max_sentences_per_section": int(max_sentences_per_section),
        "fallback_used": fallback_used,
    }
    return packed_context, report


def build_extractive_plan(
    question: str,
    context: str,
    intent: str,
    max_items: int = 8,
    min_score_threshold: float = 0.12,
    max_per_page: int = 2,
    hierarchical_max_sections: int = 6,
    hierarchical_max_sentences_per_section: int = 3,
) -> Dict[str, Any]:
    """
    Собирает и возвращает компонент/цепочку `extractive_plan` для этапов RAG-пайплайна.
    """
    blocks = _context_blocks(context)
    q_toks = coverage_tokens(question)
    constraints = _question_lexical_constraints(question)
    candidates: List[Tuple[float, str, str]] = []

    for page, text in blocks:
        parts = re.split(r"(?<=[\.!?])\s+", text)
        for sent in parts:
            s = sent.strip().strip('"')
            if len(s) < 30:
                continue
            score = _score_sentence(q_toks, s, intent)
            if score < float(min_score_threshold):
                continue
            candidates.append((score, page, s))

    candidates.sort(key=lambda x: x[0], reverse=True)
    evidence: List[Dict[str, str]] = []
    seen = set()
    per_page_count: Dict[str, int] = {}
    for _, page, s in candidates:
        key = re.sub(r"\W+", "", s.lower())[:120]
        if key in seen:
            continue
        if per_page_count.get(page, 0) >= max(1, int(max_per_page)):
            continue
        seen.add(key)
        evidence.append({"page": page, "quote": clamp_text(s, 220)})
        per_page_count[page] = per_page_count.get(page, 0) + 1
        if len(evidence) >= max_items:
            break

    evidence_text = "\n".join([f"- (стр. {x['page']}) \"{x['quote']}\"" for x in evidence])

    covered_tokens = [t for t in constraints["tokens"] if evidence_text and re.search(rf"(?<![a-zа-яё0-9]){re.escape(t)}(?![a-zа-яё0-9])", evidence_text, flags=re.IGNORECASE)]
    missing_tokens = [t for t in constraints["tokens"] if t not in covered_tokens]

    flattened_evidence = re.sub(r"[\s\u00A0\u202F]+", "", evidence_text).replace("−", "-")
    covered_numbers = [n for n in constraints["numbers"] if n in flattened_evidence]
    missing_numbers = [n for n in constraints["numbers"] if n not in covered_numbers]

    lexical_report = {
        "required_tokens": constraints["tokens"],
        "covered_tokens": covered_tokens,
        "missing_tokens": missing_tokens,
        "required_numbers": constraints["numbers"],
        "covered_numbers": covered_numbers,
        "missing_numbers": missing_numbers,
    }

    packed_context, hierarchy_report = _build_hierarchical_context(
        question,
        context,
        intent,
        max_sections=int(hierarchical_max_sections),
        max_sentences_per_section=int(hierarchical_max_sentences_per_section),
    )

    synthesis_context = (
        "EXTRACTIVE-PLAN (используй как первичную опору):\n"
        + (evidence_text or "- нет уверенных извлечений")
        + "\n\nLEXICAL-CONSTRAINTS:\n"
        + f"- термины из вопроса: {', '.join(constraints['tokens']) or 'нет'}\n"
        + f"- числа/даты из вопроса: {', '.join(constraints['numbers']) or 'нет'}\n"
        + f"- непокрытые термины: {', '.join(missing_tokens) or 'нет'}\n"
        + f"- непокрытые числа: {', '.join(missing_numbers) or 'нет'}\n\n"
        + "HIERARCHICAL-CONTEXT:\n"
        + (packed_context or "- не удалось собрать сокращенный контекст")
        + "\n\nSYNTHESIS-RULES:\n"
        + "1) Сначала опирайся на EXTRACTIVE-PLAN, не придумывай новые факты.\n"
        + "2) Если элемент из constraints не найден в extractive-фрагментах, явно пометь как не найдено.\n"
        + "3) При конфликте между HIERARCHICAL-CONTEXT и RAW-CONTEXT приоритет у EXTRACTIVE-PLAN.\n"
        + "4) Добавляй ссылки на страницы только из extractive-фрагментов или исходного контекста.\n\n"
        + "RAW-CONTEXT:\n"
        + context
    )

    return {
        "evidence": evidence,
        "evidence_text": evidence_text,
        "lexical_report": lexical_report,
        "guardrails": {
            "min_score_threshold": float(min_score_threshold),
            "max_per_page": int(max_per_page),
            "per_page_count": per_page_count,
        },
        "hierarchical_context": {
            "context": packed_context,
            "report": hierarchy_report,
        },
        "synthesis_context": synthesis_context,
    }
