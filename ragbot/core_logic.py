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
        ("как " in q and any(v in q for v in ["сдел", "выполн", "оформ", "подат", "заполн", "пройти", "настро", "созда", "получ"]))
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
    if not ans:
        return True
    s = ans.strip().lower()
    if _NOT_FOUND_PAT.match(s):
        return True
    return "в документе не найдено" in s and len(s) < 140


def add_neighbors_from_parent_map(parent_map: dict, docs: List[Any], window: int, max_total: int = 320) -> List[Any]:
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
    q = (question or "").lower()
    text = (doc.page_content or "").lower()
    toks = coverage_tokens(q)
    hit = word_hit_ratio(toks, text)
    has_num = 1.0 if has_number(text) else 0.0
    has_pct = 1.0 if "%" in text or "процент" in text else 0.0
    has_date = 1.0 if DATE_RE.search(text) else 0.0
    return 0.55 * hit + 0.20 * has_num + 0.15 * has_pct + 0.10 * has_date


def rerank_numbers_heuristic(question: str, docs: List[Any], keep: int) -> List[Any]:
    if not docs:
        return docs
    scored = [(_num_rerank_score(question, d), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep]]


def answer_numbers_not_in_context(answer: str, context: str) -> bool:
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
