import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# Better number/date detection
_NBSP = "\u00A0"
_NNBSP = "\u202F"

NUM_RE = re.compile(
    rf"""
    (?<![\w])
    [+-]?
    (?:
        \d{{1,3}}(?:[ ,{_NBSP}{_NNBSP}]\d{{3}})+
        |
        \d{{4,}}
        |
        \d{{1,3}}
    )
    (?:[.,]\d+)?
    %?
    (?![\w])
    """,
    re.VERBOSE | re.UNICODE,
)

DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}\b", re.UNICODE)

BAD_PAGE_PAT = re.compile(r"\bстр\.?\s*(xx|\?\?|\?)(?!\d)", re.IGNORECASE)
EXPLICIT_BAD_PAGE = re.compile(r"\(стр\.?\s*(xx|\?\?|\?)\)", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    """
    Создаёт директорию, если она отсутствует.
    """
    os.makedirs(path, exist_ok=True)


def normalize_query(q: str) -> str:
    """
    Нормализует текст запроса (trim + схлопывание пробелов).
    """
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def clamp_text(s: str, n: int) -> str:
    """
    Обрезает текст до заданной длины после нормализации пробелов.
    """
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:n]


def safe_page_range(meta: Dict[str, Any]) -> str:
    """
    Безопасно форматирует диапазон страниц из metadata.
    """
    p1 = meta.get("page_start", meta.get("page", None))
    p2 = meta.get("page_end", meta.get("page", None))
    try:
        if p1 is None:
            return "?"
        if p2 is None:
            return str(int(p1) + 1)
        p1i = int(p1) + 1
        p2i = int(p2) + 1
        return f"{p1i}-{p2i}" if p1i != p2i else f"{p1i}"
    except Exception:
        if p2 is None:
            return str(p1)
        return f"{p1}-{p2}"


def doc_key(d: Any) -> Tuple[Any, Any, Any, Any]:
    """
    Строит уникальный ключ документа для dedup.
    """
    m = getattr(d, "metadata", None) or {}
    return (
        m.get("source", ""),
        m.get("page_start", m.get("page", None)),
        m.get("page_end", m.get("page", None)),
        m.get("chunk_id", None),
    )


def dedup_docs(docs: List[Any], max_total: int = 240) -> List[Any]:
    """
    Удаляет дубликаты документов с сохранением порядка.
    """
    seen = set()
    out: List[Any] = []
    for d in docs:
        k = doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
        if len(out) >= max_total:
            break
    return out


def coverage_tokens(q: str) -> List[str]:
    """
    Извлекает информативные токены вопроса для оценки покрытия.
    """
    q = (q or "").lower()
    words = re.findall(r"[a-zа-яё0-9]+", q, flags=re.IGNORECASE)
    stop = {
        "и", "в", "на", "по", "за", "к", "о", "об", "что", "это", "как", "ли", "или", "для", "про", "из",
        "при", "без", "над", "под", "до", "от", "же", "мы", "вы", "он", "она", "они", "оно", "кто", "где",
    }
    words = [w for w in words if len(w) >= 4 and w not in stop]
    return words[:10]


def word_hit_ratio(tokens: List[str], text: str) -> float:
    """
    Считает долю токенов вопроса, найденных в тексте.
    """
    if not tokens:
        return 0.0
    hits = 0
    for t in tokens:
        if re.search(rf"(?<![a-zа-яё0-9]){re.escape(t)}(?![a-zа-яё0-9])", text, flags=re.IGNORECASE):
            hits += 1
    return hits / max(1, len(tokens))


def diversify_docs(docs: List[Any], max_per_group: int) -> List[Any]:
    """
    Ограничивает число документов из одной группы (source/page/section).
    """
    counts = defaultdict(int)
    out: List[Any] = []
    for d in docs:
        m = getattr(d, "metadata", None) or {}
        key = (m.get("source", ""), safe_page_range(m), m.get("section_title", ""))
        if counts[key] >= max_per_group:
            continue
        counts[key] += 1
        out.append(d)
    return out


def has_number(text: str) -> bool:
    """
    Проверяет наличие числового паттерна в тексте.
    """
    return bool(NUM_RE.search(text or ""))


def fix_broken_numbers(text: str) -> str:
    """
    Склеивает артефакты OCR/переносов в числах и процентах.
    """
    if not text:
        return text
    text = text.replace("\u00A0", " ").replace("\u202F", " ")
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s*[\n\r]+\s*(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s*[\n\r ]+\s*([.,])\s*[\n\r ]+\s*(\d)", r"\1\2\3", text)
    text = re.sub(r"(\d)\s*([.,])\s*(\d)\s*%", r"\1\2\3%", text)
    return text


def contains_fake_pages(s: str) -> bool:
    """
    Проверяет наличие некорректных ссылок на страницы (`стр. XX/??`).
    """
    if not s:
        return False
    return bool(BAD_PAGE_PAT.search(s) or EXPLICIT_BAD_PAGE.search(s))


def extract_numbers_from_text(s: str) -> List[str]:
    """
    Извлекает и нормализует уникальные числовые значения из текста.
    """
    if not s:
        return []
    nums = re.findall(r"(?<!\w)[+-]?\d[\d\s\u00A0\u202F]*(?:[.,]\d+)?%?(?!\w)", s)
    clean = []
    for n in nums:
        n2 = re.sub(r"[\s\u00A0\u202F]+", "", n)
        n2 = n2.replace("−", "-")
        if len(n2) >= 2:
            clean.append(n2)
    seen = set()
    out = []
    for x in clean:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:30]
