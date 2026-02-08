# app.py
# Single-file TG bot + RAG agent (VM-ready)
# - Polling bot (aiogram v3)
# - Per-user PDF indexing (FAISS + BM25)
# - /reset clears user session (index)
# - Secrets: env BOT_TOKEN + GIGACHAT_API_KEY (preferred),
#   fallback: Google Secret Manager (optional)

import os
import re
import time
import json
import gzip
import shutil
import asyncio
import logging
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("tg-rag-bot")


# -----------------------------
# Secrets (env first, GSM fallback)
# -----------------------------
def get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v:
        v = v.strip()
    return v or None

def load_secret_from_gsm(secret_id: str, project_id: Optional[str] = None) -> Optional[str]:
    """
    Optional fallback: read secret from Google Secret Manager.
    Requires:
      pip install google-cloud-secret-manager
    And VM SA must have roles/secretmanager.secretAccessor.
    """
    try:
        from google.cloud import secretmanager  # type: ignore
    except Exception:
        return None

    project_id = project_id or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8").strip()

def must_get_secret(name: str) -> str:
    # Prefer env
    v = get_env(name)
    if v:
        return v

    # Fallback GSM with same secret name
    v = load_secret_from_gsm(name)
    if v:
        return v

    raise RuntimeError(
        f"Missing secret {name}. Provide env var {name} (preferred) or enable GSM fallback."
    )


# -----------------------------
# Agent code (your version, only paths made session-aware)
# -----------------------------
AGENT_VERSION = "v_final_patched_qos_1_4_plus_vm"

# IMPORTANT: install pymupdf (not the wrong 'fitz' package)
# pip install pymupdf
import pymupdf as fitz  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402

from langchain_gigachat.embeddings import GigaChatEmbeddings  # noqa: E402
try:
    from langchain_gigachat.chat_models import GigaChat  # noqa: E402
except Exception:
    from langchain_gigachat import GigaChat  # type: ignore # noqa: E402

from ragbot.text_utils import (  # noqa: E402
    DATE_RE,
    clamp_text,
    contains_fake_pages,
    coverage_tokens,
    dedup_docs,
    diversify_docs,
    doc_key,
    ensure_dir,
    extract_numbers_from_text,
    fix_broken_numbers,
    has_number,
    normalize_query,
    safe_page_range,
    word_hit_ratio as _word_hit_ratio,
)


@dataclass(frozen=True)
class Settings:
    # Base data dir (mounted volume in docker)
    data_dir: str = "./data"

    # chunking
    chunk_size: int = 1100
    chunk_overlap: int = 150

    # retrieval (base)
    bm25_k: int = 14
    faiss_k: int = 14

    # multiquery
    multiquery_enabled: bool = True
    multiquery_n: int = 2
    multiquery_parallel: bool = True
    multiquery_threads: int = 6
    multiquery_k_per_query: int = 12

    # separate multiquery count for numbers/dates
    numbers_multiquery_n: int = 4

    # rerank
    rerank_enabled: bool = True
    rerank_pool: int = 18
    rerank_keep: int = 10

    # context
    final_docs_limit: int = 10
    max_context_chars: int = 16000
    diversify_max_per_group: int = 3

    # intent
    llm_intent_threshold: float = 0.78

    # llm
    temperature: float = 0.2
    top_p: float = 0.9
    model: str = "GigaChat-2-Max"

    # embeddings
    embeddings_model: str = "GigaEmbeddings-3B-2025-09"
    query_embedding_cache_size: int = 512

    # safety / compatibility
    allow_dangerous_faiss_deserialization: bool = False
    allow_index_settings_mismatch: bool = False

    # ---- Quality-over-speed knobs (universal) ----
    numbers_k_multiplier: int = 3
    numbers_neighbors_window: int = 2
    numbers_disable_diversify: bool = True

    # Second pass
    second_pass_enabled: bool = True
    second_pass_multiquery_n: int = 6
    second_pass_k_multiplier: int = 5


def coverage_score(question: str, docs: List[Document], intent: str) -> float:
    if not docs:
        return 0.0
    toks = coverage_tokens(question)
    text = " ".join((d.page_content or "")[:900] for d in docs[:6])
    token_part = _word_hit_ratio(toks, text)
    if intent == "numbers_and_dates":
        num_part = 1.0 if has_number(text) or ("%" in text) else 0.0
        return 0.55 * token_part + 0.45 * num_part
    return token_part

# Robust JSON parsing for LLM outputs
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

# LLM / Embeddings
def build_llm(settings: Settings):
    key = os.environ.get("GIGA_API_KEY") or os.environ.get("GIGACHAT_API_KEY")
    if not key:
        raise RuntimeError("Не найден env GIGA_API_KEY / GIGACHAT_API_KEY")
    return GigaChat(
        credentials=key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_B2B",
        model=settings.model,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )

class CachedEmbeddings(Embeddings):
    def __init__(self, base: Embeddings, max_size: int = 512):
        self.base = base
        self.max_size = max_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()

    def embed_query(self, text: str) -> List[float]:
        key = normalize_query(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        vec = self.base.embed_query(text)
        self._cache[key] = vec
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base.embed_documents(texts)

def build_embeddings(settings: Settings) -> Embeddings:
    giga_api_key = os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGA_API_KEY")
    if not giga_api_key:
        raise RuntimeError("Не найден ключ. Установи env GIGACHAT_API_KEY (или GIGA_API_KEY).")
    base = GigaChatEmbeddings(
        credentials=giga_api_key,
        scope="GIGACHAT_API_B2B",
        verify_ssl_certs=False,
        model=settings.embeddings_model,
    )
    return CachedEmbeddings(base, max_size=settings.query_embedding_cache_size)

# PDF loader + light numeric glue
def load_pdf_pages(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    with fitz.open(pdf_path) as pdf:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)

            flags = 0
            if hasattr(fitz, "TEXT_PRESERVE_WHITESPACE"):
                flags |= fitz.TEXT_PRESERVE_WHITESPACE
            if hasattr(fitz, "TEXT_DEHYPHENATE"):
                flags |= fitz.TEXT_DEHYPHENATE

            text = page.get_text("text", sort=True, flags=flags) or ""
            if not text.strip():
                blocks = page.get_text("blocks") or []
                text = "\n".join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])

            text = fix_broken_numbers(text)
            docs.append(Document(page_content=text, metadata={"source": pdf_path, "page": i}))
    return docs

# Section grouping + chunking
_HEADING_PATTERNS = [
    re.compile(r"^\s*(\d{1,2}(\.\d{1,2}){0,4})\s+([^\n]{3,120})\s*$", re.UNICODE),
    re.compile(r"^\s*([A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\s\-\–—,\.]{6,120})\s*$", re.UNICODE),
    re.compile(r"^\s*(СОДЕРЖАНИЕ|ОТЧЕТ|ОТЧЁТ|ОБЗОР|ПРИЛОЖЕНИЕ|ПРИЛОЖЕНИЯ)\s*$", re.IGNORECASE | re.UNICODE),
]
_HEADING_IGNORE = [
    re.compile(r"^\s*(ТАБЛИЦА|РИС\.?|ПРИМЕЧАНИЕ|NOTE)\b", re.IGNORECASE | re.UNICODE),
]

def detect_heading_on_page(page_text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
    for ln in lines[:12]:
        if len(ln) < 6:
            continue
        if re.fullmatch(r"[\d\W_]{3,}", ln):
            continue
        if any(pat.search(ln) for pat in _HEADING_IGNORE):
            continue
        for pat in _HEADING_PATTERNS:
            if pat.match(ln):
                cand = ln.strip("•-—– \t")
                if 6 <= len(cand) <= 120:
                    return cand
    return None

def build_parent_docs_section_aware(pages: List[Document]) -> List[Document]:
    parents: List[Document] = []
    cur_pages: List[Document] = []
    cur_title: str = "Без заголовка"
    last_title: Optional[str] = None

    def flush():
        nonlocal cur_pages, cur_title
        if not cur_pages:
            return
        content = "\n\n".join([p.page_content for p in cur_pages]).strip()
        if not content:
            cur_pages = []
            return
        start_page = cur_pages[0].metadata.get("page", 0)
        end_page = cur_pages[-1].metadata.get("page", 0)
        meta = {
            "source": cur_pages[0].metadata.get("source", ""),
            "page_start": start_page,
            "page_end": end_page,
            "section_title": cur_title,
        }
        parents.append(Document(page_content=content, metadata=meta))
        cur_pages = []

    for p in pages:
        txt = p.page_content or ""
        title = detect_heading_on_page(txt)
        if title and title != last_title and cur_pages:
            flush()
            cur_title = title
        cur_pages.append(p)
        if title:
            cur_title = title
            last_title = title
    flush()
    return parents

def chunk_parent_docs(parent_docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: List[Document] = []
    for pd in parent_docs:
        parts = splitter.split_text(pd.page_content or "")
        total = len(parts)
        for idx, part in enumerate(parts):
            meta = dict(pd.metadata)
            meta["chunk_id"] = idx
            meta["chunk_total"] = total
            chunks.append(Document(page_content=part, metadata=meta))
    return chunks

# Persist chunks/meta (no pickle)
def _write_chunks_jsonl_gz(path: str, docs: List[Document]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            rec = {"page_content": d.page_content, "metadata": d.metadata or {}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _read_chunks_jsonl_gz(path: str) -> List[Document]:
    out: List[Document] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(Document(page_content=rec.get("page_content",""), metadata=rec.get("metadata", {}) or {}))
    return out

def _check_settings_compat(stored: dict, current: Settings) -> None:
    if current.allow_index_settings_mismatch:
        return
    stored_s = (stored or {}).get("settings") or {}
    keys = ["chunk_size", "chunk_overlap", "embeddings_model", "model"]
    mism = []
    for k in keys:
        if k in stored_s and getattr(current, k, None) != stored_s.get(k):
            mism.append((k, stored_s.get(k), getattr(current, k, None)))
    if mism:
        msg = "Несовпадение настроек индекса и текущих Settings:\n" + "\n".join(
            [f"- {k}: index={a} vs current={b}" for k, a, b in mism]
        ) + "\nПересобери индекс (ingest) или установи allow_index_settings_mismatch=True."
        raise RuntimeError(msg)

def ingest_pdf(pdf_path: str, settings: Settings, index_dir: str) -> None:
    faiss_dir = os.path.join(index_dir, "faiss")
    meta_json = os.path.join(index_dir, "meta.json")
    chunks_jsonl_gz = os.path.join(index_dir, "chunks.jsonl.gz")
    pdf_path_txt = os.path.join(index_dir, "pdf_path.txt")

    ensure_dir(index_dir)
    ensure_dir(faiss_dir)

    embeddings = build_embeddings(settings)

    logger.info(f"[ingest] Loading PDF pages: {pdf_path}")
    pages = load_pdf_pages(pdf_path)
    logger.info(f"[ingest] Pages loaded: {len(pages)}")

    logger.info("[ingest] Building section-aware parent docs...")
    parent_docs = build_parent_docs_section_aware(pages)
    logger.info(f"[ingest] Parent docs: {len(parent_docs)}")

    logger.info("[ingest] Chunking parents...")
    chunk_docs = chunk_parent_docs(parent_docs, settings)
    logger.info(f"[ingest] Chunk docs: {len(chunk_docs)}")

    logger.info("[ingest] Building FAISS...")
    vectorstore = FAISS.from_documents(chunk_docs, embeddings)
    vectorstore.save_local(faiss_dir)

    logger.info("[ingest] Writing chunks.jsonl.gz + meta.json ...")
    _write_chunks_jsonl_gz(chunks_jsonl_gz, chunk_docs)

    meta = {
        "agent_version": AGENT_VERSION,
        "settings": dict(settings.__dict__),
        "chunk_count": len(chunk_docs),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(pdf_path_txt, "w", encoding="utf-8") as f:
        f.write(pdf_path)

    logger.info("[ingest] Done ✅")

def load_index(settings: Settings, index_dir: str):
    faiss_dir = os.path.join(index_dir, "faiss")
    meta_json = os.path.join(index_dir, "meta.json")
    chunks_jsonl_gz = os.path.join(index_dir, "chunks.jsonl.gz")

    if not os.path.exists(meta_json) or not os.path.exists(chunks_jsonl_gz):
        raise RuntimeError("Индекс не найден. Сначала надо ingest_pdf(...)")

    llm = build_llm(settings)
    embeddings = build_embeddings(settings)

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _check_settings_compat(meta, settings)

    allow_danger = bool(settings.allow_dangerous_faiss_deserialization) or (
        os.getenv("ALLOW_DANGEROUS_FAISS_DESERIALIZATION", "").strip() == "1"
    )
    try:
        vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=False)
    except Exception:
        if not allow_danger:
            raise RuntimeError(
                "FAISS.load_local не удалось без dangerous deserialization. "
                "Если доверяешь файлам индекса — включи allow_dangerous_faiss_deserialization=True "
                "или env ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1."
            )
        vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

    chunk_docs: List[Document] = _read_chunks_jsonl_gz(chunks_jsonl_gz)
    if not chunk_docs:
        raise RuntimeError("chunks.jsonl.gz пуст. Удали индекс и сделай ingest заново.")

    bm25 = BM25Retriever.from_documents(chunk_docs)
    bm25.k = max(settings.bm25_k, settings.multiquery_k_per_query, 10)
    return llm, vectorstore, bm25, chunk_docs

# Intent router
INTENTS = {"definition","procedure","requirements","numbers_and_dates","compare","summary","citation_only","default"}
_CITATION_ONLY_STRONG = [
    "только цитаты", "только цитат",
    "без пересказа", "без объяснений",
    "дословно", "дословная",
    "приведи цитату", "приведи цитаты",
    "процитируй", "цитируй",
    "покажи фрагмент", "покажи отрывок",
    "quote only", "verbatim",
]

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
    if any(k in q for k in ["как ", "шаг", "процедур", "инструкц", "порядок", "алгоритм"]):
        return "procedure", 0.70
    if any(k in q for k in ["что такое", "определени", "термин", "означает"]):
        return "definition", 0.70
    if re.search(r"\d", q) or any(k in q for k in ["сколько","когда","дата","процент","сумм","руб","млн","млрд","показател","метрик","доля","mau","dau"]):
        return "numbers_and_dates", 0.72
    return "default", 0.55

def build_intent_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Верни ТОЛЬКО JSON (без markdown, без текста вокруг) в формате:\n"
         "{\"intent\":\"definition|procedure|requirements|numbers_and_dates|compare|summary|citation_only|default\","
         "\"confidence\":0.0,"
         "\"query\":\"...\"}"
        ),
        ("human", "{question}")
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()

def get_intent_hybrid(intent_chain_json, intent_chain_str, user_question: str, threshold: float):
    intent_h, conf_h = detect_intent_fast(user_question)
    q = (user_question or "").strip()
    if conf_h >= threshold and len(q) >= 18:
        return {"intent": intent_h, "confidence": conf_h, "query": q}

    fallback = {"intent": intent_h, "confidence": conf_h, "query": q}
    out = invoke_json_robust(intent_chain_json, intent_chain_str, {"question": q}, fallback, retries=1)

    intent = out.get("intent", intent_h)
    if intent not in INTENTS:
        intent = intent_h
    try:
        conf = float(out.get("confidence", conf_h))
    except Exception:
        conf = conf_h
    query = (out.get("query") or q).strip() or q
    return {"intent": intent, "confidence": conf, "query": query}

# Multiquery + rerank
def build_multiquery_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Сгенерируй {n} коротких переформулировок запроса для поиска по документу.\n"
         "Верни ТОЛЬКО JSON: {\"queries\": [\"...\", \"...\"]}"
        ),
        ("human", "{question}")
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()

def build_rerank_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — reranker. Верни ТОЛЬКО JSON: {\"ranked_ids\": [0,1,2]}"),
        ("human", "Вопрос: {question}\n\nФрагменты:\n{items}")
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()

def rerank_docs(rerank_chain_json, rerank_chain_str, question: str, docs: List[Document], keep: int, pool: int) -> List[Document]:
    if not docs:
        return docs
    pool_docs = docs[:pool]
    items = []
    for i, d in enumerate(pool_docs):
        pages = safe_page_range(d.metadata)
        title = (d.metadata or {}).get("section_title", "Без заголовка")
        snippet = clamp_text(d.page_content, 340)
        items.append(f"- id:{i} | стр:{pages} | раздел:{title} | текст:{snippet}")

    fallback = {"ranked_ids": list(range(min(len(pool_docs), keep)))}
    out = invoke_json_robust(
        rerank_chain_json, rerank_chain_str,
        {"question": question, "items": "\n".join(items)},
        fallback, retries=1
    )

    ranked_ids = []
    for x in out.get("ranked_ids", []):
        try:
            ranked_ids.append(int(x))
        except Exception:
            pass

    chosen: List[int] = []
    for rid in ranked_ids:
        if 0 <= rid < len(pool_docs) and rid not in chosen:
            chosen.append(rid)
        if len(chosen) >= keep:
            break
    if not chosen:
        return docs[:keep]

    reranked = [pool_docs[rid] for rid in chosen]
    for d in pool_docs:
        if d not in reranked:
            reranked.append(d)
        if len(reranked) >= keep:
            break
    return reranked[:keep]

# Answer chains
def format_context(docs: List[Document], limit: int, max_chars: int) -> str:
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

def build_answer_chain_default(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ответь строго по контексту PDF. Если не найдено — 'В документе не найдено'.\n"
         "Формат:\n"
         "1) краткий ответ (1-3 предложения)\n"
         "2) Доказательства: 2-5 пунктов с (стр. X) и короткой цитатой.\n"
         "Не выдумывай. Никаких страниц XX/??."
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:")
    ])
    return prompt | llm | StrOutputParser()

def build_answer_chain_summary(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Дай сводку 5-10 пунктов СТРОГО по контексту PDF.\n"
         "Каждый пункт: (стр. X) + короткая цитата в кавычках (до 12 слов).\n"
         "Не добавляй фактов, которых нет в контексте.\n"
         "Никогда не пиши выдуманные страницы типа XX/??.\n"
         "Если по контексту нельзя — 'В документе не найдено'."
        ),
        ("human", "Запрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:")
    ])
    return prompt | llm | StrOutputParser()

def build_answer_chain_compare(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Сравнение: приводи ТОЛЬКО то, что явно есть в контексте.\n"
         "Формат:\n"
         "1) Короткий вывод (1-2 предложения)\n"
         "2) Сравнения пунктами (3-8 пунктов): <показатель>: 2021 -> 2022 (стр. X) — \"цитата\".\n"
         "Никогда не пиши выдуманные страницы типа XX/??.\n"
         "Если нет данных — 'В документе не найдено'."
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:")
    ])
    return prompt | llm | StrOutputParser()

def build_answer_chain_citation_only(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Верни 3-7 коротких цитат из контекста (каждая в кавычках) + (стр. X).\n"
         "Только цитаты, без пересказа.\n"
         "Если страницы нет в контексте — не выдумывай.\n"
         "Никаких страниц XX/??.\n"
         "Если нет — 'В документе не найдено'."
        ),
        ("human", "Запрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:")
    ])
    return prompt | llm | StrOutputParser()

def build_answer_chain_numbers_strict(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Числа/проценты/даты: давай ТОЛЬКО то, что ДОСЛОВНО в контексте.\n"
         "Для каждого значения: страница + короткая цитата (1 строка).\n"
         "Формат:\n"
         "- <показатель>: <значение> (стр. X) — \"цитата\"\n"
         "Если ничего надежного — 'В документе не найдено'.\n"
         "Никаких страниц XX/??."
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:")
    ])
    return prompt | llm | StrOutputParser()

# QoS helpers
_NOT_FOUND_PAT = re.compile(r"^\s*(в документе не найдено|не найдено)\s*\.?\s*$", re.IGNORECASE)

def is_not_found_answer(ans: str) -> bool:
    if not ans:
        return True
    s = ans.strip().lower()
    if _NOT_FOUND_PAT.match(s):
        return True
    return "в документе не найдено" in s and len(s) < 140

def add_neighbors_from_parent_map(parent_map: dict, docs: List[Document], window: int, max_total: int = 320) -> List[Document]:
    if not docs or window <= 0:
        return docs
    out = list(docs)
    seen = set(doc_key(d) for d in out)

    for d in docs:
        m = d.metadata or {}
        parent_key = (
            m.get("source",""),
            m.get("page_start", m.get("page", None)),
            m.get("page_end", m.get("page", None)),
            m.get("section_title",""),
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

def _num_rerank_score(question: str, doc: Document) -> float:
    q = (question or "").lower()
    text = (doc.page_content or "").lower()
    toks = coverage_tokens(q)
    hit = _word_hit_ratio(toks, text)
    has_num = 1.0 if has_number(text) else 0.0
    has_pct = 1.0 if "%" in text or "процент" in text else 0.0
    has_date = 1.0 if DATE_RE.search(text) else 0.0
    return 0.55 * hit + 0.20 * has_num + 0.15 * has_pct + 0.10 * has_date

def rerank_numbers_heuristic(question: str, docs: List[Document], keep: int) -> List[Document]:
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


class BestStableRAGAgent:
    def __init__(self, settings: Settings, index_dir: str):
        self.settings = settings
        self.index_dir = index_dir
        self.llm, self.vectorstore, self.bm25, self._all_chunks = load_index(settings, index_dir)

        self._parent_to_chunks = defaultdict(list)
        for d in self._all_chunks:
            m = d.metadata or {}
            parent_key = (
                m.get("source",""),
                m.get("page_start", m.get("page", None)),
                m.get("page_end", m.get("page", None)),
                m.get("section_title",""),
            )
            cid = m.get("chunk_id", None)
            if cid is None:
                continue
            self._parent_to_chunks[parent_key].append(d)
        for k in list(self._parent_to_chunks.keys()):
            self._parent_to_chunks[k].sort(key=lambda x: int((x.metadata or {}).get("chunk_id", 0)))

        self.intent_chain_json, self.intent_chain_str = build_intent_chains(self.llm)
        self.multiquery_chain_json, self.multiquery_chain_str = build_multiquery_chains(self.llm)
        self.rerank_chain_json, self.rerank_chain_str = build_rerank_chains(self.llm)

        self.answer_chain_default = build_answer_chain_default(self.llm)
        self.answer_chain_summary = build_answer_chain_summary(self.llm)
        self.answer_chain_compare = build_answer_chain_compare(self.llm)
        self.answer_chain_citation_only = build_answer_chain_citation_only(self.llm)
        self.answer_chain_numbers = build_answer_chain_numbers_strict(self.llm)

    def _bm25_invoke(self, q: str, k: int) -> List[Document]:
        self.bm25.k = max(1, int(k))
        return (self.bm25.invoke(q) or [])[:k]

    def _retrieve_single(self, q: str, k_per_query: int) -> List[Document]:
        q = normalize_query(q)
        bm25_docs = self._bm25_invoke(q, k_per_query)
        faiss_docs = self.vectorstore.similarity_search(q, k=k_per_query)
        return dedup_docs(bm25_docs + faiss_docs, max_total=120)

    def _retrieve(self, query: str, intent: str, overrides: Optional[Dict[str, Any]] = None) -> List[Document]:
        query = normalize_query(query)
        o = overrides or {}

        bm25_k = int(o.get("bm25_k", self.settings.bm25_k))
        faiss_k = int(o.get("faiss_k", self.settings.faiss_k))
        k_per_query = int(o.get("multiquery_k_per_query", self.settings.multiquery_k_per_query))
        multiquery_enabled = bool(o.get("multiquery_enabled", self.settings.multiquery_enabled))

        if intent == "numbers_and_dates":
            mq_n = int(o.get("multiquery_n", self.settings.numbers_multiquery_n))
        else:
            mq_n = int(o.get("multiquery_n", self.settings.multiquery_n))

        if intent == "numbers_and_dates":
            mult = max(1, int(o.get("numbers_k_multiplier", self.settings.numbers_k_multiplier)))
            bm25_k = max(bm25_k * mult, bm25_k + 20)
            faiss_k = max(faiss_k * mult, faiss_k + 20)
            force_multiquery = True
        else:
            force_multiquery = False

        bm25_docs = self._bm25_invoke(query, bm25_k)
        faiss_docs = self.vectorstore.similarity_search(query, k=faiss_k)
        docs = dedup_docs(bm25_docs + faiss_docs, max_total=280)

        cov = coverage_score(query, docs, intent)

        if multiquery_enabled and (force_multiquery or cov < 0.40):
            fallback = {"queries": []}
            mq = invoke_json_robust(
                self.multiquery_chain_json, self.multiquery_chain_str,
                {"question": query, "n": mq_n},
                fallback, retries=1
            )
            queries = [normalize_query(x) for x in (mq.get("queries", []) or []) if isinstance(x, str)]
            queries = [x for x in queries if x][:mq_n]

            if queries:
                all_docs: List[Document] = []
                if bool(o.get("multiquery_parallel", self.settings.multiquery_parallel)):
                    threads = int(o.get("multiquery_threads", self.settings.multiquery_threads))
                    with ThreadPoolExecutor(max_workers=threads) as ex:
                        futures = [ex.submit(self._retrieve_single, q, k_per_query) for q in queries]
                        for fu in as_completed(futures):
                            try:
                                all_docs.extend(fu.result() or [])
                            except Exception:
                                pass
                else:
                    for q in queries:
                        all_docs.extend(self._retrieve_single(q, k_per_query))

                all_docs.extend(docs)
                docs = dedup_docs(all_docs, max_total=320)

        if intent == "numbers_and_dates":
            window = int(o.get("numbers_neighbors_window", self.settings.numbers_neighbors_window))
            docs = add_neighbors_from_parent_map(self._parent_to_chunks, docs, window=window, max_total=360)
            keep = int(o.get("rerank_keep", max(self.settings.rerank_keep, 14)))
            docs = rerank_numbers_heuristic(query, docs, keep=keep)
            if bool(o.get("numbers_disable_diversify", self.settings.numbers_disable_diversify)):
                return docs[: self.settings.final_docs_limit * 2]
            return diversify_docs(docs, max_per_group=self.settings.diversify_max_per_group)

        if bool(o.get("rerank_enabled", self.settings.rerank_enabled)) and len(docs) >= 10:
            docs = rerank_docs(
                self.rerank_chain_json, self.rerank_chain_str,
                question=query, docs=docs,
                keep=int(o.get("rerank_keep", self.settings.rerank_keep)),
                pool=int(o.get("rerank_pool", self.settings.rerank_pool)),
            )

        return diversify_docs(docs, max_per_group=self.settings.diversify_max_per_group)

    def ask(self, user_question: str) -> str:
        t0 = time.time()
        user_question = normalize_query(user_question)

        intent_obj = get_intent_hybrid(
            self.intent_chain_json, self.intent_chain_str,
            user_question,
            threshold=self.settings.llm_intent_threshold
        )
        intent = intent_obj.get("intent", "default")
        conf = float(intent_obj.get("confidence", 0.5))
        query = normalize_query(intent_obj.get("query", user_question)) or user_question

        docs = self._retrieve(query, intent)
        context = format_context(docs, limit=self.settings.final_docs_limit, max_chars=self.settings.max_context_chars)

        if intent == "summary":
            answer = self.answer_chain_summary.invoke({"question": user_question, "context": context})
        elif intent == "compare":
            answer = self.answer_chain_compare.invoke({"question": user_question, "context": context})
        elif intent == "citation_only":
            answer = self.answer_chain_citation_only.invoke({"question": user_question, "context": context})
        elif intent == "numbers_and_dates":
            answer = self.answer_chain_numbers.invoke({"question": user_question, "context": context})
        else:
            answer = self.answer_chain_default.invoke({"question": user_question, "context": context})

        needs_retry = False
        if contains_fake_pages(answer):
            needs_retry = True
        if intent in {"numbers_and_dates", "compare"} and answer_numbers_not_in_context(answer, context):
            needs_retry = True

        if (
            self.settings.second_pass_enabled
            and (needs_retry or is_not_found_answer(answer))
            and intent in {"numbers_and_dates", "citation_only", "summary", "compare"}
        ):
            mult = int(self.settings.second_pass_k_multiplier)
            overrides = {
                "multiquery_n": int(self.settings.second_pass_multiquery_n),
                "multiquery_k_per_query": max(self.settings.multiquery_k_per_query * mult, self.settings.multiquery_k_per_query + 30),
                "bm25_k": max(self.settings.bm25_k * mult, self.settings.bm25_k + 30),
                "faiss_k": max(self.settings.faiss_k * mult, self.settings.faiss_k + 30),
                "numbers_k_multiplier": max(self.settings.numbers_k_multiplier, 1),
                "numbers_neighbors_window": self.settings.numbers_neighbors_window,
                "numbers_disable_diversify": self.settings.numbers_disable_diversify,
                "rerank_keep": max(self.settings.rerank_keep, 14),
                "rerank_pool": self.settings.rerank_pool,
            }

            docs2 = self._retrieve(query, intent, overrides=overrides)
            context2 = format_context(docs2, limit=self.settings.final_docs_limit, max_chars=self.settings.max_context_chars)

            if intent == "summary":
                answer2 = self.answer_chain_summary.invoke({"question": user_question, "context": context2})
            elif intent == "compare":
                answer2 = self.answer_chain_compare.invoke({"question": user_question, "context": context2})
            elif intent == "citation_only":
                answer2 = self.answer_chain_citation_only.invoke({"question": user_question, "context": context2})
            elif intent == "numbers_and_dates":
                answer2 = self.answer_chain_numbers.invoke({"question": user_question, "context": context2})
            else:
                answer2 = answer

            if answer2 and not contains_fake_pages(answer2):
                if intent in {"numbers_and_dates", "compare"}:
                    if not answer_numbers_not_in_context(answer2, context2) and not is_not_found_answer(answer2):
                        answer = answer2
                else:
                    if not is_not_found_answer(answer2):
                        answer = answer2

        dt = time.time() - t0
        debug = f"[agent={AGENT_VERSION}] [intent={intent}, conf={conf:.2f}] time={dt:.2f}s query='{query}'"
        return debug + "\n\n" + (answer or "").strip()


# -----------------------------
# Session storage (per user)
# -----------------------------
def user_root_dir(settings: Settings, user_id: int) -> str:
    return os.path.join(settings.data_dir, "users", str(user_id))

def user_index_dir(settings: Settings, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "rag_index")

def user_pdf_path(settings: Settings, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "document.pdf")

def user_has_index(settings: Settings, user_id: int) -> bool:
    idx = user_index_dir(settings, user_id)
    return os.path.exists(os.path.join(idx, "meta.json")) and os.path.exists(os.path.join(idx, "chunks.jsonl.gz"))

def reset_user_session(settings: Settings, user_id: int) -> None:
    root = user_root_dir(settings, user_id)
    if os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)


# -----------------------------
# Telegram bot (aiogram polling)
# -----------------------------
from aiogram import Bot, Dispatcher, F  # noqa: E402
from aiogram.filters import Command  # noqa: E402
from aiogram.types import Message, Document as TgDocument  # noqa: E402
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError


BOT_TOKEN = must_get_secret("BOT_TOKEN")
GIGACHAT_API_KEY = must_get_secret("GIGACHAT_API_KEY")

# Expose to agent builders
os.environ["GIGACHAT_API_KEY"] = GIGACHAT_API_KEY

DATA_DIR = os.getenv("DATA_DIR", "./data")
SETTINGS = Settings(data_dir=DATA_DIR)

dp = Dispatcher()


# Keep agent instances in-memory (per user) to avoid reload for every message
# If container restarts — will be rebuilt from on-disk index.
AGENTS: Dict[int, BestStableRAGAgent] = {}
LOCKS: Dict[int, asyncio.Lock] = {}


def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in LOCKS:
        LOCKS[user_id] = asyncio.Lock()
    return LOCKS[user_id]

def get_or_build_agent(user_id: int) -> BestStableRAGAgent:
    if user_id in AGENTS:
        return AGENTS[user_id]
    if not user_has_index(SETTINGS, user_id):
        raise RuntimeError("NO_INDEX")
    agent = BestStableRAGAgent(SETTINGS, index_dir=user_index_dir(SETTINGS, user_id))
    AGENTS[user_id] = agent
    return agent


@dp.message(Command("start"))
async def cmd_start(msg: Message):
    text = (
        "Привет! Пришли мне PDF, я построю индекс и буду отвечать на вопросы по документу.\n\n"
        "Команды:\n"
        "/reset — очистить текущий документ и начать заново\n"
        "/status — статус (есть ли документ)\n\n"
        "После загрузки PDF просто задавай вопросы текстом."
    )
    await msg.answer(text)

@dp.message(Command("status"))
async def cmd_status(msg: Message):
    uid = msg.from_user.id
    has = user_has_index(SETTINGS, uid)
    await msg.answer("✅ Документ загружен и проиндексирован." if has else "❌ Документ не загружен. Пришли PDF.")

@dp.message(Command("reset"))
async def cmd_reset(msg: Message):
    uid = msg.from_user.id
    async with get_user_lock(uid):
        reset_user_session(SETTINGS, uid)
        AGENTS.pop(uid, None)
    await msg.answer("Ок. Сессия очищена. Пришли новый PDF.")

def is_pdf(doc: TgDocument) -> bool:
    if doc.mime_type == "application/pdf":
        return True
    if (doc.file_name or "").lower().endswith(".pdf"):
        return True
    return False

@dp.message(F.document)
async def on_pdf(msg: Message, bot: Bot):
    uid = msg.from_user.id
    doc: TgDocument = msg.document

    if not doc or not is_pdf(doc):
        await msg.answer("Пришли, пожалуйста, PDF-файл.")
        return

    # Можно заранее проверить размер, если Telegram прислал file_size
    # (не всегда помогает, но иногда сразу даёт понятное сообщение)
    max_hint_mb = 20
    if getattr(doc, "file_size", None) and doc.file_size > max_hint_mb * 1024 * 1024:
        await msg.answer(
            f"⚠️ Файл выглядит большим (~{doc.file_size/1024/1024:.1f} MB). "
            "Telegram Bot API может не дать его скачать. Если не получится — сожми/разбей PDF."
        )

    async with get_user_lock(uid):
        # New doc → wipe old session
        reset_user_session(SETTINGS, uid)
        AGENTS.pop(uid, None)

        ensure_dir(user_root_dir(SETTINGS, uid))
        pdf_path = user_pdf_path(SETTINGS, uid)

        await msg.answer("📥 Скачиваю PDF...")

        # Скачивание
        try:
            # Получаем file_path (может упасть на больших файлах: "file is too big")
            file = await bot.get_file(doc.file_id)

            # Таймаут на скачивание, чтобы не зависать бесконечно
            await asyncio.wait_for(
                bot.download_file(file.file_path, destination=pdf_path),
                timeout=180,
            )

        except TelegramBadRequest as e:
            # Самый частый кейс у тебя: Telegram не отдаёт file_path
            if "file is too big" in str(e).lower():
                await msg.answer(
                    "❌ PDF слишком большой для загрузки через Telegram Bot API.\n\n"
                    "Что можно сделать:\n"
                    "1) Сжать PDF (уменьшить качество/картинки)\n"
                    "2) Разбить на несколько частей и прислать по очереди\n"
                    "3) Прислать текстовую версию (без сканов), если есть\n\n"
                    "После этого пришли PDF заново."
                )
                return

            await msg.answer(
                f"❌ Telegram вернул ошибку при загрузке файла: {e}\n"
                "Попробуй прислать PDF ещё раз или /reset."
            )
            return

        except asyncio.TimeoutError:
            await msg.answer(
                "❌ Не удалось скачать PDF: таймаут.\n"
                "Попробуй ещё раз, либо сожми файл/разбей на части."
            )
            return

        except TelegramNetworkError as e:
            await msg.answer(
                "❌ Сетевая ошибка при скачивании PDF.\n"
                "Попробуй ещё раз через минуту."
            )
            return

        except Exception as e:
            logger.exception("PDF download failed: %s", e)
            await msg.answer("❌ Не удалось скачать PDF из Telegram. Попробуй ещё раз или /reset.")
            return

        # Базовая проверка, что файл реально появился и не пустой
        try:
            if (not os.path.exists(pdf_path)) or os.path.getsize(pdf_path) < 1024:
                await msg.answer(
                    "❌ Файл скачался некорректно (пустой/повреждён). "
                    "Попробуй прислать PDF ещё раз или сжать его."
                )
                return
        except Exception:
            pass

        await msg.answer("🔎 Индексирую… Это может занять 1–5 минут (зависит от PDF).")

        # Run ingest in thread to avoid blocking event loop
        idx_dir = user_index_dir(SETTINGS, uid)

        def _do_ingest():
            ingest_pdf(pdf_path, SETTINGS, index_dir=idx_dir)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _do_ingest)
        except Exception as e:
            logger.exception("ingest failed: %s", e)
            await msg.answer(
                "❌ Ошибка при индексировании PDF.\n"
                "Попробуй другой PDF или /reset."
            )
            return

        # Build agent and cache it
        try:
            AGENTS[uid] = BestStableRAGAgent(SETTINGS, index_dir=idx_dir)
        except Exception as e:
            logger.exception("agent build failed: %s", e)
            await msg.answer(
                "❌ Индекс построен, но не удалось поднять агента.\n"
                "Попробуй /reset и загрузить PDF заново."
            )
            return

    await msg.answer("✅ Готово! Теперь задавай вопросы по документу.")

@dp.message(F.text)
async def on_text(msg: Message):
    uid = msg.from_user.id
    q = (msg.text or "").strip()
    if not q:
        return

    async with get_user_lock(uid):
        try:
            agent = get_or_build_agent(uid)
        except RuntimeError as e:
            if str(e) == "NO_INDEX":
                await msg.answer("Сначала пришли PDF-файл, чтобы я построил индекс.")
                return
            raise

        await msg.answer("⏳ Думаю...")

        # Run LLM in executor (langchain sync calls)
        loop = asyncio.get_running_loop()
        def _ask():
            return agent.ask(q)

        try:
            ans = await loop.run_in_executor(None, _ask)
        except Exception as e:
            logger.exception("ask failed: %s", e)
            await msg.answer("Ошибка при обработке запроса. Попробуй ещё раз или /reset.")
            return

    # Telegram message limit ~4096
    if len(ans) <= 3800:
        await msg.answer(ans)
    else:
        # split
        for part in [ans[i:i+3800] for i in range(0, len(ans), 3800)]:
            await msg.answer(part)


async def main():
    bot = Bot(token=BOT_TOKEN)
    logger.info("Bot starting (polling). data_dir=%s", SETTINGS.data_dir)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
