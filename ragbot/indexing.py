import gzip
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Any, List, Optional

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.text_utils import ensure_dir, fix_broken_numbers, normalize_query


logger = logging.getLogger("tg-rag-bot")


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


def build_embeddings(settings: Any) -> Embeddings:
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


def chunk_parent_docs(parent_docs: List[Document], settings: Any) -> List[Document]:
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
            out.append(Document(page_content=rec.get("page_content", ""), metadata=rec.get("metadata", {}) or {}))
    return out


def _check_settings_compat(stored: dict, current: Any) -> None:
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


def ingest_pdf(pdf_path: str, settings: Any, index_dir: str, agent_version: str) -> None:
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
        "agent_version": agent_version,
        "settings": dict(settings.__dict__),
        "chunk_count": len(chunk_docs),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(pdf_path_txt, "w", encoding="utf-8") as f:
        f.write(pdf_path)

    logger.info("[ingest] Done ✅")


def load_index(settings: Any, index_dir: str, llm_builder):
    faiss_dir = os.path.join(index_dir, "faiss")
    meta_json = os.path.join(index_dir, "meta.json")
    chunks_jsonl_gz = os.path.join(index_dir, "chunks.jsonl.gz")

    if not os.path.exists(meta_json) or not os.path.exists(chunks_jsonl_gz):
        raise RuntimeError("Индекс не найден. Сначала надо ingest_pdf(...)")

    llm = llm_builder(settings)
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
