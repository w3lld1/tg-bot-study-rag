import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ragbot.chains import (
    build_answer_chain_citation_only,
    build_answer_chain_compare,
    build_answer_chain_default,
    build_answer_chain_numbers_strict,
    build_answer_chain_summary,
    build_intent_chains,
    build_llm,
    build_multiquery_chains,
    build_rerank_chains,
    get_intent_hybrid,
    rerank_docs,
)
from ragbot.core_logic import (
    add_neighbors_from_parent_map,
    answer_numbers_not_in_context,
    coverage_score,
    format_context,
    invoke_json_robust,
    is_not_found_answer,
    rerank_numbers_heuristic,
)
from ragbot.indexing import load_index
from ragbot.text_utils import contains_fake_pages, dedup_docs, diversify_docs, normalize_query


logger = logging.getLogger("tg-rag-bot")

_CITATION_RE = re.compile(r"(?:\(\s*)?стр\.?\s*\d+", re.IGNORECASE)


def _has_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer or ""))


class BestStableRAGAgent:
    def __init__(self, settings: Any, index_dir: str, agent_version: str):
        self.settings = settings
        self.index_dir = index_dir
        self.agent_version = agent_version
        self.llm, self.vectorstore, self.bm25, self._all_chunks = load_index(settings, index_dir, build_llm)

        self._parent_to_chunks = defaultdict(list)
        for d in self._all_chunks:
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

        if multiquery_enabled and (force_multiquery or cov < 0.50):
            fallback = {"queries": []}
            mq = invoke_json_robust(
                self.multiquery_chain_json,
                self.multiquery_chain_str,
                {"question": query, "n": mq_n},
                fallback,
                retries=1,
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
                self.rerank_chain_json,
                self.rerank_chain_str,
                question=query,
                docs=docs,
                keep=int(o.get("rerank_keep", self.settings.rerank_keep)),
                pool=int(o.get("rerank_pool", self.settings.rerank_pool)),
            )

        return diversify_docs(docs, max_per_group=self.settings.diversify_max_per_group)

    def ask(self, user_question: str) -> str:
        t0 = time.time()
        user_question = normalize_query(user_question)

        intent_obj = get_intent_hybrid(
            self.intent_chain_json,
            self.intent_chain_str,
            user_question,
            threshold=self.settings.llm_intent_threshold,
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

        citation_missing = (not is_not_found_answer(answer)) and (not _has_citation(answer))

        if (
            self.settings.second_pass_enabled
            and (needs_retry or is_not_found_answer(answer) or citation_missing)
            and intent in {"numbers_and_dates", "citation_only", "summary", "compare", "default", "definition", "procedure", "requirements"}
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

        # Final citation repair pass (universal): if answer has no page refs, append citation-only evidence.
        if context and (not is_not_found_answer(answer)) and (not _has_citation(answer)):
            citations = self.answer_chain_citation_only.invoke({"question": user_question, "context": context})
            if citations and (not is_not_found_answer(citations)) and _has_citation(citations):
                answer = (answer or "").strip() + "\n\nДоказательства:\n" + citations.strip()

        dt = time.time() - t0
        debug = f"[agent={self.agent_version}] [intent={intent}, conf={conf:.2f}] time={dt:.2f}s query='{query}'"
        return debug + "\n\n" + (answer or "").strip()
