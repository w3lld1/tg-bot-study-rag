import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

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
    build_extractive_evidence,
    build_extractive_plan,
    coverage_score,
    format_context,
    invoke_json_robust,
    is_not_found_answer,
    rerank_numbers_heuristic,
)
from ragbot.indexing import load_index
from ragbot.policy import get_retrieval_policy, get_second_pass_overrides, should_trigger_multiquery
from ragbot.text_utils import contains_fake_pages, coverage_tokens, dedup_docs, diversify_docs, has_number, normalize_query


logger = logging.getLogger("tg-rag-bot")

_CITATION_RE = re.compile(r"(?:\(\s*)?стр\.?\s*\d+", re.IGNORECASE)


def _has_citation(answer: str) -> bool:
    """
    Проверяет, содержит ли текст ссылку на страницу вида `(стр. X)` или `стр. X`.
    """
    return bool(_CITATION_RE.search(answer or ""))


def _numeric_fallback_line_quality(question: str, line: str) -> float:
    """
    Эвристика качества строки для numeric fallback: число + лексическое попадание по вопросу.
    """
    if not line:
        return 0.0
    q_toks = coverage_tokens(question)
    line_low = line.lower()
    lexical_hits = sum(1 for t in q_toks if t and t in line_low)
    lexical_ratio = lexical_hits / max(1, len(q_toks))
    number_bonus = 0.35 if has_number(line) else 0.0
    return lexical_ratio + number_bonus


class BestStableRAGAgent:
    """
    Основной runtime-класс RAG-агента: управляет retrieval, генерацией ответа, валидацией и fallback-стратегиями.
    """
    def __init__(self, settings: Any, index_dir: str, agent_version: str):
        """
        Инициализирует агент: загружает индексы, строит parent→chunks map и поднимает все LLM-цепочки.
        """
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

        self._last_trace: Dict[str, Any] = {}

    def get_last_trace(self) -> Dict[str, Any]:
        """
        Возвращает технический trace последнего запроса (intent/retrieval/validation/repair).
        """
        return dict(self._last_trace or {})

    def _bm25_invoke(self, q: str, k: int) -> List[Document]:
        """
        Выполняет BM25-поиск с заданным `k` и обрезает результат до нужного размера.
        """
        self.bm25.k = max(1, int(k))
        return (self.bm25.invoke(q) or [])[:k]

    def _retrieve_single(self, q: str, k_per_query: int) -> List[Document]:
        """
        Один проход гибридного retrieval для одного запроса: BM25 + FAISS + dedup.
        """
        q = normalize_query(q)
        bm25_docs = self._bm25_invoke(q, k_per_query)
        faiss_docs = self.vectorstore.similarity_search(q, k=k_per_query)
        return dedup_docs(bm25_docs + faiss_docs, max_total=120)

    def _retrieve_with_trace(self, query: str, intent: str, overrides: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Полный retrieval-этап с трассировкой: policy, coverage, multiquery, rerank и post-processing.
        """
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

        policy = get_retrieval_policy(self.settings, intent)
        force_multiquery = bool(policy.get("force_multiquery", intent == "numbers_and_dates"))
        cov_threshold = float(policy.get("cov_threshold", 0.50))

        trace: Dict[str, Any] = {
            "query": query,
            "intent": intent,
            "bm25_k": bm25_k,
            "faiss_k": faiss_k,
            "multiquery_enabled": multiquery_enabled,
            "multiquery_n": mq_n,
            "k_per_query": k_per_query,
            "force_multiquery": force_multiquery,
            "cov_threshold": cov_threshold,
            "policy_variant": policy.get("variant"),
            "policy_rule_id": policy.get("rule_id"),
            "overrides": dict(o),
            "multiquery_triggered": False,
            "multiquery_queries": [],
        }

        bm25_docs = self._bm25_invoke(query, bm25_k)
        faiss_docs = self.vectorstore.similarity_search(query, k=faiss_k)
        docs = dedup_docs(bm25_docs + faiss_docs, max_total=280)

        if intent == "numbers_and_dates":
            # Детерминированный query-rewrite для числовых вопросов (без LLM)
            toks = coverage_tokens(query)
            numeric_hints = [t for t in ["млрд", "млн", "трлн", "%", "процент", "ai", "ии", "mau", "dau", "p2p"] if t in (query or "").lower()]
            seed = " ".join(dict.fromkeys(toks + numeric_hints))
            if seed and seed != query:
                seed_bm25 = self._bm25_invoke(seed, max(12, min(40, bm25_k // 2)))
                seed_faiss = self.vectorstore.similarity_search(seed, k=max(12, min(40, faiss_k // 2)))
                docs = dedup_docs(docs + seed_bm25 + seed_faiss, max_total=320)
                trace["numeric_seed_query"] = seed

        trace["docs_initial"] = len(docs)

        cov = coverage_score(query, docs, intent)
        trace["coverage"] = float(cov)

        mq_triggered = should_trigger_multiquery(
            multiquery_enabled=multiquery_enabled,
            force_multiquery=force_multiquery,
            coverage=float(cov),
            cov_threshold=cov_threshold,
        )
        trace["multiquery_decision"] = {
            "coverage": float(cov),
            "cov_threshold": cov_threshold,
            "force_multiquery": force_multiquery,
            "triggered": mq_triggered,
        }

        if mq_triggered:
            trace["multiquery_triggered"] = True
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
            trace["multiquery_queries"] = list(queries)

            if queries:
                all_docs: List[Document] = []
                if bool(o.get("multiquery_parallel", self.settings.multiquery_parallel)):
                    threads = int(o.get("multiquery_threads", self.settings.multiquery_threads))
                    trace["multiquery_parallel"] = True
                    trace["multiquery_threads"] = threads
                    with ThreadPoolExecutor(max_workers=threads) as ex:
                        futures = [ex.submit(self._retrieve_single, q, k_per_query) for q in queries]
                        for fu in as_completed(futures):
                            try:
                                all_docs.extend(fu.result() or [])
                            except Exception:
                                pass
                else:
                    trace["multiquery_parallel"] = False
                    for q in queries:
                        all_docs.extend(self._retrieve_single(q, k_per_query))

                all_docs.extend(docs)
                docs = dedup_docs(all_docs, max_total=320)

        if intent == "numbers_and_dates":
            window = int(o.get("numbers_neighbors_window", self.settings.numbers_neighbors_window))
            docs = add_neighbors_from_parent_map(self._parent_to_chunks, docs, window=window, max_total=360)
            keep = int(o.get("rerank_keep", max(self.settings.rerank_keep, 14)))
            docs = rerank_numbers_heuristic(query, docs, keep=keep)
            trace["numbers_neighbors_window"] = window
            trace["numbers_rerank_keep"] = keep
            trace["numbers_disable_diversify"] = bool(o.get("numbers_disable_diversify", self.settings.numbers_disable_diversify))
            if bool(o.get("numbers_disable_diversify", self.settings.numbers_disable_diversify)):
                docs = docs[: self.settings.final_docs_limit * 2]
                trace["docs_final"] = len(docs)
                return docs, trace
            docs = diversify_docs(docs, max_per_group=self.settings.diversify_max_per_group)
            trace["docs_final"] = len(docs)
            return docs, trace

        if bool(o.get("rerank_enabled", self.settings.rerank_enabled)) and len(docs) >= 10:
            docs = rerank_docs(
                self.rerank_chain_json,
                self.rerank_chain_str,
                question=query,
                docs=docs,
                keep=int(o.get("rerank_keep", self.settings.rerank_keep)),
                pool=int(o.get("rerank_pool", self.settings.rerank_pool)),
            )
            trace["rerank_applied"] = True
        else:
            trace["rerank_applied"] = False

        docs = diversify_docs(docs, max_per_group=self.settings.diversify_max_per_group)
        trace["docs_final"] = len(docs)
        return docs, trace

    def _retrieve(self, query: str, intent: str, overrides: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Упрощённая обёртка над `_retrieve_with_trace`, возвращает только документы.
        """
        docs, _ = self._retrieve_with_trace(query, intent, overrides=overrides)
        return docs

    def _intent_stage(self, user_question: str) -> Dict[str, Any]:
        """
        Определяет intent/уверенность/рабочий query для вопроса пользователя.
        """
        intent_obj = get_intent_hybrid(
            self.intent_chain_json,
            self.intent_chain_str,
            user_question,
            threshold=self.settings.llm_intent_threshold,
        )
        intent = intent_obj.get("intent", "default")
        conf = float(intent_obj.get("confidence", 0.5))
        query = normalize_query(intent_obj.get("query", user_question)) or user_question
        return {"intent": intent, "confidence": conf, "query": query}

    def _answer_stage(self, intent: str, user_question: str, context: str, plan: Optional[Dict[str, Any]] = None, fallback_answer: str = "") -> str:
        """
        Выбирает нужную answer-chain по intent и генерирует черновой ответ.
        """
        synthesis_context = (plan or {}).get("synthesis_context", context)
        if intent == "summary":
            return self.answer_chain_summary.invoke({"question": user_question, "context": synthesis_context})
        if intent == "compare":
            return self.answer_chain_compare.invoke({"question": user_question, "context": synthesis_context})
        if intent == "citation_only":
            return self.answer_chain_citation_only.invoke({"question": user_question, "context": context})
        if intent == "numbers_and_dates":
            return self.answer_chain_numbers.invoke({"question": user_question, "context": synthesis_context})
        if fallback_answer:
            return fallback_answer
        return self.answer_chain_default.invoke({"question": user_question, "context": synthesis_context})

    def _validation_stage(self, intent: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Проверяет ответ на типовые ошибки: fake pages, numeric mismatch, отсутствие цитат.
        """
        has_fake_pages = contains_fake_pages(answer)
        numbers_mismatch = intent in {"numbers_and_dates", "compare"} and answer_numbers_not_in_context(answer, context)
        not_found = is_not_found_answer(answer)
        citation_missing = (not not_found) and (not _has_citation(answer))
        needs_retry = bool(has_fake_pages or numbers_mismatch)
        return {
            "has_fake_pages": has_fake_pages,
            "numbers_mismatch": numbers_mismatch,
            "not_found": not_found,
            "citation_missing": citation_missing,
            "needs_retry": needs_retry,
        }

    def _second_pass_overrides(self, intent: str) -> Dict[str, Any]:
        """
        Формирует усиленные параметры retrieval для второго прохода.
        """
        return get_second_pass_overrides(self.settings, intent)

    def _should_second_pass(self, intent: str, validation: Dict[str, Any]) -> bool:
        """
        Решает, нужен ли second pass по результатам первичной валидации.
        """
        return bool(
            self.settings.second_pass_enabled
            and (validation["needs_retry"] or validation["not_found"] or validation["citation_missing"])
            and intent in {"numbers_and_dates", "citation_only", "summary", "compare", "default", "definition", "procedure", "requirements"}
        )

    def _repair_stage(self, user_question: str, answer: str, context: str, intent: str) -> str:
        """
        Пост-обработка ответа: добавление цитат/фрагментов и выравнивание not-found поведения.
        """
        if context and (not is_not_found_answer(answer)) and (not _has_citation(answer)):
            citations = self.answer_chain_citation_only.invoke({"question": user_question, "context": context})
            if citations and (not is_not_found_answer(citations)) and _has_citation(citations):
                return (answer or "").strip() + "\n\nДоказательства:\n" + citations.strip()

        extractive = build_extractive_evidence(user_question, context, max_items=5)
        if extractive:
            if is_not_found_answer(answer):
                if intent == "numbers_and_dates":
                    num_lines = [ln for ln in extractive.splitlines() if has_number(ln)]
                    if num_lines:
                        ranked = sorted(
                            [(_numeric_fallback_line_quality(user_question, ln), ln) for ln in num_lines],
                            key=lambda x: x[0],
                            reverse=True,
                        )
                        good = [ln for score, ln in ranked if score >= 0.42]
                        if good:
                            return "По релевантным фрагментам:\n" + "\n".join(good[:2])
                    return "В документе не найдено."
                if intent in {"structure_list"}:
                    return "В документе не найдено."
                return "Найденные релевантные фрагменты:\n" + extractive
            body = (answer or "").strip()
            if "Фрагменты из документа:" not in body and "Ключевые цитаты:" not in body:
                tail_title = "Фрагменты из документа:" if not _has_citation(body) else "Ключевые цитаты:"
                return body + f"\n\n{tail_title}\n" + extractive
        return answer

    def _enforce_numbers_terminal_quality(self, answer: str) -> Dict[str, Any]:
        """
        Финальный guard для numeric-intent: либо число+ссылка, либо "В документе не найдено.".
        """
        ans = (answer or "").strip()
        low = ans.lower()

        if low.startswith("найденные релевантные фрагменты"):
            return {"answer": "В документе не найдено.", "guard_applied": True, "reason": "fragments_fallback"}

        if is_not_found_answer(ans):
            return {"answer": "В документе не найдено.", "guard_applied": False, "reason": "already_not_found"}

        if has_number(ans) and _has_citation(ans):
            return {"answer": ans, "guard_applied": False, "reason": "ok"}

        return {"answer": "В документе не найдено.", "guard_applied": True, "reason": "missing_numeric_or_citation"}

    def ask(self, user_question: str) -> str:
        """
        Основной метод QA: orchestrator всего пайплайна от intent до финального ответа.
        """
        t0 = time.time()
        user_question = normalize_query(user_question)

        trace: Dict[str, Any] = {
            "version": 1,
            "question": user_question,
        }

        intent_obj = self._intent_stage(user_question)
        intent = intent_obj["intent"]
        conf = intent_obj["confidence"]
        query = intent_obj["query"]
        trace["intent_stage"] = intent_obj

        docs, retrieval_trace = self._retrieve_with_trace(query, intent)
        context = format_context(docs, limit=self.settings.final_docs_limit, max_chars=self.settings.max_context_chars)
        trace["retrieve_stage"] = {
            **retrieval_trace,
            "context_chars": len(context or ""),
            "final_docs_limit": int(self.settings.final_docs_limit),
        }

        plan = build_extractive_plan(
            user_question,
            context,
            intent,
            max_items=8,
            min_score_threshold=float(getattr(self.settings, "planner_min_score_threshold", 0.12)),
            max_per_page=int(getattr(self.settings, "planner_max_per_page", 2)),
            hierarchical_max_sections=int(getattr(self.settings, "planner_hierarchical_max_sections", 6)),
            hierarchical_max_sentences_per_section=int(getattr(self.settings, "planner_hierarchical_max_sentences_per_section", 3)),
        )
        trace["planner_stage"] = {
            "evidence_items": len(plan.get("evidence", [])),
            "lexical_report": plan.get("lexical_report", {}),
            "guardrails": plan.get("guardrails", {}),
            "hierarchical_report": (plan.get("hierarchical_context", {}) or {}).get("report", {}),
        }

        active_context = context
        active_plan = plan

        answer = self._answer_stage(intent, user_question, active_context, plan=active_plan)
        validation = self._validation_stage(intent, answer, active_context)
        trace["answer_stage"] = {
            "intent": intent,
            "chain": intent if intent in {"summary", "compare", "citation_only", "numbers_and_dates"} else "default",
            "answer_chars": len(answer or ""),
            "planner_used": True,
            **validation,
        }

        trace["second_pass"] = {"triggered": False}
        if self._should_second_pass(intent, validation):
            overrides = self._second_pass_overrides(intent)
            docs2, retrieval_trace2 = self._retrieve_with_trace(query, intent, overrides=overrides)
            context2 = format_context(docs2, limit=self.settings.final_docs_limit, max_chars=self.settings.max_context_chars)
            plan2 = build_extractive_plan(
                user_question,
                context2,
                intent,
                max_items=8,
                min_score_threshold=float(getattr(self.settings, "planner_min_score_threshold", 0.12)),
                max_per_page=int(getattr(self.settings, "planner_max_per_page", 2)),
                hierarchical_max_sections=int(getattr(self.settings, "planner_hierarchical_max_sections", 6)),
                hierarchical_max_sentences_per_section=int(getattr(self.settings, "planner_hierarchical_max_sentences_per_section", 3)),
            )
            answer2 = self._answer_stage(intent, user_question, context2, plan=plan2, fallback_answer=answer)

            accepted = False
            if answer2 and not contains_fake_pages(answer2):
                if intent in {"numbers_and_dates", "compare"}:
                    if not answer_numbers_not_in_context(answer2, context2) and not is_not_found_answer(answer2):
                        answer = answer2
                        active_context = context2
                        active_plan = plan2
                        accepted = True
                else:
                    if not is_not_found_answer(answer2):
                        answer = answer2
                        active_context = context2
                        active_plan = plan2
                        accepted = True

            trace["second_pass"] = {
                "triggered": True,
                "accepted": accepted,
                "reason": {
                    "needs_retry": validation["needs_retry"],
                    "not_found": validation["not_found"],
                    "citation_missing": validation["citation_missing"],
                },
                "overrides": overrides,
                "retrieve": {
                    **retrieval_trace2,
                    "context_chars": len(context2 or ""),
                },
                "answer_chars": len(answer2 or ""),
            }

        answer_before_repair = answer
        answer = self._repair_stage(user_question, answer, active_context, intent)

        numeric_guard = {"guard_applied": False, "reason": "not_applicable"}
        if intent == "numbers_and_dates":
            numeric_guard = self._enforce_numbers_terminal_quality(answer)
            answer = numeric_guard["answer"]

        trace["repair_stage"] = {
            "citation_repair_applied": answer != answer_before_repair,
            "final_answer_has_citation": _has_citation(answer),
            "final_answer_not_found": is_not_found_answer(answer),
            "numeric_terminal_guard": numeric_guard,
        }

        dt = time.time() - t0
        trace["timing"] = {"total_seconds": round(dt, 4)}
        self._last_trace = trace

        debug = f"[agent={self.agent_version}] [intent={intent}, conf={conf:.2f}] time={dt:.2f}s query='{query}'"
        return debug + "\n\n" + (answer or "").strip()
