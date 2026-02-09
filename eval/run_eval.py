#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragbot.agent import BestStableRAGAgent
from ragbot.indexing import ingest_pdf


@dataclass(frozen=True)
class EvalSettings:
    data_dir: str = "./data"
    chunk_size: int = 1100
    chunk_overlap: int = 150
    bm25_k: int = 14
    faiss_k: int = 14
    multiquery_enabled: bool = True
    multiquery_n: int = 2
    multiquery_parallel: bool = True
    multiquery_threads: int = 6
    multiquery_k_per_query: int = 12
    numbers_multiquery_n: int = 4
    rerank_enabled: bool = True
    rerank_pool: int = 18
    rerank_keep: int = 10
    final_docs_limit: int = 10
    max_context_chars: int = 16000
    diversify_max_per_group: int = 3
    llm_intent_threshold: float = 0.78
    temperature: float = 0.2
    top_p: float = 0.9
    model: str = "GigaChat-2-Max"
    embeddings_model: str = "GigaEmbeddings-3B-2025-09"
    query_embedding_cache_size: int = 512
    allow_dangerous_faiss_deserialization: bool = False
    allow_index_settings_mismatch: bool = False
    numbers_k_multiplier: int = 3
    numbers_neighbors_window: int = 2
    numbers_disable_diversify: bool = True
    second_pass_enabled: bool = True
    second_pass_multiquery_n: int = 6
    second_pass_k_multiplier: int = 5
    policy_variant: str = "control"  # control | ab_retrieval_v1
    policy_strict: bool = True


CITATION_PAT = re.compile(r"(?:\(\s*)?стр\.?\s*\d+", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not obj.get("id") or not obj.get("question"):
            raise ValueError(f"Bad question row: {line}")
        items.append(obj)
    return items


def split_debug_answer(raw: str) -> tuple[str, str]:
    raw = raw or ""
    if "\n\n" in raw:
        head, body = raw.split("\n\n", 1)
        if head.startswith("[agent="):
            return head, body.strip()
    return "", raw.strip()


def _normalize_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = s.replace("\u00a0", " ").replace("\u202f", " ")

    s = re.sub(r"\bболее\s+", ">", s)
    s = re.sub(r"\bсвыше\s+", ">", s)
    s = re.sub(r"\bменее\s+", "<", s)

    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _groups_from_question(q: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []

    for x in q.get("must_include", []) or []:
        groups.append([str(x)])

    # OR groups: each item can be string or list[str]
    for g in q.get("must_include_any", []) or []:
        if isinstance(g, list):
            groups.append([str(x) for x in g])
        else:
            groups.append([str(g)])

    return groups


def score_answer(answer: str, q: dict[str, Any]) -> dict[str, Any]:
    norm_answer = _normalize_text(answer)

    groups = _groups_from_question(q)
    must_not_include = [str(x) for x in (q.get("must_not_include") or [])]

    include_hits = 0
    for g in groups:
        g_norm = [_normalize_text(x) for x in g]
        if any(x and x in norm_answer for x in g_norm):
            include_hits += 1

    exclude_hits = 0
    for x in must_not_include:
        xn = _normalize_text(x)
        if xn and xn in norm_answer:
            exclude_hits += 1

    include_total = len(groups)
    include_rate = 1.0 if include_total == 0 else include_hits / include_total
    safe_ok = 1.0 if exclude_hits == 0 else 0.0

    has_citation = bool(CITATION_PAT.search(answer or ""))
    require_citation = bool(q.get("require_citation", False))
    citation_ok = 1.0 if (not require_citation or has_citation) else 0.0

    base_score = 0.7 * include_rate + 0.3 * safe_ok
    citation_penalty = 0.2 if (require_citation and not has_citation) else 0.0
    score = max(0.0, base_score - citation_penalty)

    return {
        "include_hits": include_hits,
        "include_total": include_total,
        "exclude_hits": exclude_hits,
        "include_rate": include_rate,
        "safe_ok": bool(safe_ok),
        "require_citation": require_citation,
        "citation_ok": bool(citation_ok),
        "citation_penalty": citation_penalty,
        "score": score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-variant", default=os.getenv("POLICY_VARIANT", "control"))
    ap.add_argument("--policy-strict", action="store_true", help="Fail on unknown policy variant/intent")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    questions_path = Path(args.questions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    settings = EvalSettings(
        policy_variant=str(args.policy_variant or "control").strip().lower(),
        policy_strict=bool(args.policy_strict),
    )

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        idx_dir = Path(tmp) / "index"
        ingest_pdf(str(pdf), settings, str(idx_dir), agent_version="eval")
        agent = BestStableRAGAgent(settings, index_dir=str(idx_dir), agent_version="eval")

        rows = []
        weighted_sum = 0.0
        weighted_score = 0.0

        for q in questions:
            raw = agent.ask(q["question"])
            debug, answer = split_debug_answer(raw)
            m = score_answer(answer, q)
            w = float(q.get("weight", 1.0))
            weighted_sum += w
            weighted_score += m["score"] * w

            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "weight": w,
                    "debug": debug,
                    "answer": answer,
                    "trace": agent.get_last_trace() if hasattr(agent, "get_last_trace") else {},
                    "metrics": m,
                }
            )

    report = {
        "summary": {
            "questions": len(rows),
            "weighted_score": (weighted_score / weighted_sum) if weighted_sum else 0.0,
            "policy_variant": settings.policy_variant,
            "policy_strict": settings.policy_strict,
            "pdf_sha256": sha256_file(pdf),
            "questions_sha256": sha256_file(questions_path),
            "formula": {
                "base": "question_score = 0.7 * include_rate + 0.3 * safe_ok",
                "citation_penalty": "-0.2 if require_citation and citation missing",
                "overall": "weighted_score = sum(question_score_i * weight_i) / sum(weight_i)",
            },
        },
        "results": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
