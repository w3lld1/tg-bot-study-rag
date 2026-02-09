#!/usr/bin/env python3
import argparse
import json
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


def score_answer(answer: str, q: dict[str, Any]) -> dict[str, Any]:
    lower = answer.lower()
    must_include = q.get("must_include") or []
    must_not_include = q.get("must_not_include") or []

    include_hits = sum(1 for x in must_include if str(x).lower() in lower)
    exclude_hits = sum(1 for x in must_not_include if str(x).lower() in lower)

    include_rate = 1.0 if not must_include else include_hits / len(must_include)
    safe_ok = 1.0 if exclude_hits == 0 else 0.0

    score = 0.7 * include_rate + 0.3 * safe_ok
    return {
        "include_hits": include_hits,
        "include_total": len(must_include),
        "exclude_hits": exclude_hits,
        "include_rate": include_rate,
        "safe_ok": bool(safe_ok),
        "score": score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    questions_path = Path(args.questions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    settings = EvalSettings()

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
                    "metrics": m,
                }
            )

    report = {
        "summary": {
            "questions": len(rows),
            "weighted_score": (weighted_score / weighted_sum) if weighted_sum else 0.0,
        },
        "results": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
