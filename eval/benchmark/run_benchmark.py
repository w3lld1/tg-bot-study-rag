#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_one(
    repo_root: Path,
    pdf: str,
    questions: str,
    out: Path,
    policy_variant: str,
    policy_strict: bool,
    index_dir: Path,
    reuse_index: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "eval/run_eval.py",
        "--pdf",
        pdf,
        "--questions",
        questions,
        "--out",
        str(out),
        "--policy-variant",
        policy_variant,
        "--index-dir",
        str(index_dir),
    ]
    if policy_strict:
        cmd.append("--policy-strict")
    if not reuse_index:
        cmd.append("--no-reuse-index")
    p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"run_eval failed for {pdf}:\n{p.stderr or p.stdout}")
    return json.loads(out.read_text(encoding="utf-8"))


def run_benchmark_once(
    *,
    repo_root: Path,
    datasets: list[dict[str, Any]],
    tmp_dir: Path,
    index_cache_dir: Path,
    policy_variant: str,
    policy_strict: bool,
    reuse_index: bool,
    run_tag: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0
    total_questions = 0
    question_weighted_score = 0.0

    for idx, ds in enumerate(datasets, start=1):
        if "id" not in ds or "pdf" not in ds or "questions" not in ds:
            raise ValueError(f"Dataset #{idx} in config is missing one of required keys: id, pdf, questions")

        ds_id = ds["id"]
        w = float(ds.get("weight", 1.0))
        pdf_path = (repo_root / ds["pdf"]).resolve()
        questions_path = (repo_root / ds["questions"]).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found for dataset '{ds_id}': {pdf_path}")
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found for dataset '{ds_id}': {questions_path}")

        one_out = tmp_dir / f"{run_tag}.{ds_id}.json"

        pdf_hash_short = sha256_file(pdf_path)[:12]
        index_dir = index_cache_dir / f"{ds_id}-{pdf_hash_short}"

        report = run_one(
            repo_root,
            str(pdf_path),
            str(questions_path),
            one_out,
            policy_variant=policy_variant,
            policy_strict=policy_strict,
            index_dir=index_dir,
            reuse_index=reuse_index,
        )
        score = float(report["summary"]["weighted_score"])
        q_count = int(report["summary"].get("questions", 0) or 0)

        total_weight += w
        weighted_score += score * w

        total_questions += q_count
        question_weighted_score += score * q_count

        rows.append(
            {
                "id": ds_id,
                "kind": ds.get("kind", "unknown"),
                "weight": w,
                "pdf": ds["pdf"],
                "questions": ds["questions"],
                "pdf_sha256": sha256_file(pdf_path),
                "questions_sha256": sha256_file(questions_path),
                "questions_count": q_count,
                "weighted_score": score,
                "run": str(one_out.relative_to(repo_root)),
                "index_dir": str(index_dir.relative_to(repo_root)),
            }
        )

    return {
        "datasets": rows,
        "summary": {
            "weighted_score": (weighted_score / total_weight) if total_weight else 0.0,
            "question_weighted_score": (question_weighted_score / total_questions) if total_questions else 0.0,
            "total_questions": total_questions,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp-dir", default="eval/runs")
    ap.add_argument("--index-cache-dir", default="eval/.cache/indexes")
    ap.add_argument("--no-reuse-index", action="store_true")
    ap.add_argument("--policy-variant", default="control")
    ap.add_argument("--policy-strict", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    out_path = (repo_root / args.out).resolve()
    tmp_dir = (repo_root / args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir = (repo_root / args.index_cache_dir).resolve()
    index_cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])

    run_reports: list[dict[str, Any]] = []
    for i in range(1, args.runs + 1):
        tag = f"run{i}"
        rep = run_benchmark_once(
            repo_root=repo_root,
            datasets=datasets,
            tmp_dir=tmp_dir,
            index_cache_dir=index_cache_dir,
            policy_variant=args.policy_variant,
            policy_strict=bool(args.policy_strict),
            reuse_index=not bool(args.no_reuse_index),
            run_tag=tag,
        )
        run_file = out_path if args.runs == 1 else out_path.with_name(f"{out_path.stem}.run{i}{out_path.suffix}")

        out_single = {
            "summary": {
                "benchmark": cfg.get("name", "benchmark"),
                "datasets": len(rep["datasets"]),
                "policy_variant": args.policy_variant,
                "policy_strict": bool(args.policy_strict),
                "index_cache_dir": str(index_cache_dir.relative_to(repo_root)),
                "reuse_index": not bool(args.no_reuse_index),
                "run": i,
                **rep["summary"],
            },
            "datasets": rep["datasets"],
        }
        run_file.parent.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps(out_single, ensure_ascii=False, indent=2), encoding="utf-8")
        run_reports.append({"run": i, "file": str(run_file.relative_to(repo_root)), **out_single["summary"]})

    if args.runs == 1:
        final_output = {
            "summary": run_reports[0],
            "datasets": json.loads((repo_root / run_reports[0]["file"]).read_text(encoding="utf-8"))["datasets"],
        }
    else:
        ws = [float(r["weighted_score"]) for r in run_reports]
        qws = [float(r["question_weighted_score"]) for r in run_reports]
        final_output = {
            "summary": {
                "benchmark": cfg.get("name", "benchmark"),
                "datasets": len(datasets),
                "policy_variant": args.policy_variant,
                "policy_strict": bool(args.policy_strict),
                "index_cache_dir": str(index_cache_dir.relative_to(repo_root)),
                "reuse_index": not bool(args.no_reuse_index),
                "runs": args.runs,
                "weighted_score": statistics.mean(ws),
                "question_weighted_score": statistics.mean(qws),
                "weighted_score_stats": _stats(ws),
                "question_weighted_score_stats": _stats(qws),
                "total_questions": int(run_reports[0]["total_questions"] if run_reports else 0),
            },
            "runs": run_reports,
            "datasets": json.loads((repo_root / run_reports[0]["file"]).read_text(encoding="utf-8"))["datasets"] if run_reports else [],
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.runs == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(final_output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
