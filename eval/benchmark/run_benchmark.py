#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.benchmark.gate_utils import GateConfig, evaluate_multi_run_gate

EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_INFRA_FAIL = 2


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_gate_enabled(args: argparse.Namespace) -> bool:
    return any(
        x is not None
        for x in [
            args.max_regressions,
            args.max_std_weighted,
            args.max_std_question,
            args.max_drop_vs_baseline,
            args.baseline_file,
        ]
    )


def _load_baseline_summary(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("summary", d)


def _rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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


def _run_dataset_once(
    *,
    repo_root: Path,
    ds: dict[str, Any],
    ds_idx: int,
    tmp_dir: Path,
    index_cache_dir: Path,
    policy_variant: str,
    policy_strict: bool,
    reuse_index: bool,
    run_tag: str,
) -> tuple[int, dict[str, Any], float, float, int]:
    if "id" not in ds or "pdf" not in ds or "questions" not in ds:
        raise ValueError(f"Dataset #{ds_idx} in config is missing one of required keys: id, pdf, questions")

    ds_id = ds["id"]
    w = float(ds.get("weight", 1.0))
    pdf_path = (repo_root / ds["pdf"]).resolve()
    questions_path = (repo_root / ds["questions"]).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found for dataset '{ds_id}': {pdf_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found for dataset '{ds_id}': {questions_path}")

    one_out = tmp_dir / f"{run_tag}.{ds_id}.json"
    pdf_hash = sha256_file(pdf_path)
    questions_hash = sha256_file(questions_path)
    pdf_hash_short = pdf_hash[:12]
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

    row = {
        "id": ds_id,
        "kind": ds.get("kind", "unknown"),
        "weight": w,
        "pdf": ds["pdf"],
        "questions": ds["questions"],
        "pdf_sha256": pdf_hash,
        "questions_sha256": questions_hash,
        "questions_count": q_count,
        "weighted_score": score,
        "run": _rel_or_abs(one_out, repo_root),
        "index_dir": _rel_or_abs(index_dir, repo_root),
    }
    return ds_idx, row, w, score, q_count


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
    max_workers: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0
    total_questions = 0
    question_weighted_score = 0.0

    workers = max(1, int(max_workers))
    tasks = [(idx, ds) for idx, ds in enumerate(datasets, start=1)]

    if workers == 1:
        results = [
            _run_dataset_once(
                repo_root=repo_root,
                ds=ds,
                ds_idx=idx,
                tmp_dir=tmp_dir,
                index_cache_dir=index_cache_dir,
                policy_variant=policy_variant,
                policy_strict=policy_strict,
                reuse_index=reuse_index,
                run_tag=run_tag,
            )
            for idx, ds in tasks
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_dataset_once,
                    repo_root=repo_root,
                    ds=ds,
                    ds_idx=idx,
                    tmp_dir=tmp_dir,
                    index_cache_dir=index_cache_dir,
                    policy_variant=policy_variant,
                    policy_strict=policy_strict,
                    reuse_index=reuse_index,
                    run_tag=run_tag,
                ): idx
                for idx, ds in tasks
            }
            for fu in as_completed(futures):
                results.append(fu.result())

    results.sort(key=lambda x: x[0])

    for _idx, row, w, score, q_count in results:
        total_weight += w
        weighted_score += score * w
        total_questions += q_count
        question_weighted_score += score * q_count
        rows.append(row)

    return {
        "datasets": rows,
        "summary": {
            "weighted_score": (weighted_score / total_weight) if total_weight else 0.0,
            "question_weighted_score": (question_weighted_score / total_questions) if total_questions else 0.0,
            "total_questions": total_questions,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp-dir", default="eval/runs")
    ap.add_argument("--index-cache-dir", default="eval/.cache/indexes")
    ap.add_argument("--no-reuse-index", action="store_true")
    ap.add_argument("--policy-variant", default="control")
    ap.add_argument("--policy-strict", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--run-mode", choices=["fail-fast", "best-effort"], default="fail-fast")

    ap.add_argument("--baseline-file")
    ap.add_argument("--max-regressions", type=int)
    ap.add_argument("--max-std-weighted", type=float)
    ap.add_argument("--max-std-question", type=float)
    ap.add_argument("--max-drop-vs-baseline", type=float)
    ap.add_argument("--min-delta", type=float, default=0.0)
    args = ap.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

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
    run_manifest: list[dict[str, Any]] = []

    for i in range(1, args.runs + 1):
        tag = f"run{i}"
        run_start = _now_iso()
        t0 = datetime.now(timezone.utc)
        run_file = out_path if args.runs == 1 else out_path.with_name(f"{out_path.stem}.run{i}{out_path.suffix}")
        manifest_entry: dict[str, Any] = {
            "run": i,
            "tag": tag,
            "start_at": run_start,
            "end_at": None,
            "elapsed_sec": None,
            "status": "unknown",
            "error": None,
            "file": _rel_or_abs(run_file, repo_root),
        }
        try:
            rep = run_benchmark_once(
                repo_root=repo_root,
                datasets=datasets,
                tmp_dir=tmp_dir,
                index_cache_dir=index_cache_dir,
                policy_variant=args.policy_variant,
                policy_strict=bool(args.policy_strict),
                reuse_index=not bool(args.no_reuse_index),
                run_tag=tag,
                max_workers=args.max_workers,
            )
            out_single = {
                "summary": {
                    "benchmark": cfg.get("name", "benchmark"),
                    "datasets": len(rep["datasets"]),
                    "policy_variant": args.policy_variant,
                    "policy_strict": bool(args.policy_strict),
                    "index_cache_dir": _rel_or_abs(index_cache_dir, repo_root),
                    "reuse_index": not bool(args.no_reuse_index),
                    "max_workers": args.max_workers,
                    "run": i,
                    **rep["summary"],
                },
                "datasets": rep["datasets"],
            }
            run_file.parent.mkdir(parents=True, exist_ok=True)
            run_file.write_text(json.dumps(out_single, ensure_ascii=False, indent=2), encoding="utf-8")
            run_reports.append({"run": i, "file": _rel_or_abs(run_file, repo_root), **out_single["summary"]})
            manifest_entry["status"] = "ok"
            manifest_entry["weighted_score"] = out_single["summary"]["weighted_score"]
            manifest_entry["question_weighted_score"] = out_single["summary"]["question_weighted_score"]
            manifest_entry["total_questions"] = out_single["summary"]["total_questions"]
        except Exception as e:  # noqa: BLE001
            manifest_entry["status"] = "infra_error"
            manifest_entry["error"] = str(e)
            if args.run_mode == "fail-fast":
                run_manifest.append(manifest_entry)
                break
        finally:
            t1 = datetime.now(timezone.utc)
            manifest_entry["end_at"] = _now_iso()
            manifest_entry["elapsed_sec"] = round((t1 - t0).total_seconds(), 3)
            if manifest_entry not in run_manifest:
                run_manifest.append(manifest_entry)

    if run_reports:
        ws = [float(r["weighted_score"]) for r in run_reports]
        qws = [float(r["question_weighted_score"]) for r in run_reports]
        final_summary = {
            "benchmark": cfg.get("name", "benchmark"),
            "datasets": len(datasets),
            "policy_variant": args.policy_variant,
            "policy_strict": bool(args.policy_strict),
            "index_cache_dir": _rel_or_abs(index_cache_dir, repo_root),
            "reuse_index": not bool(args.no_reuse_index),
            "max_workers": args.max_workers,
            "runs": args.runs,
            "completed_runs": len(run_reports),
            "failed_runs": len([r for r in run_manifest if r.get("status") != "ok"]),
            "run_mode": args.run_mode,
            "weighted_score": statistics.mean(ws),
            "question_weighted_score": statistics.mean(qws),
            "weighted_score_stats": _stats(ws),
            "question_weighted_score_stats": _stats(qws),
            "total_questions": int(run_reports[0]["total_questions"]),
        }
        datasets_out = json.loads((repo_root / run_reports[0]["file"]).read_text(encoding="utf-8"))["datasets"]
    else:
        final_summary = {
            "benchmark": cfg.get("name", "benchmark"),
            "datasets": len(datasets),
            "policy_variant": args.policy_variant,
            "policy_strict": bool(args.policy_strict),
            "runs": args.runs,
            "completed_runs": 0,
            "failed_runs": len(run_manifest),
            "run_mode": args.run_mode,
        }
        datasets_out = []

    final_output = {
        "summary": final_summary,
        "runs": run_reports,
        "run_manifest": run_manifest,
        "datasets": datasets_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    failed_runs = [r for r in run_manifest if r.get("status") != "ok"]
    if failed_runs:
        print(json.dumps(final_summary, ensure_ascii=False))
        print(f"INFRA_FAIL: {len(failed_runs)} run(s) failed; first={failed_runs[0].get('error')}", file=sys.stderr)
        return EXIT_INFRA_FAIL

    if _is_gate_enabled(args):
        baseline_summary = None
        if args.baseline_file:
            baseline_summary = _load_baseline_summary((repo_root / args.baseline_file).resolve())
        gate_result = evaluate_multi_run_gate(
            candidate_summary=final_summary,
            run_manifest=run_manifest,
            baseline_summary=baseline_summary,
            gate=GateConfig(
                min_delta=float(args.min_delta),
                max_regressions=args.max_regressions,
                max_std_weighted=args.max_std_weighted,
                max_std_question=args.max_std_question,
                max_drop_vs_baseline=args.max_drop_vs_baseline,
            ),
        )
        final_output["gate"] = gate_result
        out_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(final_summary, ensure_ascii=False))
        if not gate_result["passed"]:
            print(f"GATE_FAIL: {'; '.join(gate_result['reasons'])}", file=sys.stderr)
            return EXIT_GATE_FAIL

    print(json.dumps(final_summary, ensure_ascii=False))
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
