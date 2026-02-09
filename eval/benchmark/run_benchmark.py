#!/usr/bin/env python3
import argparse
import hashlib
import json
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


def run_one(repo_root: Path, pdf: str, questions: str, out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "eval/run_eval.py",
        "--pdf",
        pdf,
        "--questions",
        questions,
        "--out",
        str(out),
    ]
    env = dict(**{k: v for k, v in dict().items()})
    p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"run_eval failed for {pdf}:\n{p.stderr or p.stdout}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp-dir", default="eval/runs")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    out_path = (repo_root / args.out).resolve()
    tmp_dir = (repo_root / args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])

    rows: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0

    for ds in datasets:
        ds_id = ds["id"]
        w = float(ds.get("weight", 1.0))
        pdf = str((repo_root / ds["pdf"]).resolve())
        questions = str((repo_root / ds["questions"]).resolve())
        one_out = tmp_dir / f"{ds_id}.json"

        report = run_one(repo_root, pdf, questions, one_out)
        score = float(report["summary"]["weighted_score"])

        total_weight += w
        weighted_score += score * w

        rows.append(
            {
                "id": ds_id,
                "kind": ds.get("kind", "unknown"),
                "weight": w,
                "pdf": ds["pdf"],
                "questions": ds["questions"],
                "pdf_sha256": sha256_file(Path(pdf)),
                "questions_sha256": sha256_file(Path(questions)),
                "questions_count": report["summary"].get("questions", 0),
                "weighted_score": score,
                "run": str(one_out.relative_to(repo_root)),
            }
        )

    output = {
        "summary": {
            "benchmark": cfg.get("name", "benchmark"),
            "datasets": len(rows),
            "weighted_score": (weighted_score / total_weight) if total_weight else 0.0,
        },
        "datasets": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
