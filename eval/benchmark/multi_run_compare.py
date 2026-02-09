#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load_scores(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        s = d["summary"]
        rows.append(
            {
                "path": str(p),
                "weighted_score": float(s["weighted_score"]),
                "question_weighted_score": float(s["question_weighted_score"]),
                "policy_variant": s.get("policy_variant", "unknown"),
            }
        )
    return rows


def stats(vals):
    return {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    rows = load_scores(args.runs)
    ws = [r["weighted_score"] for r in rows]
    qws = [r["question_weighted_score"] for r in rows]

    out = {
        "label": args.label,
        "n": len(rows),
        "weighted_score": stats(ws),
        "question_weighted_score": stats(qws),
        "runs": rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
