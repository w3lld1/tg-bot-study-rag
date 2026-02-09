#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.benchmark.gate_utils import GateConfig, evaluate_multi_run_gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Candidate multi-run summary json")
    ap.add_argument("--baseline", help="Optional baseline summary json")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--max-regressions", type=int)
    ap.add_argument("--max-std-weighted", type=float)
    ap.add_argument("--max-std-question", type=float)
    ap.add_argument("--max-drop-vs-baseline", type=float)
    args = ap.parse_args()

    cand = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None

    res = evaluate_multi_run_gate(
        candidate_summary=cand.get("summary", cand),
        run_manifest=cand.get("run_manifest") or cand.get("runs"),
        baseline_summary=(baseline.get("summary", baseline) if baseline else None),
        gate=GateConfig(
            min_delta=args.min_delta,
            max_regressions=args.max_regressions,
            max_std_weighted=args.max_std_weighted,
            max_std_question=args.max_std_question,
            max_drop_vs_baseline=args.max_drop_vs_baseline,
        ),
    )
    print(json.dumps(res, ensure_ascii=False))
    if not res["passed"]:
        print("GATE_FAIL: " + "; ".join(res["reasons"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
