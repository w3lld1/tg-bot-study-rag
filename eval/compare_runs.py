#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("summary", {}) or {}


def _get_hashes(run: dict[str, Any]) -> tuple[str | None, str | None]:
    s = _summary(run)
    return s.get("pdf_sha256"), s.get("questions_sha256")


def collect_changes(base: dict[str, Any], cand: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    bmap = {r["id"]: r for r in base.get("results", [])}
    cmap = {r["id"]: r for r in cand.get("results", [])}

    changed: list[tuple[str, float, float, float]] = []
    for qid in sorted(set(bmap) & set(cmap)):
        bs = float(bmap[qid]["metrics"]["score"])
        cs = float(cmap[qid]["metrics"]["score"])
        if abs(cs - bs) > 1e-9:
            changed.append((qid, bs, cs, cs - bs))
    return changed


def gate(
    base: dict[str, Any],
    cand: dict[str, Any],
    *,
    min_delta: float,
    max_regressions: int,
) -> dict[str, Any]:
    bscore = float(_summary(base).get("weighted_score", 0.0))
    cscore = float(_summary(cand).get("weighted_score", 0.0))
    delta = cscore - bscore

    changed = collect_changes(base, cand)
    regressions = [x for x in changed if x[3] < 0]
    improvements = [x for x in changed if x[3] > 0]

    pass_delta = delta >= min_delta
    pass_regressions = len(regressions) <= max_regressions
    passed = pass_delta and pass_regressions

    return {
        "base": bscore,
        "candidate": cscore,
        "delta": delta,
        "changed": len(changed),
        "improvements": len(improvements),
        "regressions": len(regressions),
        "min_delta": min_delta,
        "max_regressions": max_regressions,
        "passed": passed,
        "changes": changed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--max-regressions", type=int, default=0)
    ap.add_argument("--strict-hashes", action="store_true", default=True)
    ap.add_argument("--no-strict-hashes", dest="strict_hashes", action="store_false")
    args = ap.parse_args()

    base = load(Path(args.base))
    cand = load(Path(args.cand))

    bpdf, bq = _get_hashes(base)
    cpdf, cq = _get_hashes(cand)
    if args.strict_hashes and (bpdf != cpdf or bq != cq):
        print(
            json.dumps(
                {
                    "error": "hash_mismatch",
                    "base_pdf_sha256": bpdf,
                    "cand_pdf_sha256": cpdf,
                    "base_questions_sha256": bq,
                    "cand_questions_sha256": cq,
                },
                ensure_ascii=False,
            )
        )
        sys.exit(2)

    out = gate(
        base,
        cand,
        min_delta=args.min_delta,
        max_regressions=args.max_regressions,
    )

    print(
        json.dumps(
            {
                "base": out["base"],
                "candidate": out["candidate"],
                "delta": out["delta"],
                "changed": out["changed"],
                "improvements": out["improvements"],
                "regressions": out["regressions"],
                "min_delta": out["min_delta"],
                "max_regressions": out["max_regressions"],
                "passed": out["passed"],
            },
            ensure_ascii=False,
        )
    )

    for qid, bs, cs, d in out["changes"]:
        sign = "+" if d >= 0 else ""
        print(f"{qid}: {bs:.3f} -> {cs:.3f} ({sign}{d:.3f})")

    if not out["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
