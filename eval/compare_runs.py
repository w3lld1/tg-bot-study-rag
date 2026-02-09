#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    args = ap.parse_args()

    base = load(Path(args.base))
    cand = load(Path(args.cand))

    bscore = float(base["summary"]["weighted_score"])
    cscore = float(cand["summary"]["weighted_score"])
    delta = cscore - bscore

    bmap = {r["id"]: r for r in base.get("results", [])}
    cmap = {r["id"]: r for r in cand.get("results", [])}

    changed = []
    for qid in sorted(set(bmap) & set(cmap)):
        bs = float(bmap[qid]["metrics"]["score"])
        cs = float(cmap[qid]["metrics"]["score"])
        if abs(cs - bs) > 1e-9:
            changed.append((qid, bs, cs, cs - bs))

    print(json.dumps({
        "base": bscore,
        "candidate": cscore,
        "delta": delta,
        "changed": len(changed),
    }, ensure_ascii=False))

    for qid, bs, cs, d in changed:
        sign = "+" if d >= 0 else ""
        print(f"{qid}: {bs:.3f} -> {cs:.3f} ({sign}{d:.3f})")


if __name__ == "__main__":
    main()
