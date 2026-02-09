#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateConfig:
    min_delta: float = 0.0
    max_regressions: int | None = None
    max_std_weighted: float | None = None
    max_std_question: float | None = None
    max_drop_vs_baseline: float | None = None


def _mean_value(summary: dict[str, Any], key: str) -> float:
    if key in summary and isinstance(summary[key], (int, float)):
        return float(summary[key])
    if key in summary and isinstance(summary[key], dict) and "mean" in summary[key]:
        return float(summary[key]["mean"])
    stats = summary.get(f"{key}_stats")
    if isinstance(stats, dict) and "mean" in stats:
        return float(stats["mean"])
    return 0.0


def _std_value(summary: dict[str, Any], key: str) -> float:
    stats = summary.get(f"{key}_stats")
    if isinstance(stats, dict) and "std" in stats:
        return float(stats["std"])
    if key in summary and isinstance(summary[key], dict) and "std" in summary[key]:
        return float(summary[key]["std"])
    return 0.0


def evaluate_multi_run_gate(
    *,
    candidate_summary: dict[str, Any],
    run_manifest: list[dict[str, Any]] | None,
    gate: GateConfig,
    baseline_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    weighted_std = _std_value(candidate_summary, "weighted_score")
    question_std = _std_value(candidate_summary, "question_weighted_score")

    if gate.max_std_weighted is not None:
        ok = weighted_std <= gate.max_std_weighted
        checks["max_std_weighted"] = {
            "ok": ok,
            "actual": weighted_std,
            "limit": gate.max_std_weighted,
        }
        if not ok:
            reasons.append(f"weighted std {weighted_std:.4f} > {gate.max_std_weighted:.4f}")

    if gate.max_std_question is not None:
        ok = question_std <= gate.max_std_question
        checks["max_std_question"] = {
            "ok": ok,
            "actual": question_std,
            "limit": gate.max_std_question,
        }
        if not ok:
            reasons.append(f"question std {question_std:.4f} > {gate.max_std_question:.4f}")

    if baseline_summary is not None:
        cand_weighted = _mean_value(candidate_summary, "weighted_score")
        base_weighted = _mean_value(baseline_summary, "weighted_score")
        delta = cand_weighted - base_weighted
        checks["baseline_delta"] = {
            "candidate": cand_weighted,
            "baseline": base_weighted,
            "delta": delta,
            "min_delta": gate.min_delta,
            "ok": delta >= gate.min_delta,
        }
        if delta < gate.min_delta:
            reasons.append(f"delta {delta:.4f} < min_delta {gate.min_delta:.4f}")

        if gate.max_drop_vs_baseline is not None:
            drop = base_weighted - cand_weighted
            ok = drop <= gate.max_drop_vs_baseline
            checks["max_drop_vs_baseline"] = {
                "ok": ok,
                "actual_drop": drop,
                "limit": gate.max_drop_vs_baseline,
            }
            if not ok:
                reasons.append(f"drop {drop:.4f} > {gate.max_drop_vs_baseline:.4f}")

        if gate.max_regressions is not None:
            completed_runs = [r for r in (run_manifest or []) if r.get("status") == "ok"]
            regressions = 0
            floor = base_weighted + gate.min_delta
            for r in completed_runs:
                if float(r.get("weighted_score", 0.0)) < floor:
                    regressions += 1
            ok = regressions <= gate.max_regressions
            checks["max_regressions"] = {
                "ok": ok,
                "actual": regressions,
                "limit": gate.max_regressions,
                "floor": floor,
            }
            if not ok:
                reasons.append(f"regressions {regressions} > {gate.max_regressions}")

    return {"passed": not reasons, "reasons": reasons, "checks": checks}
