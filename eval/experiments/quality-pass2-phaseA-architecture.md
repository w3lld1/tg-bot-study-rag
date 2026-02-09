# Phase A: architecture instrumentation + stage split (no policy change)

## Goal
Improve debuggability and architectural clarity without introducing new retrieval/generation heuristics.

## Scope
- Split `BestStableRAGAgent.ask()` into explicit stages:
  - intent
  - retrieve
  - answer
  - validation
  - second-pass decision
  - repair
- Add structured per-question trace in memory (`agent.get_last_trace()`)
- Persist trace into eval output (`eval/run_eval.py` adds `trace` in each result row)

## Non-goals
- No new quality-tuning heuristics.
- No benchmark-targeted overfitting.

## Why
Recent run-to-run variance is hard to diagnose from only final score + short debug line. Trace-first architecture should make regressions explainable and reduce blind tuning loops.
