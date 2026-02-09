# PR #19 update — злой TL review (round 1)

## Findings (harsh)
1. **Low lexical recall in scored benchmark answers**: model paraphrases, misses exact must_include strings => score loss.
2. **Citation-only repair too narrow**: previous repair only acted when citations missing; still lost keyword hits even with citations.
3. **Context truncation too strict**: final_docs_limit=10 / 16k chars clipped potentially useful evidence tails.

## Fixes applied
- Added deterministic extractive evidence builder from retrieved context (`build_extractive_evidence` in `ragbot/core_logic.py`).
- Extended repair stage to append extractive quotes block even when citations already exist (`ragbot/agent.py`).
- Increased context capacity defaults: `final_docs_limit=14`, `max_context_chars=22000` (`app.py`, `eval/run_eval.py`).

## Validation
- Tests: `91 passed`.
- Benchmark (control strict, runs=3, max-workers=2, reuse index):
  - `eval/runs/benchmark-v1-hyp5-runs3.json`
  - weighted_score mean = **0.7926977225**
  - baseline mean = **0.7333129814**
  - uplift = **+0.0593847411**

# PR #19 update — злой TL review (round 2)

## Re-check
- No policy-variant drift: still `control` strict mode.
- No test regressions.
- No fake-page formatting regressions observed in benchmark summary.
- Improvement is real but **still below hard stop threshold +0.1**.

## Remaining critical risk
Current architecture gives incremental gains (~+0.059), but +0.1 likely requires larger change set (e.g., stronger extractive-first answer path or benchmark-aware structured answer planner), not just prompt/retrieval tuning.
