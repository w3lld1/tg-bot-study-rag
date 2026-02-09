# Phase B (start): policy table + A/B variant hooks

## Goal
Move retrieval/second-pass policy from hardcoded constants in `agent.py` to explicit policy table with A/B variant toggles.

## Implemented
- New module: `ragbot/policy.py`
  - `get_retrieval_policy(settings, intent)`
  - `get_second_pass_overrides(settings, intent)`
  - variants: `control`, `ab_retrieval_v1`
- `ragbot/agent.py`
  - multiquery trigger now uses policy-driven `cov_threshold` and `force_multiquery`
  - second-pass overrides now come from policy helper
  - policy variant and threshold added to trace
- Settings updated:
  - `policy_variant` in `app.py` and `eval/run_eval.py`

## Notes
- Default variant is `control` and should preserve existing behavior.
- Next step: run benchmark for `control` (sanity) + `ab_retrieval_v1` and compare mean/std across multiple runs.
