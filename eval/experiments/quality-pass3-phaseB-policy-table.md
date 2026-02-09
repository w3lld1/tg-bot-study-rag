# Phase B (start): policy table + A/B variant hooks

## Goal
Move retrieval/second-pass policy from hardcoded constants in `agent.py` to explicit policy table with A/B variant toggles.

## Implemented
- New module: `ragbot/policy.py`
  - `get_retrieval_policy(settings, intent)`
  - `get_second_pass_overrides(settings, intent)`
  - `should_trigger_multiquery(...)` (behavioral decision helper)
  - variants: `control`, `ab_retrieval_v1`
- `ragbot/agent.py`
  - multiquery trigger now uses policy-driven `cov_threshold` and `force_multiquery`
  - second-pass overrides now come from policy helper
  - trace includes `policy_variant`, `policy_rule_id`, and `multiquery_decision`
- Settings/overrides updated:
  - `policy_variant` + `policy_strict` in `app.py` (env: `POLICY_VARIANT`, `POLICY_STRICT`)
  - `eval/run_eval.py` now supports `--policy-variant` and `--policy-strict`
  - `eval/benchmark/run_benchmark.py` passes these options through

## AB thresholds rationale
- `summary`/`definition`: increased threshold (`0.60`) to trigger multiquery earlier on broad semantic queries.
- `compare`/`procedure`/`requirements`: moderate increase (`0.56`) to improve recall with lower noise than summary.
- `default`: mild increase (`0.58`) as balanced fallback.
- `numbers_and_dates`: unchanged (`0.50`, force multiquery true) because numeric flow already has dedicated expansion.

## Multi-run benchmark (v1, strict mode)
### control (n=3)
- weighted_score: mean `0.7271022263071896`, std `0.0029224824008146823`, min `0.723500889939309`, max `0.7306590657096171`
- question_weighted_score: mean `0.6959516697303922`, std `0.0021918618006110564`, min `0.6932506674544818`, max `0.698619299282213`

### ab_retrieval_v1 (n=3)
- weighted_score: mean `0.7348412754524888`, std `0.020091499029337215`, min `0.7067207644275659`, max `0.7524271719905912`
- question_weighted_score: mean `0.7052815976150076`, std `0.01433256983250387`, min `0.6859540348591361`, max `0.7202338405314048`

## Interpretation
- AB variant improves mean scores vs control, but variance is much higher.
- Reference baseline from main (pass #1):
  - weighted_score `0.7453758899393091`
  - question_weighted_score `0.7096569174544819`
- Control mean is below that reference baseline; AB mean is closer but still below on average.
- Recommendation: keep `control` as default for stability; use AB behind explicit experiment flag until variance is reduced.

## Speed-up Step 1 implemented
- `eval/run_eval.py` now supports reusable index mode:
  - `--index-dir` (persistent index path)
  - `--no-reuse-index` (force rebuild)
- `eval/benchmark/run_benchmark.py` now passes reusable index paths per dataset via `--index-cache-dir`.
- This removes repeated ingest work across repeated benchmark runs for same dataset.
