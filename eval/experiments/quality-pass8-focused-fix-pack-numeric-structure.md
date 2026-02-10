# Quality Pass 8 — Focused Fix Pack (numeric finalization + structure extraction)

## Scope
TL+DS recommendation, strict scope:
1. Numeric answer finalizer for `numbers_and_dates`:
   - final normalized answer (`value + unit`),
   - page ref + short supporting quote,
   - explicit `В документе не найдено.` when uncertain.
2. Structure extractor path for list-like questions (`какие разделы/какие комитеты/список`).
3. No large refactors outside scope.
4. Added tests (>=3 for each mode).

## Code changes (targeted)
- `ragbot/core_logic.py`
  - Added intent `structure_list`.
  - Added `finalize_numeric_answer()` with strict numeric normalization.
  - Added `extract_structure_entities()` and `format_structure_answer()`.
  - Extended `detect_intent_fast()` with structure/list triggers.
- `ragbot/agent.py`
  - Added structure path in `_answer_stage()`.
  - Applied numeric finalizer in `ask()` for numeric intent.
  - Prevented numeric/structure fallback from returning "релевантные фрагменты" on not-found.
- `ragbot/chains.py`
  - Extended LLM intent enum with `structure_list`.
- `ragbot/policy.py`
  - Added `structure_list` to allowed intents and policy tables.

## Tests added/updated
- `tests/test_app_core_logic.py`
  - Numeric finalizer: 3 new tests.
  - Structure extractor: 3 new tests.
  - Intent detection for structure queries.
- `tests/test_agent_planner_behavior.py`
  - Added structure-path behavior test.

## Runs

### 1) Unit tests
```bash
pytest -q
```
Result: **109 passed**.

### 2) Sber eval dump (24 questions)
```bash
PYTHONPATH=. .venv/bin/python eval/run_eval.py \
  --pdf eval/data/source.pdf \
  --questions eval/questions.jsonl \
  --out eval/runs/sber-2022-focused-fix-pack.json \
  --policy-variant control \
  --policy-strict \
  --index-dir eval/.cache/indexes/sber-2022-8a859081ed75
```
Result summary:
- weighted_score = **0.5975**

Baseline for comparison (existing):
- `eval/runs/sber-2022.json`: weighted_score = **0.6025**
- delta = **-0.0050**

### 3) Benchmark v1 control strict, runs=3, max-workers=2, reuse index
```bash
PYTHONPATH=. .venv/bin/python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-focused-fix-pack-runs3.json \
  --policy-variant control \
  --policy-strict \
  --runs 3 \
  --max-workers 2
```
Result summary:
- weighted_score = **0.7497294439**
- question_weighted_score = **0.7116720829**

Comparisons:
- vs baseline after phase C (`eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`):
  - weighted delta: **+0.0164164624**
  - question-weighted delta: **+0.0102290135**
- vs current main snapshot (`eval/runs/benchmark-v1-pr21-latest-runs3.json`):
  - weighted delta: **-0.0436145788**
  - question-weighted delta: **-0.0603070880**

## Mini-report: 8 weak Sber questions (before/after)
(Tracked set: q003–q008 numeric weak set + q023–q024 structure set)

| id | before score | after score | delta | before | after |
|---|---:|---:|---:|---|---|
| q003 | 0.10 | 0.30 | +0.20 | fail | fail |
| q004 | 0.10 | 0.30 | +0.20 | fail | fail |
| q005 | 0.10 | 0.30 | +0.20 | fail | fail |
| q006 | 0.10 | 0.30 | +0.20 | fail | fail |
| q007 | 0.10 | 0.65 | +0.55 | fail | partial |
| q008 | 0.10 | 0.65 | +0.55 | fail | partial |
| q023 | 0.30 | 0.30 | +0.00 | fail | fail |
| q024 | 0.30 | 0.30 | +0.00 | fail | fail |

## Notes
- Numeric finalizer now guarantees deterministic final format for numeric intent, including strict not-found fallback.
- Structure path is now isolated and extractive (list-focused), but this pass did not yet improve q023/q024 scoring in eval set.
