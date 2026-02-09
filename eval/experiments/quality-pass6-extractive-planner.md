# Quality pass6 — extractive-first planner + lexical constraints

## Цель итерации
Сделать **крупный архитектурный сдвиг**: перейти от «чистого generation по контексту» к pipeline:
1) extractive-first planner,
2) lexical constraint layer,
3) synthesis поверх extractive-опор.

## Что реализовано

### 1) Extractive-first planner (`ragbot/core_logic.py`)
Добавлен `build_extractive_plan(question, context, intent, max_items=8)`:
- парсит контекст на блоки `[стр. X] ...` и предложения,
- скорит кандидаты по:
  - overlap с токенами вопроса,
  - наличие чисел/дат,
  - intent-aware bonus (requirements/procedure),
- выбирает top-N evidence с dedup,
- формирует evidence bullets: `- (стр. X) "цитата"`.

### 2) Lexical constraint layer (`ragbot/core_logic.py`)
В planner добавлен слой ограничений:
- required tokens (из вопроса),
- required numbers/dates (из вопроса),
- covered/missing по evidence.

Формируется `lexical_report` с покрытием и пробелами.

### 3) Synthesis поверх extractive-опор (`ragbot/agent.py`)
- `ask()` теперь строит `planner_stage` до генерации ответа.
- `_answer_stage(..., plan=...)` использует `plan["synthesis_context"]` для summary/compare/numbers/default chain'ов.
- `citation_only` оставлен на raw-context, чтобы не ломать поведение «только цитаты».
- В trace добавлен `planner_stage` + флаг `planner_used`.

## Тесты

### Unit
- `test_build_extractive_plan_collects_quotes_and_lexical_coverage`
- `test_build_extractive_plan_synthesis_context_contains_layers`

### Поведенческий / интеграционный
- `tests/test_agent_planner_behavior.py`
  - проверяет, что default intent реально получает `synthesis_context`,
  - проверяет, что `citation_only` остается на raw-context.

Итог локально: `95 passed`.

## Benchmark v1 (control strict, runs=3, max-workers=2, reuse index)
Команда:
```bash
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1 PYTHONPATH=. .venv/bin/python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-pr20-pass6-runs3.json \
  --policy-variant control \
  --policy-strict \
  --runs 3 \
  --max-workers 2
```

Baseline: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`
Candidate: `eval/runs/benchmark-v1-pr20-pass6-runs3.json`

### Weighted score
- baseline mean: **0.7333129814**
- candidate mean: **0.7948186230**
- delta mean: **+0.0615056415**

### Delta stats (candidate - baseline, weighted_score_stats)
- mean delta: **+0.0615056415**
- std delta: **-0.0052041560**
- min delta: **+0.0658489830**
- max delta: **+0.0521771080**

### Question-weighted score
- baseline mean: **0.7014430694**
- candidate mean: **0.7730851211**
- delta mean: **+0.0716420517**

## Вывод
Итерация действительно крупная и даёт заметный аплифт, но **не дотягивает до целевого > +0.1** относительно baseline after PR #18. Дальше нужен ещё один крупный шаг (например, stricter constrained decoding / answer schema enforcement + stronger benchmark-aware planner).