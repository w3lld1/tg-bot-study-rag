# Quality pass 7: Hierarchical Context Compression (planner-compatible)

## Goal
Дожать качество benchmark v1 после phase C за счёт **крупной гипотезы**: уменьшить шум в контексте генерации через иерархическую компрессию, но сохранить совместимость с extractive planner из pass6.

## Hypothesis
Даже при хорошем retrieval LLM теряет точность из-за длинного/шумного RAW-CONTEXT. Если перед synthesis построить **двухслойный контекст**:
1) extractive evidence (как было в pass6),
2) hierarchical context (top sections + top sentences per section),
то модель будет:
- реже галлюцинировать в summary/compare/default;
- лучше удерживать числа/термины вопроса;
- стабильнее ссылаться на релевантные страницы.

## Changes
- `ragbot/core_logic.py`
  - добавлены:
    - `_score_sentence(...)`
    - `_intent_phrase_bonus(...)`
    - `_build_hierarchical_context(...)`
  - `build_extractive_plan(...)` расширен параметрами:
    - `hierarchical_max_sections`
    - `hierarchical_max_sentences_per_section`
  - в `synthesis_context` добавлен блок `HIERARCHICAL-CONTEXT`.
  - в выходе planner добавлен `hierarchical_context.report` для трассировки.

- `ragbot/agent.py`
  - planner получает новые конфиги из settings.
  - trace/planner_stage теперь включает `hierarchical_report`.

- `app.py`
  - добавлены новые planner-настройки в `Settings`.
  - добавлено чтение env `ALLOW_DANGEROUS_FAISS_DESERIALIZATION` для совместимости с reuse-index.

- `eval/run_eval.py`
  - добавлено чтение env `ALLOW_DANGEROUS_FAISS_DESERIALIZATION` в `EvalSettings`.

## Tests
- Обновлены/добавлены тесты в `tests/test_app_core_logic.py`:
  - наличие слоя `HIERARCHICAL-CONTEXT` в synthesis-context;
  - проверка сборки `hierarchical_context.report` и выбора релевантной страницы.

## Benchmark protocol
- benchmark v1
- policy: `control strict`
- runs=3
- max-workers=2
- reuse index (default)
- baseline: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`

## Current benchmark status
- В этой итерации получены 2 строгих run (run1/run2), оба с одинаковым score.
- Полный runs=3 требует отдельного дожима стабильности запуска (инфра/процессный хвост), отмечено как риск в PR.
