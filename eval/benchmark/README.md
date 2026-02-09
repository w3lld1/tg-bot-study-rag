# Multi-document benchmark v1

Цель: проверять качество агента не на одном документе, а на нескольких разных PDF.

Состав v1:
- `sber-2022` — финансовый отчет (existing baseline)
- `tech-gis-epd` — техническая документация (руководство пользователя)
- `legal-labor-rules` — юридический/кадровый документ (правила внутреннего трудового распорядка)

Локальные PDF (не в git):
- `eval/data/source.pdf`
- `eval/data/tech-gis-epd.pdf`
- `eval/data/legal-labor-rules.pdf`

## Запуск

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-baseline.json
```

Мульти-run + gate (пример локально):

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-pr-summary.json \
  --runs 3 \
  --run-mode best-effort \
  --baseline-file eval/experiments/phaseb-control-multirun-summary.json \
  --max-regressions 1 \
  --max-std-weighted 0.021 \
  --max-std-question 0.02 \
  --max-drop-vs-baseline 0.03 \
  --min-delta -0.01
```

## CI quality gate (PR)

В `benchmark-gate` в CI теперь используется **fresh-run lightweight gate**:

1. Запускается свежий benchmark:
   - `eval/benchmark/run_benchmark.py`
   - `--runs 2`
   - `--max-workers 2`
   - `--policy-variant control`
   - `reuse_index = true` (по умолчанию, `--no-reuse-index` не передаётся)
2. После этого запускается `eval/benchmark/gate_summary.py` по **только что сгенерированному** summary (`eval/runs/benchmark-ci-pr-summary.json`).
3. Baseline используется из `eval/experiments/phaseb-control-multirun-summary.json`, если файл есть.
   - Если baseline отсутствует, gate выполняется без baseline-сравнения, но со стабильностными проверками (std/shape checks), с явным сообщением в логах.

Пороги в CI gate:
- `max_regressions = 1` (только когда baseline доступен)
- `max_std_weighted = 0.021`
- `max_std_question = 0.02`
- `max_drop_vs_baseline = 0.03` (только когда baseline доступен)
- `min_delta = -0.01` (только когда baseline доступен)

Ограничения:
- Gate проверяет качество на облегчённом fresh-run (2 запуска), а не на заранее сохранённом candidate JSON.
- Для ограничения длительности job в CI включён `timeout-minutes`.

Коды выхода для CI:
- `0` — PASS
- `1` — GATE_FAIL (качество/стабильность не прошли пороги)
- `2` — INFRA_FAIL (ошибки запуска run'ов/инфры)

Выходной файл содержит:
- summary по каждому документу
- агрегированный `weighted_score` (по весам датасетов)
- агрегированный `question_weighted_score` (по количеству вопросов)
- хэши PDF и question-set'ов для валидации сравнения
- `run_manifest` по каждому run: `start_at`, `end_at`, `elapsed_sec`, `status`, `error`

Текущий baseline v1 зафиксирован в:
- `eval/baselines/benchmark-v1-baseline-2026-02-09.md`

## Принцип

- Улучшаем качество на всех документах одновременно
- Не допускаем оптимизацию под один PDF
- Сравнение кандидата с baseline делается на всём benchmark
