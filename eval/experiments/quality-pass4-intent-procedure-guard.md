# Quality pass #4: guard для intent `procedure`

## Гипотеза
Слишком широкое правило `"как " -> procedure` ошибочно уводит часть дефиниционных/общих вопросов в procedural-цепочку.

Мини-правка: считать `procedure`, только если:
- есть procedural-маркеры (`шаг`, `процедур`, `инструкц`, ...), **или**
- `как` встречается вместе с глаголами действия (`сдел`, `выполн`, `оформ`, `подат`, `заполн`, ...).

## Изменения
- `ragbot/core_logic.py`: сужен fast-path для `procedure`.
- `tests/test_app_core_logic.py`: добавлен кейс `"как называется подсистема" -> default`.

## Проверки
- Tests: `PYTHONPATH=. .venv/bin/pytest -q` → `85 passed`.
- Benchmark v1 (control, strict, runs=3, max-workers=2, reuse index):

```bash
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1 PYTHONPATH=. .venv/bin/python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --policy-variant control \
  --policy-strict \
  --runs 3 \
  --max-workers 2 \
  --out eval/runs/benchmark-v1-hyp1-runs3.json
```

## Сравнение с baseline
Baseline: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`

- **weighted_score**
  - baseline: mean `0.733313`, std `0.009071`, min `0.723501`, max `0.745376`
  - candidate: mean `0.745799`, std `0.007966`, min `0.736261`, max `0.755761`
  - uplift (mean): `+0.012486`
- **question_weighted_score**
  - baseline: mean `0.701443`, std `0.006698`, min `0.693251`, max `0.709657`
  - candidate: mean `0.712571`, std `0.009379`, min `0.702821`, max `0.725234`

Критерий остановки выполнен:
- mean uplift по `weighted_score` > 0
- std по `weighted_score` не ухудшился (даже улучшился: `-0.001105`)

## Артефакты
- `eval/runs/benchmark-v1-hyp1-runs3.json`
- `eval/runs/benchmark-v1-hyp1-runs3.run1.json`
- `eval/runs/benchmark-v1-hyp1-runs3.run2.json`
- `eval/runs/benchmark-v1-hyp1-runs3.run3.json`
