# Quality pass #4: guard для intent `procedure` (итерация 2)

## Гипотеза
После первой версии guard (`"как" -> procedure` только с action-маркерами) остаётся риск FN для реальных procedural-вопросов с глаголами вроде `подключить/войти/проверить/открыть`.

Ожидаемый эффект:
- лучшее разделение `procedure` vs `default/numbers_and_dates` на реальных user phrasing;
- сохранение положительного uplift по benchmark v1;
- uplift по `weighted_score` не ниже `+0.1 п.п.` (требование Никиты).

## Изменения
- `ragbot/core_logic.py`
  - расширен список action-глаголов для ветки `"как" -> procedure`:
    - `подключ`, `откры`, `войти`, `провер`, `включ`, `отключ`, `измен`
- `tests/test_app_core_logic.py`
  - расширено покрытие intent-тестов (позитив/негатив для `procedure`):
    - позитив: `как подключить модуль`, `как войти в систему`, `как проверить статус`, `как открыть форму заявки`
    - негатив: `как устроена подсистема`, `как формулируется миссия`

## Проверки
- Tests: `PYTHONPATH=. .venv/bin/pytest -q` → `91 passed`.

- Benchmark v1 (control, strict, runs=3, max-workers=2, reuse index):

```bash
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1 PYTHONPATH=. .venv/bin/python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --policy-variant control \
  --policy-strict \
  --runs 3 \
  --max-workers 2 \
  --out eval/runs/benchmark-v1-pr19-pass4b-runs3.json
```

## Сравнение с baseline
Baseline: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`

- **weighted_score**
  - baseline mean: `0.733313`
  - candidate mean: `0.742327`
  - uplift (abs): `+0.009014`
  - uplift (п.п.): `+0.901`
  - baseline std: `0.009071`
  - candidate std: `0.007427` (стабильность улучшилась)
- **question_weighted_score**
  - baseline mean: `0.701443`
  - candidate mean: `0.710896`
  - uplift (abs): `+0.009453`
  - baseline std: `0.006698`
  - candidate std: `0.006633` (не ухудшился)

## Drift intent-распределения (до/после)
Проверено на всех 48 benchmark-вопросах через fast-classifier до/после изменения списка action-глаголов:

- `procedure`: `1 -> 5` (`+4`)
- `numbers_and_dates`: `20 -> 16` (`-4`)
- `default`: `22 -> 22` (`0`)
- `requirements`: `3 -> 3` (`0`)
- `definition`: `2 -> 2` (`0`)

Интерпретация: сдвиг локальный и ожидаемый — часть "как + действие" вопросов ушла из `numbers_and_dates` в `procedure`; остальные intent-классы не поплыли.

## Вывод
- Существенные замечания злого TL из предыдущего review закрыты:
  - расширен хрупкий лексический список action-глаголов;
  - добавлено более широкое intent-покрытие тестами;
  - добавлен отчёт по drift intent-распределения.
- Прирост относительно baseline остаётся положительным и > `0.1 п.п.`.

## Артефакты
- `eval/runs/benchmark-v1-pr19-pass4b-runs3.json`
- `eval/runs/benchmark-v1-pr19-pass4b-runs3.run1.json`
- `eval/runs/benchmark-v1-pr19-pass4b-runs3.run2.json`
- `eval/runs/benchmark-v1-pr19-pass4b-runs3.run3.json`
