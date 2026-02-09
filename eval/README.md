# Eval harness for answer quality

Цель: фиксировать baseline качества ответов и сравнивать каждый инкремент.

## 1) Куда класть PDF

Рекомендуемо (локально, без коммита в git):

- `eval/data/source.pdf`

Файл добавлен в `.gitignore` через маску `eval/data/`.

## 2) Формат вопросов

Файл: `eval/questions.jsonl`

Одна строка = один JSON-объект:

```json
{"id":"q001","question":"Какая выручка за 2023 год?","must_include":["2023"],"must_not_include":["XX","??"],"weight":1.0}
```

Поля:
- `id` (str, обязательно)
- `question` (str, обязательно)
- `must_include` (list[str], опционально; каждое значение обязательно к наличию)
- `must_include_any` (list[list[str]|str], опционально; OR-группы эквивалентных формулировок)
- `must_not_include` (list[str], опционально)
- `require_citation` (bool, опционально; если `true`, нужен паттерн ссылки на страницу `стр. N` — в скобках или без)
- `weight` (float, опционально, по умолчанию 1.0)

## 3) Запуск baseline

```bash
python eval/run_eval.py \
  --pdf eval/data/source.pdf \
  --questions eval/questions.jsonl \
  --out eval/runs/baseline.json
```

## 4) Сравнение двух прогонов

```bash
python eval/compare_runs.py \
  --base eval/runs/baseline.json \
  --cand eval/runs/candidate.json
```

По умолчанию скрипт проверяет совместимость прогонов по:
- `pdf_sha256`
- `questions_sha256`

Если хэши не совпадают, сравнение завершается с ошибкой.

## 5) Формула расчёта score

Для каждого вопроса:

- `include_rate = include_hits / include_total`
  - `include_total` = количество обязательных групп (`must_include` + `must_include_any`)
  - если групп нет, `include_rate = 1.0`
- `safe_ok = 1.0`, если в ответе нет попаданий по `must_not_include`, иначе `0.0`
- `base_score = 0.7 * include_rate + 0.3 * safe_ok`
- если `require_citation=true` и нет ссылки вида `(стр. N...)`, применяется штраф:
  - `citation_penalty = 0.2`
- итог по вопросу:
  - **`question_score = max(0.0, base_score - citation_penalty)`**

Итоговый score прогона:

- **`weighted_score = sum(question_score_i * weight_i) / sum(weight_i)`**

Дополнительно в `summary` фиксируются:
- `pdf_sha256`
- `questions_sha256`

## 6) Multi-document benchmark (v1)

Для проверки качества на нескольких документах:

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-baseline.json
```

`summary.weighted_score` в этом отчёте — агрегированный benchmark-score по всем датасетам.

## 7) Процесс в PR

1. До изменения кода: прогон baseline.
2. После изменения: прогон candidate.
3. Прогнать quality gate:

```bash
python eval/compare_runs.py \
  --base eval/runs/baseline.json \
  --cand eval/runs/candidate.json \
  --min-delta 0.0 \
  --max-regressions 0
```

- exit code `0` — gate passed
- exit code `1` — gate failed (delta/regressions)
- exit code `2` — incompatible runs (hash mismatch)

4. Если gate не passed — PR не мержим.
