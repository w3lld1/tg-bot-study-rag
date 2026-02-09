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
- `must_include` (list[str], опционально)
- `must_not_include` (list[str], опционально)
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

## 5) Процесс в PR

1. До изменения кода: прогон baseline.
2. После изменения: прогон candidate.
3. `compare_runs.py` должен показать улучшение/не ухудшение.
4. Если ухудшение — PR не мержим.
