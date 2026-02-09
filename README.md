# tg-bot-study-rag

Короткий гайд: как с нуля поднять бота и прогнать `eval` / `benchmark`.

## 1) Требования / окружение

- OS: Linux/macOS (на Windows лучше через WSL)
- Python: **3.11+** (рекомендовано 3.12)
- Доступ к Telegram Bot API
- Доступ к GigaChat API (ключ)

Проверка:

```bash
python3 --version
```

## 2) Установка зависимостей

```bash
cd /home/faststepbyme_gmail_com/.openclaw/workspace/tg-bot-study-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3) Переменные окружения и где взять ключи

Минимально нужны:

- `BOT_TOKEN` — токен Telegram-бота из **@BotFather**
- `GIGACHAT_API_KEY` — ключ из кабинета/консоли GigaChat API

Опционально:

- `DATA_DIR` (по умолчанию `./data`)
- `LOG_LEVEL` (по умолчанию `INFO`)
- `POLICY_VARIANT` (`control` или `ab_retrieval_v1`)
- `POLICY_STRICT` (`1/true` для strict-режима)

Пример:

```bash
export BOT_TOKEN="..."
export GIGACHAT_API_KEY="..."
export DATA_DIR="./data"
export POLICY_VARIANT="control"
export POLICY_STRICT="0"
```

> В `app.py` есть fallback в Google Secret Manager, но для локального старта проще и быстрее использовать env-переменные.

## 4) Запуск бота

```bash
source .venv/bin/activate
python app.py
```

Что делает бот:
- принимает PDF от пользователя,
- строит индекс,
- отвечает на вопросы по документу.

## 5) Как прогонять eval

### Подготовка

- PDF-файл (например `eval/data/source.pdf`)
- вопросы в формате JSONL (`eval/questions.jsonl`)

Пример запуска:

```bash
source .venv/bin/activate
PYTHONPATH=. python eval/run_eval.py \
  --pdf eval/data/source.pdf \
  --questions eval/questions.jsonl \
  --out eval/runs/baseline.json \
  --policy-variant control
```

С strict-режимом и переиспользованием индекса:

```bash
PYTHONPATH=. python eval/run_eval.py \
  --pdf eval/data/source.pdf \
  --questions eval/questions.jsonl \
  --out eval/runs/candidate.json \
  --policy-variant ab_retrieval_v1 \
  --policy-strict \
  --index-dir eval/.cache/indexes/source \
  --no-reuse-index
```

## 6) Как прогонять benchmark

Основной скрипт:

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1.json
```

### Новые флаги benchmark

- `--policy-variant` — вариант политики (`control` / `ab_retrieval_v1`)
- `--policy-strict` — strict-проверка policy
- `--index-cache-dir` — папка кэша индексов (по умолчанию `eval/.cache/indexes`)
- `--runs` — число повторных прогонов для multi-run статистики
- `--max-workers` — параллелизм по датасетам
- `--no-reuse-index` — не переиспользовать индекс, пересоздавать заново

### Примеры команд (control / ab_retrieval_v1)

`control`:

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-control.json \
  --policy-variant control \
  --runs 3 \
  --max-workers 2
```

`ab_retrieval_v1` (strict + отдельный кэш + без reuse):

```bash
PYTHONPATH=. python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-ab.json \
  --policy-variant ab_retrieval_v1 \
  --policy-strict \
  --index-cache-dir eval/.cache/indexes-ab \
  --runs 3 \
  --max-workers 2 \
  --no-reuse-index
```

## 7) Что в выходных JSON и как читать метрики

### `eval/run_eval.py` (`eval/runs/*.json`)

- `summary.weighted_score` — главная метрика качества (0..1, выше лучше)
- `summary.questions` — число вопросов
- `summary.policy_variant`, `summary.policy_strict` — с какими настройками считали
- `summary.pdf_sha256`, `summary.questions_sha256` — контроль совместимости прогонов
- `results[]` — детализация по каждому вопросу:
  - `answer`, `trace`,
  - `metrics.score`, `metrics.include_rate`, `metrics.exclude_hits`, `metrics.citation_ok`.

### `eval/benchmark/run_benchmark.py` (`eval/runs/benchmark-*.json`)

- `summary.weighted_score` — среднее по датасетам с их весами (`weight`)
- `summary.question_weighted_score` — среднее с весом по числу вопросов
- `summary.total_questions` — всего вопросов в benchmark
- `summary.runs` + `*_stats` — появляются при `--runs > 1` (mean/std/min/max)
- `datasets[]` — метрики по каждому датасету:
  - `weighted_score`, `questions_count`,
  - `pdf_sha256`, `questions_sha256`,
  - `run` (путь к артефакту одиночного eval),
  - `index_dir`.

Практика чтения:
- сначала сравнивайте `summary.weighted_score`,
- затем проверяйте просадки в `datasets[]`,
- потом разбирайте проблемные вопросы в соответствующих `run`-файлах.

## 8) Сравнение baseline/candidate

```bash
PYTHONPATH=. python eval/compare_runs.py \
  --base eval/runs/baseline.json \
  --cand eval/runs/candidate.json \
  --min-delta 0.0 \
  --max-regressions 0
```

Коды выхода:
- `0` — ок,
- `1` — quality gate не пройден,
- `2` — прогоны несовместимы (обычно из-за разных SHA входных данных).
