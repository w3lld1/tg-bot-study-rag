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

Выходной файл содержит:
- summary по каждому документу
- агрегированный `weighted_score` (по весам датасетов)
- агрегированный `question_weighted_score` (по количеству вопросов)
- хэши PDF и question-set'ов для валидации сравнения

## Принцип

- Улучшаем качество на всех документах одновременно
- Не допускаем оптимизацию под один PDF
- Сравнение кандидата с baseline делается на всём benchmark
