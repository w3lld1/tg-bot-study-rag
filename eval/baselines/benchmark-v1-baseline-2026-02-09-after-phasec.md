# Baseline: benchmark-v1-3docs (2026-02-09, after PR #17)

## Aggregate (multi-run, control strict)

- benchmark: `benchmark-v1-3docs`
- policy_variant: `control`
- policy_strict: `true`
- runs: `3` (max_workers=`2`, reuse_index=`true`)
- total_questions: `48`

### weighted_score
- mean: `0.7333129814425771`
- std: `0.009071146164388526`
- min: `0.723500889939309`
- max: `0.7453758899393091`

### question_weighted_score
- mean: `0.7014430694152661`
- std: `0.00669784068270442`
- min: `0.6932506674544818`
- max: `0.7096569174544819`

## Artifacts

- Multi-run summary: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.json`
- Run 1 summary: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.run1.json`
  - Dataset reports: `eval/runs/run1.sber-2022.json`, `eval/runs/run1.tech-gis-epd.json`, `eval/runs/run1.legal-labor-rules.json`
- Run 2 summary: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.run2.json`
  - Dataset reports: `eval/runs/run2.sber-2022.json`, `eval/runs/run2.tech-gis-epd.json`, `eval/runs/run2.legal-labor-rules.json`
- Run 3 summary: `eval/runs/benchmark-v1-baseline-after-phasec-runs3.run3.json`
  - Dataset reports: `eval/runs/run3.sber-2022.json`, `eval/runs/run3.tech-gis-epd.json`, `eval/runs/run3.legal-labor-rules.json`

## Command

```bash
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1 PYTHONPATH=. \
.venv/bin/python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --policy-variant control --policy-strict \
  --runs 3 --max-workers 2 \
  --out eval/runs/benchmark-v1-baseline-after-phasec-runs3.json
```
