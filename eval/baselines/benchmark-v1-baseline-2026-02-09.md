# Baseline: benchmark-v1-3docs (2026-02-09)

## Aggregate

- benchmark: `benchmark-v1-3docs`
- datasets: `3`
- total_questions: `48`
- **weighted_score: 0.6854807299522374**
- **question_weighted_score: 0.6628124705411013**

## Per-dataset

1) `sber-2022` (finance)
- questions: `24`
- weighted_score: `0.5948076923076925`
- pdf_sha256: `8a859081ed75174ba1c86a4578b2450c40e517d90a15a4e1f636f94058d3bdb9`
- questions_sha256: `0c4f7257604c1ef213efb950c663c4ae8af5f246de0635f3b9b11044232dd9c9`

2) `tech-gis-epd` (tech-doc)
- questions: `12`
- weighted_score: `0.5784313725490196`
- pdf_sha256: `2629e5e87bb26ffcc0e3a3ffb330de22b18b6a2d02ebea52fa061721a3faa395`
- questions_sha256: `43ef414190fcb947e1fe9be6b21c6dc0843221ec9e5a9bb1b5f8e8bb09f47284`

3) `legal-labor-rules` (legal)
- questions: `12`
- weighted_score: `0.8832031250000001`
- pdf_sha256: `f5cc03362cedd2379be3d60ae216260a283f2293a9df073f2274e41ceac1f20d`
- questions_sha256: `c31152aa71ce3b6f3553babbb8b957db059779ff0983c51c501b6c6467b90d62`

## Command

```bash
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=1 PYTHONPATH=. \
python eval/benchmark/run_benchmark.py \
  --config eval/benchmark/config.v1.json \
  --out eval/runs/benchmark-v1-baseline.json
```

> Raw run is local-only: `eval/runs/benchmark-v1-baseline.json` (ignored by git).
