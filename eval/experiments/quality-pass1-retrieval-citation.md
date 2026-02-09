# Quality improvement pass #1 (retrieval + citation resilience)

## Hypothesis

Universal quality can be improved without document-specific tuning by:
1) making multi-query retrieval kick in earlier,
2) forcing a second pass when an answer lacks citations,
3) appending citation-only evidence when the answer is otherwise valid but uncited.

## Code changes

- `ragbot/agent.py`
  - coverage threshold for multiquery trigger: `0.40 -> 0.50`
  - second-pass condition expanded:
    - was: retry/not_found for selected intents
    - now: retry/not_found/**citation_missing** for all main intents
  - final citation repair step added:
    - if answer has no `(стр. N)` refs, invoke citation-only chain and append evidence

## Benchmark results (v1, 3 docs)

Baseline:
- weighted_score: `0.6854807299522374`
- question_weighted_score: `0.6628124705411013`

Candidate (pass #1):
- weighted_score: `0.7453758899393091`
- question_weighted_score: `0.7096569174544819`

Delta:
- weighted_score: `+0.0598951599870717`
- question_weighted_score: `+0.04684444691338056`

Per-dataset deltas:
- `sber-2022`: `0.5948076923076925 -> 0.6025000000000001` (`+0.007692307692307665`)
- `tech-gis-epd`: `0.5784313725490196 -> 0.6801120448179272` (`+0.10168067226890756`)
- `legal-labor-rules`: `0.8832031250000001 -> 0.953515625` (`+0.07031249999999989`)
