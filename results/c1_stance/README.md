# C1 — NLI evidence stance: results

Data: LIAR2, using the fact-check justification as gold evidence. Labels 4–5 count as supports, 0–1 as refutes, and 2–3 are excluded from scoring.
Thresholds and the choice of variant are made on val only. Seed 42. Produced by `backend/research/eval_stance.py`.

## Chosen configuration
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, sentence-level scoring (restatement filter) with the
`decisive` aggregation rule (the default in `app/services/stance.py`).

## Runs
| Folder | What | Test n | Key result |
|---|---|---|---|
| `20261122-193507/` | **Main result.** Full val/test, base model, passage vs sentence (decisive) | 1565 | sentence_tuned macro-F1 0.687 [0.664, 0.710] vs similarity 0.567 and passage 0.615; both gaps significant (McNemar p < 1e-6). High-similarity debunk recall: 0.63 vs 0 for similarity |
| `trials/500claims_base_20261122-200459/` | 500-claim sample, base model, compares the aggregation rules decisive/max/mean | 342 | All three rules within noise (test 0.697 / 0.683 / 0.694) |
| `trials/500claims_large_20261122-220223/` | Same 500 claims, `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 342 | **Negative result:** decisive 0.700 (vs 0.697 base). The rule picked on val (mean) scores 0.669 on test. About 3–4× slower on CPU |

## Known limitation
Sentence-level scoring leans toward "refutes": mostly-true claims (label 4) average s_ev ≈ −0.19 with
both models. Fact-check write-ups give context more than they restate the claim outright, so NLI rarely
finds strict entailment. A larger model did not fix this. Calibration is left to C2.

Each folder contains config.json (model + revision), metrics.json, predictions.csv and, for the trials,
premise_scores.jsonl. With that file, `--from-scores` re-aggregates without loading the model.
