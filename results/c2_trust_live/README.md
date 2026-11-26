# C2 Trust Score refitted on live evidence: results

This answers the open question from `results/live_evidence/` point 3: the Trust model there was fitted on
gold (fact-check article) evidence. Does it help once it is refitted on **live Wikipedia evidence**?

## How it was run

1. `python -m research.eval_live --snapshot ../results/live_evidence/20261123-133939 --val-scores 1000`
   - Retrieves Wikipedia evidence (same pipeline as the app, leak filter on) for a label-stratified sample of 1,000 LIAR2
     val claims (seed 42).
   - Scores the sentence-level stance s_ev.
   - Writes `live_s_ev.csv` with the 1,000 val claims and the 500 live test claims already in the snapshot.
2. `python -m research.eval_trust --c1-predictions <snapshot>/live_s_ev.csv --text-export
   artifacts/text-distilbert-base-uncased-liar2 --only-scored --out ../results/c2_trust_live`
   - `--only-scored` keeps only claims that have a live s_ev: 500 val claims to fit the weights, 500 to calibrate
     and run conformal, and the same 500 test claims as the live-evidence run.
   - Seeds 42/43/44 for the val split; 1,000 bootstrap resamples.

## Run `20261124-005043/` (test n = 500; 6 have no evidence)

| System | ECE | Brier | macro-F1 | err @ 80% coverage | AURC |
|---|---|---|---|---|---|
| live evidence alone, calibrated | 0.043 | 0.230 | 0.606 | 0.359 | 0.306 |
| DistilBERT alone, calibrated | 0.037 | **0.184** | 0.684 | 0.260 | **0.150** |
| **Trust Score (refit on live s_ev)** | 0.037 | 0.186 | **0.695** | 0.256 | 0.158 |
| Trust Score (gold-fitted, from live_evidence run) | 0.041 | 0.185 | 0.687 | – | – |

Paired bootstrap, seed 42, Trust minus DistilBERT: Brier +0.003 [−0.002, 0.008], ECE −0.006 [−0.037, 0.033].
Both CIs include 0.

**Conformal abstention**

| α | coverage | "needs review" | error on answered |
|---|---|---|---|
| 0.05 | 0.961 | 64% | 0.108 |
| 0.10 | 0.912 | 49% | 0.171 |
| 0.20 | 0.816 | 25% | 0.245 |

Coverage meets its target at every α, so the conformal guarantee holds with live evidence.

Fitted weights (seed 42): text 0.82, evidence 1.36, evidence missing −0.17, T = 1.00. The evidence weight
looks large only because live s_ev spans a narrow range; its effect on the probability is small.

## What this shows

1. **Refitting does not rescue the Trust Score on live evidence.** Fitted on the right distribution, it ties
   DistilBERT on calibration (ECE, Brier) and gains about 0.01 macro-F1, which is within noise. The
   gold-fitted version gave the same picture, so the lack of gain is not caused by a distribution mismatch.
2. **The reason is that Wikipedia evidence is weak and largely redundant with the text.** Live evidence alone
   reaches Brier 0.230, against 0.184 for DistilBERT, so it adds little that the text model has not already seen. On
   gold evidence (results/c2_trust) the Trust Score did beat text, so C2's benefit depends on evidence
   quality. With fact-check–grade retrieval (e.g. the Google Fact Check API) it should return.
3. **Abstention still works as designed.** Conformal coverage is valid, and answering only the confident half
   cuts the error from about 0.31 to 0.17. That is a usable "needs review" triage, but DistilBERT's own confidence gives
   a similar triage (AURC 0.150 vs 0.158).

**App:** the app keeps its current `trust_config.json`. The refit is not better, so swapping it in would bring no gain.

Files: config.json, metrics.json (per seed, bootstrap, reliability, conformal), signals.csv,
test_predictions.csv, trust_config.json (research artifact, not loaded by the app).
