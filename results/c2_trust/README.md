# C2 — Calibrated Trust Score with abstention: results

Produced by `backend/research/eval_trust.py` with the service `backend/app/services/trust.py`.
Data: LIAR2 val/test, binary: real = labels 3–5 (half-true counts as real), misleading = 0–2.
Val is split 50/50 per seed: one half fits the weights and T, the other sets the conformal
threshold. Test is used only for scoring. Seeds 42, 43, 44 (mean ± sd); bootstrap CIs use seed 42.

## Run `20261123-111702/`: trained DistilBERT text signal (main result, test n = 2296)

Command: `python -m research.eval_trust --text-export artifacts/text-distilbert-base-uncased-liar2`.
- **Text signal.** DistilBERT fine-tuned on LIAR2 (`research/train_text.py`), using the seed-42
  weights with their own temperature (T = 1.70, fitted on LIAR2 val). The ± over seeds therefore
  only reflects the val fit/calibration split.
- **Image signal.** Absent, because LIAR2 has no images.
- **Evidence.** The C1 stance score on the fact-checker's justification, which is an upper bound for evidence quality.

| System | ECE | Brier | macro-F1 | error at 80% coverage | AURC |
|---|---|---|---|---|---|
| evidence, raw (s_ev+1)/2 | 0.194 | 0.267 | 0.633 | 0.324 | 0.273 |
| DistilBERT, temperature-scaled | 0.016 | 0.185 | 0.707 | 0.249 | 0.156 |
| average of the two | 0.041 | 0.192 | 0.678 | 0.260 | 0.170 |
| evidence, calibrated | 0.024 | 0.223 | 0.589 | 0.326 | 0.270 |
| DistilBERT, recalibrated on the fit half | 0.021 | 0.185 | 0.703 | 0.249 | 0.155 |
| **Trust Score** | **0.025** | **0.178** | **0.722** | **0.227** | **0.141** |

- **H2 (combined beats single signals): supported for Brier, macro-F1 and selective risk, not for ECE.**
  - Brier vs DistilBERT: −0.007 [−0.011, −0.003].
  - Brier vs evidence: −0.046 [−0.052, −0.039].
  - macro-F1 rises 0.703 → 0.722, and the error at 80% coverage falls 0.249 → 0.227.
  - ECE does not differ significantly from either signal (vs text +0.001 [−0.018, +0.017]); every calibrated system sits near 0.02.
- **Temperature scaling matters for the neural model.**
  - On LIAR2 test, raw DistilBERT probabilities have ECE 0.077 (Brier 0.192). After its T = 1.70 they have ECE 0.016 (Brier 0.185).
  - T > 1 means the model was overconfident.
  - The Trust Score's own T is ≈ 1.00 because its inputs are already calibrated.
- **Learned weights (seed 42):** text 0.90 and evidence 0.73 (on the logit scale), bias 0.17. Both signals contribute.
- **P(real) rises steadily by LIAR2 label:** 0.21, 0.29, 0.45, 0.55, 0.59, 0.63 for labels 0–5.
- **H3 / abstention.**
  - Split conformal meets its targets: coverage 0.953 / 0.901 / 0.800 at α = 0.05 / 0.1 / 0.2.
  - At α = 0.1, 44% of claims are "Needs review", and the error on the rest is 0.175.
  - At α = 0.2 only 16% abstain, with error 0.238. The signals are still weak enough that a strict guarantee costs a lot of coverage.

`trust_config.json` in this run is the seed-42 model built on real signals. To use it in the app,
set `TRUST_MODEL_PATH` to that folder together with `TEXT_MODEL_PATH=<liar2 export>/model` and `STANCE_MODEL`.

## Run `20261122-222702/`: TF-IDF placeholder (baseline, test n = 2296)

The text signal here was a TF-IDF + logistic regression placeholder, used before training.

| System | ECE | Brier | macro-F1 | error at 80% coverage |
|---|---|---|---|---|
| evidence, raw (s_ev+1)/2 | 0.194 | 0.267 | 0.633 | 0.324 |
| text placeholder, raw | 0.041 | 0.192 | 0.683 | 0.256 |
| average of raw signals | 0.037 | 0.196 | 0.664 | 0.277 |
| evidence, calibrated | 0.024 | 0.223 | 0.589 | 0.326 |
| text placeholder, calibrated | 0.029 | 0.191 | 0.685 | 0.256 |
| **Trust Score** | **0.026** | **0.182** | **0.704** | **0.238** |

- **H2 (combined beats single signals):** partly supported. Brier is lower than each calibrated single
  signal (vs text −0.009 [−0.013, −0.005]; vs evidence −0.041 [−0.048, −0.035]). ECE does not
  differ significantly from either; all calibrated systems are already near 0.02–0.03.
- **Calibration fixes C1's refutes lean:** raw evidence has ECE 0.19. After calibration, mean P(real)
  rises steadily by label (0.24, 0.30, 0.43, 0.55, 0.58, 0.63 for labels 0–5).
- **Temperature scaling adds nothing here** (T = 1.00), because logistic regression is already
  calibrated. It is expected to matter once BERT logits, which tend to be overconfident, are added.
- **H3 / abstention:** split conformal meets its coverage targets (0.955 / 0.902 / 0.803 at
  α = 0.05 / 0.1 / 0.2). The cost is high with these weak signals: at α = 0.1, 44% of items are
  "Needs review" and the error on answered items is 0.17. At about 80% coverage, conformal (error
  0.243) performs like a plain confidence threshold (0.238), so its benefit is the guaranteed
  coverage, not a lower error. Trust error at 80% coverage (0.238) is below the uncalibrated
  signals (0.256 text, 0.277 average), consistent with H3.

Files: config.json, metrics.json (per seed, summary, bootstrap, reliability bins, P(real) by label),
signals.csv, test_predictions.csv, trust_config.json (seed-42 model). Only load the trained
run's config in the app; the placeholder run's config expects a TF-IDF text signal.
