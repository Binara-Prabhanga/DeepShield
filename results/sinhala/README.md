# Sinhala misinformation detection (LIRNEasia corpus): results

Produced by `backend/research/eval_sinhala.py`. Data: `datasets/sinhala_lirneasia/Corpus.xlsx`
(LIRNEasia, CC BY 4.0; 3,000 Sinhala news articles from 27 outlets, mostly 2020).

## Why the protocol looks like this

- **Binary task.** CREDIBLE (real, 1,003) vs FALSE + PARTIAL (misleading, 110). UNCERTAIN (1,887) has no real/misleading
  label, so it is left out. With only 110 misleading articles, a single 15% test split would hold about 17
  of them, which is too few for stable numbers. Every model is therefore scored with **stratified 5-fold
  cross-validation repeated for seeds 42/43/44**, using the pooled out-of-fold predictions.
- **3-class task.** CREDIBLE / UNCERTAIN / MISLEADING, matching the LIRNEasia paper's setting.
- **Outlet check.** Some outlets are almost all UNCERTAIN (e.g. ravaya.lk 154/156, samabima.com 111/117).
  An outlet-only probe measures that shortcut. A **grouped** protocol holds out whole outlets so test
  outlets are never seen in training. The grouped folds barely change with the seed (sklearn's
  StratifiedGroupKFold), so its ± is near 0 and is not a real spread.
- **Main metrics for an imbalanced task:** AUROC and PR-AUC (average precision; chance = 0.10). Macro-F1 and the
  misleading-class recall/precision use a 0.5 threshold with class-balanced training.

## Run `20261123-203433/`: baselines + fine-tuned XLM-R

**Binary** (n = 1,113; mean ± sd over 3 seeds; 95% bootstrap CI for AUROC, seed 42)

| Model | Protocol | AUROC [CI] | PR-AUC | macro-F1 | misleading recall / precision |
|---|---|---|---|---|---|
| outlet only (probe) | random | 0.591 ± 0.008 [0.54, 0.65] | 0.124 | 0.451 | 0.53 / 0.12 |
| char 2–5-gram TF-IDF + LR | random | **0.748 ± 0.008** [0.68, 0.79] | **0.293** | **0.628** | 0.37 / 0.31 |
| word 1–2-gram TF-IDF + LR | random | 0.709 ± 0.009 [0.64, 0.76] | 0.249 | 0.626 | 0.31 / 0.34 |
| LaBSE embeddings + LR | random | 0.727 ± 0.011 [0.67, 0.78] | 0.293 | 0.584 | 0.52 / 0.22 |
| XLM-R base, fine-tuned (seed 42 only) | random | 0.686 [0.63, 0.74] | 0.206 | 0.576 | 0.26 / 0.23 |
| char TF-IDF | grouped (unseen outlets) | 0.754 [0.70, 0.80] | 0.285 | 0.629 | 0.36 / 0.32 |
| word TF-IDF | grouped | 0.713 [0.66, 0.77] | 0.247 | 0.627 | 0.32 / 0.34 |
| LaBSE | grouped | 0.739 [0.69, 0.79] | 0.305 | 0.595 | 0.56 / 0.23 |

**3-class** (n = 3,000; macro-F1 [CI]; per-class F1 credible / uncertain / misleading)

| Model | Protocol | macro-F1 | accuracy | per-class F1 |
|---|---|---|---|---|
| outlet only (probe) | random | 0.371 [0.35, 0.38] | 0.436 | 0.47 / 0.55 / 0.09 |
| char TF-IDF | random | **0.574** [0.56, 0.61] | 0.720 | 0.63 / 0.80 / 0.29 |
| word TF-IDF | random | 0.561 [0.52, 0.58] | 0.725 | 0.62 / 0.81 / 0.26 |
| LaBSE | random | 0.512 [0.49, 0.54] | 0.642 | 0.57 / 0.75 / 0.22 |
| char TF-IDF | grouped | 0.550 [0.52, 0.58] | 0.693 | 0.60 / 0.77 / 0.27 |
| LaBSE | grouped | 0.504 [0.48, 0.53] | 0.630 | 0.56 / 0.73 / 0.22 |

## What this shows

1. **Sinhala text carries a real but modest signal for misinformation.** The best model (character
   n-grams) ranks misleading articles above credible ones 75% of the time. Precision is low (about 0.3) because
   only 10% of articles are misleading. That is usable for flagging articles for review, not for automatic decisions.
2. **Little outlet shortcut on this task.** The outlet-only probe is weak (AUROC 0.59), and every text
   model holds up on unseen outlets (AUROC is unchanged or slightly higher). This is unlike LIAR2, where
   metadata alone matched the text model (C4). For the 3-class task, holding out outlets costs about 0.02–0.03
   macro-F1, because UNCERTAIN depends partly on the outlet.
3. **Character n-grams beat multilingual sentence embeddings.** Sinhala spelling and suffixes are
   captured well by character n-grams. LaBSE, trained mostly on other languages, finds more misleading
   articles (recall 0.52) but with more false alarms.
4. **The misleading class is the hard part of the 3-class task** (F1 about 0.29). That is expected with 110 examples.

5. **Fine-tuning a large multilingual transformer does not help here.** XLM-R base (`--xlmr`;
   class-weighted loss, 3 epochs, 128 tokens, embeddings frozen so it trains on a CPU, about 2.5 h) reaches
   AUROC 0.686. That is below character n-grams (0.748), and the CIs barely overlap. With about 90 misleading
   articles per training fold, the 278M-parameter model cannot beat a linear model on n-grams. More labelled
   Sinhala misinformation, not a bigger model, is what this task needs. The run has one seed and the
   random protocol only, because of CPU time.

The app's English DistilBERT is not used for Sinhala: it does not cover the script.

## In the app

`python -m research.eval_sinhala --export artifacts/text-sinhala-char-tfidf` fits the char TF-IDF model on
all 1,113 binary rows (seed 42), wrapped in 5-fold sigmoid calibration so P(real) matches the corpus base rate.
With `SINHALA_MODEL_PATH` set, the app routes text that is at least 50% Sinhala letters to this model and
returns `mode: "model-sinhala"`. SHAP, CLIP image fusion, NLI stance and the Trust Score are English-only and are
skipped, with a note explaining why. Its expected quality is the cross-validated AUROC 0.75 above.

Files: config.json, metrics.json, oof_seed42.csv (out-of-fold P(misleading) per article, binary task).
