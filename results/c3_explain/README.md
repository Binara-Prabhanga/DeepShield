# C3 — Explanation audit (faithfulness + Explanation–Evidence Agreement): results

Produced by `backend/research/eval_explain.py`; EEA from `backend/app/services/explain_audit.py`.
Data: the 500 LIAR2 test claims from the C1 trial (`results/c1_stance/trials/500claims_base_20261122-200459`).
Evidence = each justification sentence, labelled supports/refutes/unrelated by the cached C1 NLI scores.
EEA matching rule (fixed before running): lowercase word, stopwords removed, light stemming, top-k = 10.

## Run `20261123-125219/`: trained DistilBERT (main text result, n = 500, about 85 min on CPU)

Command: `TEXT_MODEL_PATH=artifacts/text-distilbert-base-uncased-liar2/model python -m research.eval_explain --text-model bert`.
The model is DistilBERT fine-tuned on LIAR2 (seed-42 weights, T = 1.70). Its accuracy on these 500 claims is 0.694.

**Faithfulness** (AOPC over 1/5/10/20/50% of words; random = 5 orderings × 3 seeds)

| | SHAP | random | SHAP − random [95% CI] |
|---|---|---|---|
| comprehensiveness (higher is better) | 0.142 | 0.027 | +0.116 [+0.109, +0.123] |
| sufficiency (lower is better) | 0.012 | 0.146 | −0.136 [−0.143, −0.130] |

- **SHAP is faithful for the neural model too.** Removing its top words lowers confidence about 5× more than
  removing random words.
- **Keeping only its top words almost preserves the prediction** (sufficiency 0.012).
- The gap over random is a little smaller than for the near-linear placeholder (+0.116 vs +0.129), as expected.

**EEA.** The same 322 of 500 claims have supporting or refuting evidence. Mean signed EEA is 0.30 (unsigned 0.51), and 16% of claims have EEA = 0.

**H4: does low EEA warn of errors?** (322 claims, 97 errors; AUROC, where 0.5 = chance)

| Warning score | AUROC [95% CI] |
|---|---|
| low signed EEA | 0.550 [0.481, 0.620] |
| low unsigned EEA (ablation) | 0.473 [0.400, 0.541] |
| low model confidence | **0.766 [0.714, 0.817]** |
| both combined (5-fold CV) | 0.750 [0.693, 0.804] |

- **H4 is not supported with the trained model.**
  - Signed EEA's CI includes chance, and adding it to confidence does not help (0.750 vs 0.766).
  - Wrong predictions do have slightly lower EEA (0.27 vs 0.31), and signed beats unsigned again, so the
    direction of the effect is consistent but weak.
- **DistilBERT's own confidence is a much better error warning** than the placeholder's (0.766 vs 0.690).
  This is what the C2 Trust Score uses.
- **Report EEA as an interpretability measure**, showing whether the explanation's words are the ones the evidence
  talks about, not as a reliability signal. The evidence used here is fact-check prose. Live retrieval
  with more claim-specific evidence is the remaining test.

## Smoke run `20261122-224233/`: TF-IDF placeholder (baseline, n = 500, 93 s)

SHAP on a near-linear model is close to faithful by design, so this run only showed that the test works.

**Faithfulness** (AOPC over 1/5/10/20/50% of words; random = 5 orderings × 3 seeds)

| | SHAP | random | SHAP − random [95% CI] |
|---|---|---|---|
| comprehensiveness (higher is better) | 0.146 | 0.016 | +0.129 [0.125, 0.134] |
| sufficiency (lower is better) | −0.149 | 0.079 | −0.227 [−0.236, −0.218] |

As expected for this model, SHAP is clearly faithful: removing its top words drops confidence about 9× more than removing random words.

**EEA** — 322 of 500 claims have at least one supporting or refuting sentence (the other 178 have
only "unrelated" sentences, which matches C1's finding that NLI rarely commits on fact-check prose).
Mean signed EEA is 0.28; unsigned is 0.50; 19% of claims have EEA = 0.

**H4 — does low EEA warn of errors?** (322 claims, 93 errors; AUROC, 0.5 = chance)

| Warning score | AUROC [95% CI] |
|---|---|
| low signed EEA | 0.579 [0.509, 0.650] |
| low unsigned EEA (ablation) | 0.506 [0.433, 0.579] |
| low model confidence | 0.690 [0.625, 0.746] |
| both combined (5-fold CV) | 0.695 [0.633, 0.750] |

- Signed EEA is weakly better than chance, and the direction matters: unsigned EEA is at chance.
- EEA adds almost nothing on top of the model's own confidence (0.695 vs 0.690). The DistilBERT run above confirms this.
- 36% of claims get no EEA value at all. Live retrieval with more evidence items should raise this.

Files: config.json, metrics.json, items.csv (one row per claim: prediction, faithfulness, EEA, matched words).

## Image run `20261123-121344-images/`: Grad-CAM on the trained CLIP fusion head (n = 100)

Command: `MULTIMODAL_MODEL_PATH=artifacts/mm-clip-fusion-fakeddit/model python -m research.eval_explain --images 100`.
- **Data.** 100 Fakeddit test pairs with real images (imgur placeholders skipped), seed 42. Accuracy on these pairs is 0.91.
- **Method.** Grad-CAM on the 7×7 CLIP patch grid. Cells are deleted in Grad-CAM order, replaced by the image's
  mean colour, and the result is compared with random cell orders.
- **Sample size.** 100 pairs instead of 200, to keep the CPU run under 2 hours.
- **Data caveat.** Fakeddit is an unofficial mirror: these numbers are for building and testing only.

| | Grad-CAM | random | Grad-CAM − random [95% CI] |
|---|---|---|---|
| comprehensiveness (higher is better) | 0.013 | 0.001 | +0.012 [+0.004, +0.021] |
| sufficiency (lower is better) | 0.014 | 0.031 | −0.017 [−0.028, −0.009] |

- **Grad-CAM is more faithful than random** in both measures, and both CIs exclude 0.
- **But the image barely moves the prediction.** Deleting up to half the image changes P(real) by
  only about 0.01–0.03 on average. The fusion head therefore relies mostly on the title embedding and on overall
  image content that survives patch deletion. This fits the fusion gain (+0.03 F1 over title-only, see C4).
- **Grad-CAM heatmaps should be shown as a weak signal**, not as "the part of the image that decided
  the result".
