# Live-evidence test (C1–C3 with the app's own retrieval): results

Produced by `backend/research/eval_live.py`. C1–C3 were first measured using each LIAR2 claim's
PolitiFact justification as evidence ("gold" evidence, an upper bound). Here the same 500 LIAR2 test
claims as C3 go through the app pipeline as a user's claim would:

`gather_evidence` → `run_models` (trained DistilBERT + SHAP) → `assess` (C1 stance, C2 Trust Score, C3 EEA)

- **Retrieval used: Wikipedia only.** It is the app's keyless source, and the Tavily and Google Fact Check keys are not set, so
  those providers returned nothing (`provider_status`). Results with them would differ.
- **Leak filter:** items from politifact.com are dropped. None were retrieved.
- **Snapshot:** `evidence.jsonl` keeps the retrieved evidence, so the analysis can be rerun after the web changes:
  `python -m research.eval_live --snapshot <run> --from-items` recomputes the metrics.

## Run `20261123-133939/` (n = 500 claims, about 2 h on CPU)

**Coverage.** 98.8% of claims get evidence (mean 7.4 Wikipedia snippets), and 66.4% get at least one
snippet with a supports/refutes stance.

**C1: does live evidence stance separate real from misleading claims?** (AUROC of s_ev, 95% CI)

| Evidence | n | AUROC |
|---|---|---|
| live Wikipedia, all claims with evidence | 494 | 0.639 [0.588, 0.689] |
| live Wikipedia, claims with a decisive stance | 332 | 0.622 [0.561, 0.683] |
| gold PolitiFact justification (same claims) | 500 | 0.683 [0.637, 0.729] |

Where both are decisive, live and gold evidence point the same way for 73% of claims.

**C3 / H4: does low EEA warn of DistilBERT's errors?** (AUROC; 0.5 = chance)

| Evidence | n (errors) | low EEA | low confidence | both (5-fold CV) |
|---|---|---|---|---|
| live Wikipedia | 332 (98) | **0.656 [0.593, 0.715]** | 0.762 [0.711, 0.813] | 0.756 [0.698, 0.808] |
| gold PolitiFact | 322 (97) | 0.550 [0.481, 0.620] | 0.766 [0.714, 0.817] | 0.750 [0.693, 0.804] |

Mean EEA is 0.26 for correct predictions vs 0.14 for wrong ones (live); with gold evidence it is 0.31 vs 0.27.

**C2: Trust Score with live evidence** (seed-42 Trust model from `results/c2_trust/20261123-111702`)

| System | Brier | ECE | macro-F1 | "Needs review" | error on answered |
|---|---|---|---|---|---|
| DistilBERT alone | 0.184 | 0.035 | 0.690 | – | – |
| Trust Score, live evidence | 0.185 | 0.041 | 0.687 | 47% | 0.182 |

## What this shows

1. **C1 holds up outside the lab setting.** Generic Wikipedia snippets give a stance signal only a little
   weaker than the fact-checker's own article (AUROC 0.64 vs 0.68; the CIs overlap). Live retrieval also
   found evidence for almost every claim.
2. **EEA is more informative with live evidence, but it still adds nothing to confidence.** With
   Wikipedia evidence, low EEA flags wrong predictions clearly above chance (0.66). With fact-check prose it does not
   (0.55): fact-checks restate the claim's words whatever the verdict, so they agree with any explanation.
   Combining EEA with DistilBERT's confidence does not beat confidence alone (0.756 vs 0.762), so H4 (that EEA
   is a useful extra warning) is still not supported. EEA remains useful as an interpretability measure.
3. **The Trust Score does not transfer as-is from gold to live evidence.** Its weights were fitted on
   gold-evidence s_ev, whose distribution differs from live s_ev. With live evidence it is no better than DistilBERT alone
   and abstains on 47% of claims. Refitting it on live s_ev does not help either
   (`results/c2_trust_live/`: Brier 0.186 vs 0.184). Wikipedia evidence is too weak to add to the text model.

Files: config.json, evidence.jsonl (retrieved evidence per claim), items.csv, metrics.json.

## Update 2026-11-24: Google Fact Check added

A Google Fact Check key was added and the run repeated with fact checks, in `20261124-factcheck-all` (PolitiFact
dropped) and `20261124-factcheck-strict` (fact checks that restate the same claim also dropped). Results and the
Trust refit are in `results/c2_trust_live_fc/README.md`. In short: only 8% of claims have a fair fact check; live C1
AUROC rises from 0.639 to 0.663 (strict); Trust still ties the text model (Brier 0.185 vs 0.184); H4 is still not supported.
