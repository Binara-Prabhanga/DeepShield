# Trust Score on live evidence with Google Fact Check (C1 live and C2)

Produced on 2026-11-24 by `backend/research/merge_factchecks.py`, `eval_live.py` and `eval_trust.py`.
Test = 500 LIAR2 test claims, val = 1,000 val claims (500 fit the Trust model, 500 calibrate it). Seeds fixed.

## What was compared

| Setup | Evidence | Folder |
|---|---|---|
| Wikipedia only | Wikipedia (98.8% of claims) | `live_evidence/20261123-133939`, `c2_trust_live/20261124-005043` |
| + fact checks (all) | Wikipedia + Google Fact Check, PolitiFact dropped | `live_evidence/20261124-factcheck-all`, `c2_trust_live_fc/all` |
| + fact checks (strict) | as above, and fact checks of the same claim (word overlap >= 0.5 with the LIAR2 claim) also dropped | `live_evidence/20261124-factcheck-strict`, `c2_trust_live_fc/strict` |

A first fact-check run was discarded because Wikipedia rate-limited it (169/500 claims). The Wikipedia items here come from
the earlier complete run; only the Google items are new, and both are ranked with the app's `rank_evidence`.
**Use the strict row for any claim in the paper:** LIAR2 claims come from PolitiFact, and other outlets sometimes
fact-check the very same statement, which would be label leakage.

## How much fact-check evidence there is

PolitiFact items: 153 (always dropped). Same-claim items from other outlets: 28 (dropped in strict). Fair items kept:
75, covering only **41 of 500 claims (8%)**: FactCheck.org, USA Today, VOA, Snopes, AP, Washington Post.

## C1: stance as a veracity signal (AUROC of s_ev for real vs not real; 95% bootstrap CI)

| Setup | claims scored | AUROC | decisive claims only | where both live and gold are decisive: same sign |
|---|---|---|---|---|
| Wikipedia only | 494 | 0.639 [0.588, 0.689] | 0.622 (n=332) | 0.73 |
| + fact checks (all) | 495 | 0.669 [0.619, 0.715] | 0.661 (n=339) | 0.74 |
| + fact checks (strict) | 495 | **0.663** [0.613, 0.708] | 0.652 (n=338) | 0.74 |
| gold evidence, same claims (reference) | 500 | 0.683 [0.637, 0.729] | | |

Fact checks lift live AUROC by about +0.02 to +0.03 and close most of the gap to gold evidence. The CIs overlap, so this is
a small gain, not a proven one. Strict removes only 0.006 of it.

## C2: Trust Score (test n = 500; Brier lower is better)

| System | Brier | ECE | macro-F1 |
|---|---|---|---|
| text only (DistilBERT, calibrated) | 0.184 | 0.037 | 0.684 |
| evidence only (calibrated) | 0.222 (all), 0.224 (strict) | 0.039 / 0.043 | 0.63 |
| Trust, Wikipedia only | 0.186 | | 0.695 |
| Trust, + fact checks (all) | 0.185 | 0.043 | 0.697 |
| Trust, + fact checks (strict) | **0.185** | 0.040 | 0.695 |

Trust minus text-only Brier (bootstrap): all +0.001 [-0.005, 0.007]; strict +0.001 [-0.004, 0.007]. Trust beats evidence alone
by about -0.037 [-0.049, -0.023] (strict -0.038). **No difference from the text model on Brier or ECE.**
Macro-F1 is about 0.01 higher with Trust (seed-stable, 0.695 vs 0.684).

Conformal abstention at alpha = 0.1 (strict): coverage 0.915, abstains on 49% of claims, error on answered claims 0.166
(Wikipedia only: 49%). Abstention is set by the text model's uncertainty, not by the evidence.

## Claims that have a fair fact check (strict; n = 41, so only indicative)

| Subset | n | share real | Brier text / Trust | abstain | s_ev AUROC (decisive claims) |
|---|---|---|---|---|---|
| fair fact check present | 41 | 0.12 | 0.127 / 0.106 | 22% | 0.993 (n=39) |
| none | 459 | 0.44 | 0.189 / 0.191 | 49% | 0.609 (n=299) |

Where a real fact check exists, the evidence signal is nearly perfect and Trust is a little better calibrated and abstains
less. But this subset is tiny and only 5 of its claims are real, so the AUROC cannot be trusted to the second decimal.
The subset is also not random: claims that get fact-checked are mostly false.

## C3 H4 on live evidence (does low explanation-evidence agreement flag errors?)

Unchanged: low-EEA AUROC 0.668 (strict; 0.656 Wikipedia only), against 0.762 for low model confidence alone and 0.755
for both combined. EEA adds nothing beyond confidence, as before. Mean EEA is 0.28 on correct and 0.14 on wrong predictions.

## Conclusions

1. Google Fact Check improves live evidence quality a little (C1 AUROC +0.02), but only 8% of claims have a usable fact check,
   so the Trust Score cannot improve overall. It still ties the text model on calibration and wins slightly on macro-F1.
2. The earlier finding stands: the benefit of evidence shows only where good evidence exists.
3. The app's `trust_config` was **not** changed. The existing config is still the one fitted on gold evidence.

Files: `all/` and `strict/` each hold `metrics.json`, `config.json`, `signals.csv`, `test_predictions.csv`, `trust_config.json`
inside a timestamped folder.
