# Out-of-domain check and fact-check verdicts (app behaviour, 2026-11-24)

Produced by `backend/research/fit_domain.py` (run `20261124-202223/`). Code: `app/services/domain.py`,
`app/services/factcheck_verdict.py`; both are wired into `app/services/assessment.py` and `app/main.py`.

## Why

A live test with a true statement (Olympics in Tokyo in 2021) and an invented one (the Moon renamed after a cricketer)
gave both the same verdict, "Potentially misleading, 13". The text model learned US political claims (LIAR2) and does
not know facts. Two fixes were added.

## 1. A fact-checker's verdict on the same claim decides the score

If Google Fact Check returns a review of (nearly) the same claim (content-word overlap >= 0.5) rated clearly true or
false, the score becomes 92 or 8 and a note names the publisher and rating. Mixed ratings ("half true", "needs context")
and disagreeing reviews never override.

Check, without PolitiFact (LIAR2 comes from PolitiFact, so using it would be circular): on the 500 live test claims, 19 had a
same-claim true/false rating from another publisher (USA Today, AP, AFP, Snopes, FactCheck.org, Full Fact, Misbar).
The rating agreed with the LIAR2 label on **19 of 19**. Small sample, but no disagreements. Only about 4% of claims
are covered, so this fixes the claims that have been fact-checked, not claims in general.

## 2. Out-of-domain claims are sent to review

Similarity of a claim to the **25 nearest** LIAR2 training claims (mean cosine of `all-MiniLM-L6-v2` sentence
embeddings; run `20261124-231336/`). The threshold is the 5th percentile of LIAR2 validation claims (0.385), so about
5% of in-domain claims are sent to review (5.4% on LIAR2 test).

| Data | median similarity | share sent to review | AUROC "far from LIAR2" |
|---|---|---|---|
| LIAR2 validation | 0.509 | 5% (by construction) | n/a |
| Fakeddit Reddit titles (1,000) | 0.320 | 83% | **0.961** |
| Sri Lankan fact-check claims (212) | 0.333 | 78% | **0.955** |

Choosing k: the nearest single claim (k=1) gave AUROC 0.955 / 0.932 and let a lucky near-duplicate make an unfamiliar
claim look familiar. Averaging 10, 25 or 50 neighbours gave 0.966 / 0.952-0.955; k=25 was used. (A TF-IDF word-overlap
version separated the data much worse, 0.74 / 0.68, and was dropped.) The Olympics claim, which k=1 missed, falls just
below the threshold at k=25 (similarity 0.371 vs 0.385, 3rd percentile of LIAR2). That margin is small, so a similar
claim could land either side.

Limits:
- It detects unfamiliar topics, not wrong answers. On the LIAR2 test claims it rejects, the text model's error rate
  is the same as on the rest (0.30 vs 0.30; 20 claims). Its job is to say "outside what this model knows".
- Text that looks political passes, and claims near the threshold can go either way.
- Sri Lankan claims are mostly rejected, which fits the C4 finding that the model does not transfer there.

Enable with `DOMAIN_MODEL_PATH=artifacts/domain-liar2` (the artifact is not committed; rebuild it with
`python -m research.fit_domain`).
