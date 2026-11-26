# Gemini evidence judge (experiment, 2026-11-24 and 2026-11-25)

Code: `backend/app/services/judge.py` (prompts, key and model rotation), `backend/research/eval_judge.py`
(experiments), `backend/research/eval_facts_app.py` (whole app on plain facts), data
`backend/research/judge_facts.csv`. Gemini keys live only in `backend/.env`.

**Question.** Can an LLM that reads the retrieved evidence ("supported / refuted / not enough evidence") fix what the
text model cannot do: tell plain true facts from false or invented ones, without hurting political claims?

## What went wrong at first, and what fixed it

| Round | Evidence the judge read | Result |
|---|---|---|
| 1 (`20261124-203455/`, 350 LIAR2 claims) | Wikipedia **search fragments** (keyword snippets) | decided only 12%; "refuted" 34/34, "supported" 5/8 |
| 2 | Wikipedia **opening paragraphs** (real sentences) + prompt `knowledge` | plain facts: 57/60 decided, all right |
| 3 | + prompt `plausibility` (also refutes absurd claims, never refutes just because sources are silent) | invented claims 20/20 refuted; true Sri Lanka claims 15/15 supported |

So the first low coverage was mostly the weak evidence text, not the judge. A "claim kind" tag (general fact vs political)
was tried to decide when "supported" may raise a score; it mislabelled political claims (3 of 5 right) and was dropped.
Coverage of the Wikipedia title in the claim, and detail coverage in one passage, showed no link to truth (AUROC 0.48) and were not used.

## Method (prompt chosen on development data only)

Sets: LIAR2 validation (150 claims) and the first half of each facts set for choosing the prompt; the untouched
LIAR2 test claims (500) and the second half of each facts set for the final numbers.
`judge_facts.csv`: 60 true + 60 false plain facts (each false one is a changed copy of a true one), 40 invented claims,
30 true but less famous Sri Lanka claims. All Wikipedia opening paragraphs and Google Fact Check results are live.

## Results

Plain facts and invented claims, untouched test halves, **whole app** (`eval_facts_app`, text model + evidence + judge + rules):

| Test set | n | Right | Wrong | Needs review |
|---|---|---|---|---|
| Plain facts (30 true, 30 false) | 60 | 58 | 0 | 2 |
| Invented claims | 20 | 20 (all "not real") | 0 | 0 |
| True Sri Lanka claims | 15 | 15 (all "real") | 0 | 0 |

The 2 reviews are true facts the judge supported but the model treats as in-domain (Russia's 2022 invasion, "Antarctica is the
coldest continent"); see the rule below. Judge alone on the facts test half: 60/60 decided and right.

Political claims (LIAR2), untouched test (500 claims, prompt `knowledge`, opening paragraphs):

| | |
|---|---|
| Judge commits | 77 claims (15%) |
| "Refuted" right | 56 of 64 (88%) |
| "Supported" right | 9 of 13 (69%) |
| Where it decides | judge 0.84 vs text model 0.79 on the same 77 claims |
| Effect of the app's rules | accuracy over all 500: 0.696 (text model) -> 0.700; AUROC with judge added 0.780 -> 0.782 |

On validation claims the `plausibility` prompt decided 37 of 149 (`knowledge`: 28) with "refuted" right 30 of 32.
It was not rerun on the 500 test claims.

## How the app uses it (`JUDGE_ENABLED=true`)

`app/main.py: decide()`, in priority order: a fact-checker's True/False rating for the same claim; judge **refuted** caps the
score at 20; judge **supported** raises the score to at least 85 **only when the claim is outside the text model's domain**
(`app/services/domain.py`); supported on a political claim with a low score becomes "Needs review" (political "supported" was
only 69% right, so it never raises a political claim); otherwise out-of-domain claims get "Needs review".
The judge reads the Wikipedia opening paragraphs only for itself, so stance and the Trust Score keep the evidence they were fitted on.

## Limits

- Free Gemini quota is per model and per day. The judge rotates three keys and up to four models, rests a pair for an
  hour after a daily-quota 429, stops after 8 requests and 30 s, and the app simply skips it when nothing answers.
- The judge reads Wikipedia and fact checks only; recent or local news that is not there ends as "needs review".
- Plain-fact sets are small (95 test claims), written by us, and easier than real news. They show the mechanism works, not
  real-world accuracy. Sending the claim and snippets to Google is a privacy cost.
- A same-claim fact-check match now needs Jaccard 0.75 for claims of 4 content words or fewer, because "Mount Fuji is in
  Japan" matched a review of "Mount Fuji has erupted" and the Burj Khalifa a review of a fake photo.
