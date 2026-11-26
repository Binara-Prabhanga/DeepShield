# DeepShield demo guide

## Start (two terminals)

```
cd backend  && .venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
cd frontend && npm run dev          # open http://127.0.0.1:5173
```

Needs internet (Atlas database, Wikipedia, Google Fact Check, Gemini) and about 4 GB of free RAM. The first analysis
loads the models and can take 1-2 minutes; later ones take 5-40 seconds. Register a new account on the login page.
`backend/.env` holds the settings (real models are on: `MODEL_MODE=model`).

## Inputs to try (what you should see)

Paste at least 25 characters. Results depend on live services (Wikipedia, Google Fact Check, Gemini).

| Paste this | Expected result | Why |
|---|---|---|
| The 2020 Summer Olympics were held in Tokyo, Japan, in July and August 2021 after being postponed for a year because of the COVID-19 pandemic. | **Likely authentic**, 85, note "Gemini evidence judge: supported ..." | A true plain fact: the judge reads Wikipedia, the text model is out of its domain |
| Mount Fuji is the highest mountain in Japan. | **Likely authentic**, 85 | Same |
| The capital city of Australia is Sydney. | **Potentially misleading**, 20, note "judge: refuted ... Canberra" | A false plain fact |
| The Sri Lankan government announced that the Moon will be renamed after a local cricketer, and every school in the country will close for a full year to celebrate. | **Potentially misleading**, 20 (an invented claim) | The judge refutes absurd claims |
| COVID-19 vaccines contain microchips that let the government track people. | Potentially misleading, **8**, note "Fact-checked by AFP Fact Check: rated False" | A fact-checker's verdict decides |
| Paris Eiffel Tower grows as much as 6 inches in height each summer. | **Likely authentic**, about 92, "Fact-checked by Snopes: rated Mostly True" | A fact-checker's True rating |
| Unemployment doubled under Governor Walker in Wisconsin. | Needs review, about 55 | A political claim: the app does not claim to settle it; the report says the score is mostly the text model |
| A Sinhala news sentence | "model-sinhala" mode, text score only | Separate Sinhala model (AUROC 0.75 in cross-validation) |

If Gemini has no quota left (free keys, per day) the judge is skipped and true plain facts show "Needs review" instead
of "Likely authentic". That is the main demo risk: check `results/judge/README.md` (Limits) and consider a paid key.

## What to say about the research (numbers are in `results/`)

- **Not a fact-checker.** It flags claims for review. The text model learned US political claims (LIAR2).
- C1: sentence-level NLI beats similarity (macro-F1 0.687 vs 0.567). With live Wikipedia evidence the stance signal
  is weaker (AUROC 0.64 vs 0.68 with the fact-checkers' own articles), 0.66 with Google Fact Check added.
- C2: the Trust Score matches the text model on calibration and abstains on about 45-49% of claims; evidence only
  helps where a fact check exists (8% of claims).
- C3: SHAP explanations are faithful; low explanation-evidence agreement did **not** flag errors beyond model confidence.
- C4: in-dataset scores overstate real accuracy (shortcuts: LIAR2 metadata, Fakeddit subreddit = label).
- New this week: fact-check verdicts (19/19 agreement), out-of-domain check (AUROC 0.96), Gemini evidence judge reading
  Wikipedia opening paragraphs. On untouched test sets the whole app got 93 of 95 plain facts, invented claims and true
  Sri Lanka claims right, 0 wrong, 2 sent to review; on political claims the judge decides only 15% and changes accuracy
  by +0.004. See `results/domain/README.md` and `results/judge/README.md`.
- Paper draft: https://claude.ai/artifact/MCVmwDtPLfkwXGrCMG839K

## Known limits (say them before they are asked)

- Plain facts depend on the Gemini judge (free quota). Political claims stay hard: the text model is right about 70%.
- Image models: CLIP fusion on Reddit data (a Reddit-style shortcut); no Grad-CAM image heatmaps.
- Sri Lankan claims: too few real examples (14) for firm conclusions.
- Free Gemini quota and Wikipedia rate limits can make live evidence thin.

## Before real users

Narrow the Atlas IP list from 0.0.0.0/0, rotate the database password, delete `backend/.env.atlas.bak` (another
person's credentials).
