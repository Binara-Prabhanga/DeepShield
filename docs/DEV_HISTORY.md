# DeepShield development history (what we tried, what failed, what we changed)

Rebuilt 2026-11-25 from the files in the repository. **There is no git history to quote:** `git rev-parse` says the project is not a git repository
(the project owner said on 2026-11-23 that git is not needed), so there are no commit hashes, branches, tags, authors or reverted commits.
**Commit hash: NOT AVAILABLE for every row.** The evidence column gives file paths, file modification times, and result-folder names instead.
Result folders are named `YYYYMMDD-HHMMSS` (the time the run started), which gives a reliable order of events.

How to read the evidence:
- **[file]** = something you can open in the repo (a README, config, manifest, result folder, code comment, file date).
- **[notes]** = recorded only in the assistant's own project notes (`.claude/.../memory/*.md`), not in a repo file. Treat these as hearsay until you confirm them.
- "Reason: ask team" = no file states why the change was made.
- File dates are the last modification time, so an edited file shows its latest edit, not its creation. Files first written on 2026-11-11 and edited later show the later date.

Paths: `R:` = `results/`, `B:` = `backend/`.

---

## 1. Timeline

| Date range | What we tried | What happened | Evidence | What we changed next |
|---|---|---|---|---|
| 2026-11-11 | **Scaffold.** FastAPI + MongoDB + React app; a *simulated* classifier (random placeholder scores); evidence from Google Fact Check, Tavily and Wikipedia ranked by TF-IDF/FAISS; Gemini for explanation text; JWT login; a contract for model files exported from Colab | Worked as a demo only: scores were random and said so | [file] `B:app/auth.py`, `db.py`, `services/article.py`, `llm.py`, `Dockerfile`, `evaluate.py`, `requirements.txt`, `tests/test_core.py` all dated 2026-11-11 12:59; `requirements-ml.txt` 2026-11-11 18:37; `model_adapter.py` docstring ("random placeholder scores", "Colab-exported artifacts"); `main.py:92` API description | Research phase C1 to C4 on 2026-11-22 |
| 2026-11-22 afternoon | **C1.** `research/eval_stance.py`; NLI evidence stance with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; compared similarity only, whole-passage NLI and sentence-level NLI | Worked. Sentence level + "decisive" rule: macro-F1 0.687 [0.664, 0.710] vs 0.567 (similarity) and 0.615 (passage); McNemar p < 1e-6. Reason given in the README: fact checks restate the claim, so passage NLI is biased to "supports" | [file] `R:c1_stance/20261122-193507/` (5,460.5 s), `R:c1_stance/README.md`; `B:requirements-dev.txt` 17:21; `research/eval_stance.py` 19:45; `tests/test_stance.py` 19:46 | 500-claim trials |
| 2026-11-22 evening | **C1 trials.** Three aggregation rules (decisive / max / mean) on 500 claims; then a larger NLI model (`DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) | Failed (negative result). Rules within noise (test 0.697 / 0.683 / 0.694). Large model: decisive 0.700 vs 0.697 base, rule picked on val scored 0.669 on test, 3 to 4 times slower on CPU. The "refutes" lean (true claims average s_ev about -0.19) was not fixed by either | [file] `R:c1_stance/trials/500claims_base_20261122-200459/` (1,112 s), `.../500claims_large_20261122-220223/` (4,618.7 s), `R:c1_stance/README.md` | Kept base model + sentence + decisive as the final C1 configuration ([notes]: the project owner locked it in on 2026-11-22). Calibration left to C2 |
| 2026-11-22 22:26 to 23:25 | **C2 and C3 built.** `trust.py` (logistic weights + temperature + split-conformal abstention), `eval_trust.py`; `explain_audit.py` (EEA), `eval_explain.py`; both wired into the API through `services/assessment.py`, each off until configured | Built and tested with a **TF-IDF placeholder** text signal (no trained model yet). Placeholder results: Trust Brier 0.182 vs text 0.191, macro-F1 0.704; SHAP faithful (comprehensiveness 0.146 vs random 0.016); signed-EEA AUROC for errors 0.579 (weak) | [file] `R:c2_trust/20261122-222702/`, `R:c3_explain/20261122-224233/`; `B:tests/test_trust.py`, `test_eval_trust.py` 22:26; `test_explain_audit.py` 22:40; `test_model_adapter_gradcam.py` 22:40; `test_analysis_contract.py` 23:14; `services/stance.py`, `trust.py`, `explain_audit.py` 23:25 | C4 protocol, then real models |
| 2026-11-22 | **C4 protocol** (`eval_generalise.py`, `data.py`: `claim_key`, `load_liar_unseen`, `load_liar2_temporal`). First run with the TF-IDF placeholder | Showed LIAR2 shortcuts: metadata alone F1 0.694 vs text 0.683; year alone 0.661; transfer LIAR2 to Fakeddit AUROC 0.49 and back 0.46 | [file] `R:c4_generalise/20261122-232153/` and README "What the placeholder showed" | Train real models |
| 2026-11-22 (date from [notes], fix is in code) | **Data leak found and fixed:** the Fakeddit mirror shared text or images between train, val and test | Fixed in `load_fakeddit`: 99 train and 19 val rows dropped; imgur "no longer available" placeholder flagged; publisher-logo images that map to one label noted as an image shortcut | [file] `B:research/data.py:78-130` (`_FAKEDDIT_LATER`, placeholder md5); `R:c4_generalise/README.md` "Data hygiene"; [notes] | `load_fakeddit_official` later does the same de-duplication |
| 2026-11-23 morning | **Model plan changed: Colab BERT-base to local CPU DistilBERT.** `train_text.py` and `train_fusion.py` written; DistilBERT, 2 epochs, 3 seeds; CLIP fusion head on cached features | Worked. The manifests state the reason: "DistilBERT replaces the plan's BERT-base for speed" / "so training fits on a CPU". Metric before/after: **not logged** (no BERT-base run exists) | [file] `B:research/train_text.py`, `train_fusion.py` 13:54; `B:artifacts/*/manifest.json`; `R:c4_generalise/README.md`; `B:research/MODEL_EXPORT.md` still shows a `bert-base-uncased` example with `"lr": 2e-5, "epochs": 3` (the original plan, never updated) | Real runs of C2 to C4 |
| 2026-11-23 | **Official Fakeddit replaces the unofficial mirror for reporting.** `fetch_fakeddit.py` takes a stratified sample (8,841 / 1,777 / 2,807) | Worked. Found that every subreddit maps to one label: a one-feature subreddit probe scores macro-F1 1.000 / AUROC 1.000, so Fakeddit scores mostly measure recognising a community | [file] `B:research/fetch_fakeddit.py` 13:51; `datasets/README.md`; `R:c4_generalise/README.md` point 5 | Report Fakeddit only with that caveat, compare text vs text+image under equal conditions |
| 2026-11-23 | **Real runs with trained models.** C2 (Trust), C3 (SHAP, Grad-CAM), C4 (generalisation) | C2: Trust Brier 0.178 vs 0.185 text, macro-F1 0.722 vs 0.703 (gold evidence); raw DistilBERT ECE 0.077 to 0.016 after T = 1.70. C3: SHAP faithful; **H4 not supported** (EEA AUROC 0.55 vs confidence 0.766). C4: LIAR2 metadata probe 0.694 vs text 0.703; official Fakeddit text 0.845, fusion 0.884. Seed 44 of the LIAR2 model took 313 min instead of about 75 (the laptop slept) | [file] `R:c2_trust/20261123-111702/`, `R:c3_explain/20261123-125219/`, `.../20261123-121344-images/`, `R:c4_generalise/20261123-200523/`; `B:artifacts/*/predictions/seed*/seed_meta.json`; [notes] laptop sleep | Sri Lankan set, Sinhala, live evidence |
| 2026-11-23 afternoon | **Sri Lankan test set.** First a rules-only extraction of claims from FactSeeker fact checks, then Gemini-assisted extraction with a verbatim check, then human review | Rules-only was too noisy (58 suggestions kept in `review_rules.csv`). Gemini-assisted gave 229 of 304 suggestions. Final `labelled.csv`: 212 claims (14 real, 193 misleading, 5 uncertain), reviewed and approved by the researcher 2026-11-23 | [file] `datasets/README.md`; `B:research/collect_srilanka.py` 18:25; `B:tests/test_collect_srilanka.py` 15:52 | Every model's AUROC interval on this set includes 0.5 (too few real claims): reported as inconclusive |
| 2026-11-23 evening | **Sinhala.** Baselines (outlet probe, char and word TF-IDF, LaBSE) and fine-tuned XLM-R | XLM-R failed to beat the simple model: AUROC 0.686 vs 0.748 for character n-grams (about 90 misleading articles per training fold) | [file] `R:sinhala/20261123-203433/`, `R:sinhala/README.md`; `B:research/eval_sinhala.py` 01:02 (11-24) | Char TF-IDF exported and routed in the app on 2026-11-24 (`model_adapter.is_sinhala`, `tests/test_sinhala_routing.py` 01:04) |
| 2026-11-23 to 24 | **Live evidence (Wikipedia only)** through the app's own retrieval, 500 claims, about 2 h CPU. Then refit of the Trust Score on live stance | C1 live AUROC 0.639 vs gold 0.683. Trust (fitted on gold) gave no gain and abstained on 47%. Refit on live evidence still tied the text model (Brier 0.186 vs 0.184). EEA became informative (0.656) but still added nothing beyond confidence | [file] `R:live_evidence/20261123-133939/`, `R:c2_trust_live/20261124-005043/`; `B:research/eval_live.py` | Better evidence: add Google Fact Check |
| 2026-11-24 | **Google Fact Check added to the experiments.** A first run was discarded because Wikipedia rate-limited it (169 of 500 claims). Reran as `all` (PolitiFact dropped) and `strict` (also dropping fact checks of the same claim) | Worked, small effect: only 41 of 500 claims (8%) have a fair fact check. Strict C1 AUROC 0.663 (up from 0.639); Trust Brier 0.185 vs text 0.184 (tie); H4 unchanged | [file] `R:live_evidence/20261124-factcheck-raw`, `...-all`, `...-strict`; `R:c2_trust_live_fc/{all,strict}/`; `B:research/merge_factchecks.py` 11:12; `B:tests/test_eval_live.py` 11:11 | Use the strict numbers in the paper (other outlets re-check PolitiFact's own claims = label leakage). App's `trust_config.json` left unchanged |
| 2026-11-24 | **App switched from simulation to real models** (`MODEL_MODE=model`, paths to the trained models, stance model, Trust config). MongoDB moved from localhost to Atlas | Worked end to end. Leftover "simulated" wording remains in PDF/CSV/integrations (see section 2) | [file] `B:.env` keys (read, not copied); `B:app/main.py` strings; [notes] MongoDB was not running locally; an old backup connection file belongs to another person's cluster | Fix wrong verdicts seen in a live test |
| 2026-11-24 | **Open TODO comments removed.** `extract_claim` picks the most check-worthy sentence for text over 400 characters; optional `EMBEDDING_MODEL` for semantic evidence ranking (default off) | Done. Semantic ranking is off because the Trust Score was fitted on TF-IDF relevance and would need refitting | [file] `B:app/services/preprocess.py` 19:58; `B:tests/test_claim_and_embeddings.py` 22:48; `B:app/services/evidence.py` (`_semantic_vectors` docstring); `B:.env.example` 20:24 | |
| 2026-11-24 evening | **Live failure and two fixes.** A true claim (Tokyo Olympics held in 2021) and an invented one (the Moon renamed after a cricketer) both scored "Potentially misleading, 13": the text model learned US political claims and knows no facts. Fix 1: a same-claim fact-check rating decides the score (92 or 8). Fix 2: an out-of-domain check sends unfamiliar claims to "Needs review" | Fix 1: agrees with the LIAR2 label on 19 of 19 same-claim ratings, but covers about 4% of claims. Fix 2: AUROC 0.96 (Fakeddit) and 0.955 (Sri Lankan). Tried and dropped: TF-IDF word overlap (AUROC 0.74 / 0.68); nearest single claim, k = 1 (0.955 / 0.932; Olympics claim missed). Chosen: mean of the 25 nearest claims | [file] `R:domain/README.md`, `R:domain/20261124-231336/`; `B:app/services/factcheck_verdict.py`, `domain.py`, `research/fit_domain.py` (23:10) | Gemini judge |
| 2026-11-24 night | **Gemini evidence judge, round 1.** Gemini reads the retrieved snippets and answers supported / refuted / not enough evidence | Mostly failed to help: it decided only 12% of 350 LIAR2 claims (free quota ran out before 500); "refuted" 34 of 34 right, "supported" 5 of 8 | [file] `R:judge/20261124-203455/`, `R:judge/README.md` round 1 | Better evidence text |
| 2026-11-25 00:00 to 04:00 | **Judge rounds 2 and 3.** (a) Wikipedia opening paragraphs instead of search fragments; (b) prompts `strict`, `knowledge`, `knowledge_kind`, `plausibility`; (c) key and model rotation; (d) 30 s deadline and start-up warm-up | (a) plain facts 57 of 60 decided, all right. (b) `knowledge_kind` (tag claims as general vs political) mislabelled political claims (3 of 5 right) and was dropped; `plausibility` refuted 20 of 20 invented claims and supported 15 of 15 true Sri Lankan claims. Two other ideas had no link to truth and were not used: coverage of the Wikipedia title in the claim (error 0.30 vs 0.31) and detail coverage in one passage (AUROC 0.477/0.48). (c) needed because free quota is per model per day | [file] `R:judge/*` (18 run folders, one without `config.json`; the variant is in each config); `B:app/services/judge.py` 03:35; `B:tests/test_judge.py` 03:23; `R:judge/README.md` | Whole-app tests on plain facts |
| 2026-11-25 01:40 to 04:30 | **Whole-app test on plain facts** (`eval_facts_app.py`) | Progress over four runs on the untouched test half (60 facts): 49 right / 2 wrong / 9 review (014019), 50 / 2 / 8 (025924), **58 / 0 / 2** (031447, repeated 042332). Between the second and third run `factcheck_verdict.py` was edited (03:01): short claims (4 content words or fewer) now need Jaccard 0.75 instead of 0.5, because "Mount Fuji is in Japan" matched a review of an eruption and the Burj Khalifa matched a review of a fake photo. Invented claims 20 of 20; true Sri Lankan 15 of 15 | [file] `R:facts_app/20261125-014019`, `-025924`, `-031447`, `-042332`, `-042709`, `-042932`; `B:app/services/factcheck_verdict.py` 03:01; `B:tests/test_factcheck_verdict.py` 03:01 | UI check, demo notes |
| 2026-11-25 00:11 to 04:30 | **Front end and demo.** `App.jsx` updated: real-model wording, assessment notes, stance badges only on fact-check items, SHAP word chips, "TRUST SCORE / ESTIMATE" gauge; `DEMO.md` with start commands and eight demo inputs; UI test with a headless browser of 7 claims | Worked: 7 of 7 as expected (Olympics 85, Moon 13, Fuji 85, Eiffel 92, political claim "Needs review", microchips 8, Australia capital 20) | [file] `frontend/src/App.jsx` 00:11; `DEMO.md` 04:30; `B:tests/conftest.py` 00:16 (pins simulation mode so tests ignore `.env`); [notes] UI results | Final checks |
| 2026-11-25 (later) | **Blind set (72 new claims, 5 to 13 words) and three robustness fixes** | Blind result: 66 right, 0 wrong, 6 "Needs review" (3 had no judge answer, 3 judge said not enough evidence). Cause of the empty answers found: Wikipedia returned HTTP 429 (rate limit), so the app got no evidence and the judge had nothing to read; also `gemini-3.5-flash` was timing out and `gemini-3.1-flash-lite` returned 503 | [file] `R:facts_app/20261125-065216/`; `B:research/judge_facts.csv` (blind rows); `B:app/services/evidence.py` 22:20 (retry up to 2 times, at most 10 s each, on 429); `B:app/services/assessment.py` 22:08 (note "The Gemini evidence judge gave no answer ..."); `B:tests/test_assessment.py` 22:09; `B:.env` `JUDGE_MODEL` order (working lite models first) | After the fix six hand-picked statements (3 invented, 3 real) all came out right |

---

## 2. Topics you asked about

### 2.1 Model choices that were replaced

| Replaced | By | Metric before / after | Evidence |
|---|---|---|---|
| BERT-base (plan; example in `MODEL_EXPORT.md`) | DistilBERT, 2 epochs, 128 tokens | **Not logged.** No BERT-base result exists; reason in manifests: speed on CPU | `B:artifacts/*/manifest.json`, `MODEL_EXPORT.md` |
| Colab training | Local CPU training | n/a | `train_text.py` docstring line 8 |
| TF-IDF placeholder text signal | Trained DistilBERT | C2 Trust Brier 0.182 to 0.178; macro-F1 0.704 to 0.722; temperature 1.00 to 1.70 | `R:c2_trust/20261122-222702` vs `20261123-111702` |
| DeBERTa-v3-large NLI (tried) | Stayed on DeBERTa-v3-base | 0.700 (large) vs 0.697 (base) on test | `R:c1_stance/trials/` |
| Passage-level NLI | Sentence-level, "decisive" rule | 0.615 to 0.687 macro-F1 | `R:c1_stance/20261122-193507/` |
| Unofficial Fakeddit mirror | Official Fakeddit v2.0 sample (for reporting) | Text 0.750 (mirror) vs 0.845 (official); fusion 0.784 vs 0.884; not comparable (different data) | `R:c4_generalise/README.md` |
| XLM-R for Sinhala | Character n-gram TF-IDF + calibrated LR | AUROC 0.686 to 0.748 | `R:sinhala/README.md` |
| Nearest-neighbour k = 1 domain check | k = 25 | AUROC 0.955 / 0.932 to 0.961 / 0.955 (Fakeddit / Sri Lankan) | `R:domain/README.md` |
| Judge prompt `strict` | `knowledge`, then `plausibility` | See timeline | `R:judge/README.md` |

Fusion method: the concatenation head `[t, i, t*i, |t-i|]` is documented in the `model_adapter.py` contract (the date of its first version is not recorded); no other fusion method was tried in the repo (Reason: ask team).
Threshold history is in 2.3.

### 2.2 Evidence retrieval changes

| Change | When | Evidence / result |
|---|---|---|
| Providers in the scaffold: Google Fact Check, Tavily, Wikipedia; top 8 by TF-IDF cosine through FAISS, fact checks first | 2026-11-11 (or earlier) | `B:app/services/evidence.py` |
| Tavily: **never configured** (no key in the experiments or the app) | throughout | `R:live_evidence/README.md` ("Tavily and Google Fact Check keys are not set"); `B:.env` key empty |
| Google Fact Check key added | 2026-11-24 | `R:c2_trust_live_fc/README.md` |
| Wikipedia keyword queries: stopwords and reporting words removed, a 4-word and a 6-word query, because Wikipedia search needs every word to match | before 2026-11-22 (date not recorded) | `evidence.py` `wikipedia_queries` docstring; the change itself: **Reason / date: ask team** |
| Leak filters in experiments: PolitiFact removed; then same-claim fact checks removed (Jaccard >= 0.5, "strict") | 2026-11-23 and 24 | `R:c2_trust_live_fc/README.md`; `B:research/eval_live.py` |
| Optional sentence-embedding ranking (`EMBEDDING_MODEL`), default **off** | 2026-11-24 | `evidence.py`; off because Trust was fitted on TF-IDF relevance |
| Wikipedia opening paragraphs as evidence text: `WIKIPEDIA_INTRO` setting (off; did not fix the NLI confusing the Games' name with the year [notes]); used **for the judge only** | 2026-11-24 and 25 | `evidence.py` (`_with_intros`, `with_wikipedia_intros`); `R:judge/README.md` round 2 |
| Wikipedia rate-limit handling: retry on HTTP 429 | 2026-11-25 | `evidence.py` (this was a silent failure: no evidence, no error) |
| Same-claim match stricter for short claims (0.75 when 4 content words or fewer) | 2026-11-25 | `factcheck_verdict.py:15-17,41-49` |

### 2.3 Trust Score and "Needs review" over time

1. **Scaffold (2026-11-11):** a single score 0 to 100 from the placeholder, labelled by fixed thresholds: 67 or more "Likely authentic", below 40 "Potentially misleading", else "Needs review" (`B:app/main.py:211`). Still in the code as the fallback when no Trust model is configured.
2. **2026-11-22:** learned Trust Score (logistic regression on text, image and evidence signals, temperature scaling) and **split-conformal abstention** (alpha 0.1) replace fixed thresholds when `TRUST_MODEL_PATH` is set (`trust.py`, `decide`).
3. **2026-11-23:** the learned model is refit on the real DistilBERT signal: weights text 0.90, evidence 0.73, bias 0.17, T about 1.00, qhat 0.683 (`R:c2_trust/20261123-111702/trust_config.json`).
4. **2026-11-24:** a refit on live evidence was tried and **not adopted** (no gain); the app keeps the gold-evidence-fitted config (`R:c2_trust_live/README.md`, `R:c2_trust_live_fc/README.md`).
5. **2026-11-24 and 25:** `decide()` extended with overrides in priority order: fact-check verdict, judge "refuted" (cap 20), judge "supported" for out-of-domain claims (at least 85), judge "supported" with a low score (review), out-of-domain (review). Reason for "supported" raising scores only out of domain: political "supported" was right only 9 of 13 times (`R:judge/README.md`).

### 2.4 Bugs that changed results

| Bug or trap | Effect | Fixed / handled | Evidence |
|---|---|---|---|
| Fakeddit mirror splits shared text/images | Inflated in-dataset scores | De-duplicate later splits (99 train, 19 val rows dropped) | `data.py`; `R:c4_generalise/README.md` |
| LIAR2 speaker credit-history count columns (a known leak) | Would leak labels | Left out of `load_liar2` on purpose | `data.py` `load_liar2` docstring |
| LIAR claims overlapping LIAR2 (about 75%) | Cross-dataset test not unseen | `liar_unseen` and the temporal split remove repeated claims (`claim_key`) | `data.py`; `R:c4_generalise/README.md` |
| PolitiFact items and same-claim fact checks in live evidence | Label leakage (LIAR2 comes from PolitiFact) | Dropped in experiments ("strict") | `R:c2_trust_live_fc/README.md` |
| Grad-CAM on the last ViT block: the last block's patch tokens never reach the pooled output, so their gradients are zero (an all-zero map) | Empty maps | Hook moved to the second-to-last block (`model_adapter.py` comment before `_multimodal_scores`; `tests/test_model_adapter_gradcam.py`, dated 2026-11-22 22:40). **When it was first noticed: ask team** | `B:app/services/model_adapter.py:~189` |
| Wikipedia rate limit during the first fact-check run | 169 of 500 claims lacked evidence; run discarded | Reran as `all` and `strict`; Wikipedia items reused from the earlier complete run | `R:c2_trust_live_fc/README.md`; `R:live_evidence/20261124-factcheck-raw` |
| Short claims matched unrelated fact checks | Two wrong verdicts in the 60-fact test (Fuji, Burj Khalifa) | Jaccard 0.75 for 4 content words or fewer | `R:facts_app/20261125-025924` (2 wrong) vs `-031447` (0 wrong) |
| Judge hitting free-tier quota | Run stopped at 350 of 500 claims; 20 of 60 facts unanswered in-app | Key + model rotation, longer rest after a per-day 429, graceful skip | `R:judge/20261124-203455`; `judge.py` |
| Wikipedia 429 gave empty evidence silently | 5 of 6 hand-picked statements ended "Needs review" | Retry on 429; "judge unavailable" note | 2026-11-25 row above |
| Long first request vs the web UI's 100 s limit | First analysis timed out | Models warmed up in the background at start-up; judge deadline 30 s | `main.py:59-90`; `assessment.py:30` |
| Test claims shorter than 25 characters rejected | UI tests never redirected | Longer test claims; limit unchanged | `main.py:259` |

### 2.5 Dataset changes

- Unofficial Fakeddit mirror (`fakeddit_9k`) used for building; replaced by the official v2.0 sample for reporting on 2026-11-23 (`datasets/README.md`).
- LIAR2 is the main text dataset; original LIAR kept only for the unseen-claims test.
- Temporal re-split of LIAR2 added (train/val/test before 2018, 2018 gap, future 2019+).
- Sinhala: `Corpus.xlsx` used because `Corpus.csv` has broken encoding; the paper reports 3,576 articles but the repo has 3,000 (the repo's README says parts were corrupted and restored).
- Sri Lankan FactSeeker set built on 2026-11-23 (rules to Gemini-assisted to human-reviewed, 212 claims).
- `judge_facts.csv` built 2026-11-25 (262 rows) for the judge and the whole app.
- NewsCLIPpings was planned (the datasets README says "need manual access") but never obtained.

### 2.6 Front-end and back-end redesigns

- Back end: one `main.py` with an `assess()` layer added on 2026-11-22 so each research part can be switched on by a setting; lifespan start-up warm-up added 2026-11-25. No framework change is recorded.
- Front end: a single `App.jsx` (95 long lines). Only the final version exists; **the earlier version and the reasons for its layout: ask team.** The visible changes recorded on 2026-11-25 are the wording changes listed in the timeline.
- Docker: `Dockerfile` unchanged since 2026-11-11 and cannot run the real models (installs only `requirements.txt`).

---

## 3. The five most important turning points

1. **2026-11-22: sentence-level NLI beats whole-passage NLI and similarity** (macro-F1 0.687 vs 0.615 and 0.567). It made evidence stance a usable signal and set the C1 design that the rest (Trust Score, EEA) builds on. Evidence: `R:c1_stance/20261122-193507/`.
2. **2026-11-23: the plan moved from Colab BERT-base to local CPU DistilBERT, and the real models arrived.** Every C2 to C4 result, the app's real-model mode and the 3-seed manifests depend on it. Evidence: `B:artifacts/*/manifest.json`, `R:c2_trust/20261123-111702/`.
3. **2026-11-23: the shortcut discoveries.** LIAR2 metadata alone nearly matches the text model (0.694 vs 0.703), and Fakeddit labels equal subreddits (probe F1 1.000). They turned C4 from a side test into the framing for how every number must be reported. Evidence: `R:c4_generalise/20261123-200523/`.
4. **2026-11-24: live evidence did not help the Trust Score** (Brier 0.185 vs 0.184 even with Google Fact Check; only 8% of claims have a fair fact check). It moved the project's message to "the benefit depends on evidence quality" and led to the app-level fixes. Evidence: `R:c2_trust_live_fc/README.md`.
5. **2026-11-24 to 25: the Olympics-versus-Moon failure and the rules that followed** (out-of-domain check, fact-check verdict, Gemini judge on Wikipedia opening paragraphs). The whole-app plain-fact test went from 49 right / 2 wrong / 9 review to 58 / 0 / 2, and the blind set gave 66 / 0 / 6. Evidence: `R:facts_app/`, `R:judge/README.md`, `R:domain/README.md`.

---

## 4. What is missing from this history (and how to fill it)

- **Commit hashes, branches, tags, authors, reverted commits, large deletions:** NOT AVAILABLE (no git). If you want hashes in the report, `git init` and commit the current state only; history cannot be recovered.
- **Original plan or proposal text** (RQ wording, BERT-base plan, NewsCLIPpings, SpotFake baseline): not in the repo.
- **Earlier versions of files** (`App.jsx`, `evidence.py` before the Wikipedia query change, the first `fit_domain` run `20261124-202223/` that the domain README cites but whose folder no longer exists): not kept. Only `results/domain/20261124-231336/` remains.
- **The exact reasons** behind: the Wikipedia query design, when Grad-CAM zero gradients were first seen, the front-end layout, the choice of the concatenation fusion head. Marked "ask team" above.
- **Old `results/facts_app/` run folders** are kept (seven runs); two of them (`20261125-014019`, `-025924`) are the early runs with wrong verdicts and are evidence for the progress above, so do not delete them before the report is written.
