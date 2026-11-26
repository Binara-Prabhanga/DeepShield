# DeepShield: facts and numbers for the final report and revised proposal

Collected 2026-11-25 by reading the repository only. Nothing here is estimated. Each number names the file it came from
(path, plus line or result-file name). Where something was not run or does not exist, it says **NOT AVAILABLE** and what is missing.
No application code was changed. The only files created are this one and `docs/DEV_HISTORY.md`.
Two numbers (marked "computed") were calculated from the saved per-claim prediction files in `backend/artifacts/*/predictions/`; the formula is in section 4.1.

Paths are relative to the project root `DeepShield/`. `R:` = `results/`.

Reading guide: the project plan or proposal document is **not in the repository**, so "RQ1 to RQ5" are taken from your message, not from a file.
The repository calls the four contributions C1 (NLI stance), C2 (calibrated Trust Score + abstention), C3 (explanation audit, EEA) and C4 (shortcut-aware generalisation).

---

## 1. System and implementation

### 1.1 Folder structure (2 levels)

```
backend/
  Dockerfile, .dockerignore, requirements.txt, requirements-ml.txt, requirements-dev.txt, evaluate.py
  app/        __init__.py, auth.py, config.py, db.py, main.py, services/
  app/services/  article.py, assessment.py, domain.py, evidence.py, explain_audit.py, factcheck_verdict.py,
                 judge.py, llm.py, model_adapter.py, preprocess.py, stance.py, trust.py
  artifacts/  text-distilbert-base-uncased-{liar2, liar2_temporal, fakeddit, fakeddit_official}/
              mm-clip-fusion-{fakeddit, fakeddit_official}/   text-sinhala-char-tfidf/   domain-liar2/   cache/
  research/   (21 files, see 1.3)      tests/ (23 test files + conftest.py)
frontend/
  package.json, (vite config), src/  App.jsx, api.js, main.jsx, styles.css
datasets/     fakeddit_9k, fakeddit_official, liar2, liar_binary_ukplab, sinhala_lirneasia, srilanka_local, README.md
results/      c1_stance, c2_trust, c2_trust_live, c2_trust_live_fc, c3_explain, c4_generalise, domain, facts_app, judge, live_evidence, sinhala
DEMO.md
```
Source: directory listing 2026-11-25. Notes: `frontend/src/App.jsx` holds all pages in one 95-line file (long lines).
`backend/atlas-credentials.env` and `backend/.env.atlas.bak` exist and contain credentials; they were not opened and should not be shipped.

### 1.2 API endpoints (`backend/app/main.py`)

| Method | Path | Line | Purpose |
|---|---|---|---|
| GET | `/api/health` | 125 | Database ping; returns status, database state, `model_mode`, version |
| POST | `/api/auth/register` | 140 | Create account (Argon2 hash), returns a JWT |
| POST | `/api/auth/login` | 156 | Check password, return a JWT |
| POST | `/api/auth/logout` | 164 | Revoke the token's `jti` |
| GET | `/api/auth/me` | 173 | Current user |
| PATCH | `/api/auth/profile` | 178 | Change display name |
| POST | `/api/analyses` | 242 | Run a full analysis (text, optional URL, optional image) and save it |
| GET | `/api/analyses` | 301 | List the user's analyses (page, limit, search `q`, category) |
| GET | `/api/analyses/{id}` | 322 | One analysis |
| PATCH | `/api/analyses/{id}/notes` | 327 | Save analyst notes |
| DELETE | `/api/analyses/{id}` | 338 | Delete one analysis |
| GET | `/api/dashboard/overview` | 347 | Totals, category counts, 14-day activity, recent 5 |
| GET | `/api/integrations` | 395 | Which providers have a key set |
| GET | `/api/reports/export.csv` | 408 | History as CSV |
| GET | `/api/analyses/{id}/report.pdf` | 437 | One analysis as PDF |

Total: 15 endpoints. Request order inside `POST /api/analyses` (lines 242 to 306): validate input, optional URL extraction, `run_models`,
`extract_claim`, `gather_evidence`, `assess` (stance, Trust Score, EEA, fact-check verdict, out-of-domain, Gemini judge), `decide`, `explain`, save.

### 1.3 Service and research files

| File | What it does |
|---|---|
| `app/services/model_adapter.py` | Loads and runs the text model (DistilBERT), SHAP words, the CLIP fusion head with Grad-CAM, the Sinhala model; simulation mode (line 266 `run_models`) |
| `app/services/stance.py` | **C1.** NLI stance of each evidence item, sentence by sentence; fact-check ratings override NLI; `evidence_score` (line 222) |
| `app/services/trust.py` | **C2.** Calibrated Trust Score (logistic weights + temperature) and split-conformal abstention |
| `app/services/explain_audit.py` | **C3.** Explanation-Evidence Agreement (EEA, line 69) |
| `app/services/assessment.py` | Runs the optional parts (stance, trust, EEA, fact-check verdict, domain check, judge); each failure becomes a note |
| `app/services/evidence.py` | Evidence from Google Fact Check, Tavily, Wikipedia; TF-IDF/FAISS relevance ranking, top 8; Wikipedia 429 retry |
| `app/services/factcheck_verdict.py` | A same-claim fact-check rating sets the score to 92 (true) or 8 (false) |
| `app/services/domain.py` | Out-of-domain check: mean cosine to the 25 nearest LIAR2 training claims (MiniLM) |
| `app/services/judge.py` | Gemini evidence judge (supported / refuted / not_enough_evidence), key and model rotation |
| `app/services/llm.py` | Gemini 2 to 4 sentence explanation of what the citations show |
| `app/services/article.py` | URL validation (SSRF checks) and article extraction through Tavily |
| `app/services/preprocess.py` | `clean_text`, `extract_claim` (most check-worthy sentence for text over 400 characters) |
| `app/auth.py`, `app/db.py`, `app/config.py` | Password and token handling, MongoDB connection and indexes, settings |
| `research/` | `data.py` (loaders), `train_text.py`, `train_fusion.py`, `eval_stance.py`, `eval_trust.py`, `eval_explain.py`, `eval_generalise.py`, `eval_sinhala.py`, `eval_live.py`, `eval_judge.py`, `eval_facts_app.py`, `fit_domain.py`, `fetch_fakeddit.py`, `collect_srilanka.py`, `merge_factchecks.py`, `predict_export.py`, `check_export.py`, `judge_facts.csv`, `MODEL_EXPORT.md` |

**C1 stance model (exact Hugging Face ID):** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`
(`R:c1_stance/20261122-193507/config.json`; `STANCE_MODEL` in `backend/.env`).
Premise = evidence sentence, hypothesis = claim; entailment = supports, contradiction = refutes, neutral = unrelated (`stance.py` lines 1 to 20).
Settings: sentence-level, "decisive" aggregation (the sentence with the highest max(entailment, contradiction) decides), at most 12 sentences (`stance.py:112`),
sentences that restate the claim at 0.6 word overlap or more are skipped (`stance.py:119`).
Evidence score: `s_ev = sum_i r_i (P_sup_i - P_ref_i) / sum_i r_i` in [-1, 1] (`stance.py:222` and header comment), `r_i` = retrieval relevance.

### 1.4 MongoDB (`backend/app/db.py`, `backend/app/main.py`)

Database name `deepshield` (`MONGODB_DB`). Indexes are created at startup (`db.py` lines 19 to 22).

| Collection | Fields (from the code that writes them) | Index |
|---|---|---|
| `users` | `_id` (uuid), `email` (lower case), `name`, `password` (Argon2 hash), `role` ("analyst"), `created_at` (`main.py:147`) | unique `email` |
| `revoked_sessions` | `_id` (token `jti`), `expires_at` (`main.py:166`) | TTL on `expires_at` (expireAfterSeconds 0) |
| `analyses` | `_id`, `user_id`, `created_at`, `title`, `text`, `claim`, `url`, `source_domain`, `image_data` (WebP data URL), `image_metadata`, `metadata`, `prediction`, `trust_score`, `model_simulated`, `model` (text_probability_real, image_text_alignment, fused_probability_real, mode, explanation_tokens, gradcam_available, shap_available, gradcam_image, notes), `signals`, `evidence`, `providers`, `explanation`, `evidence_score`, `trust`, `explanation_audit`, `assessment_notes`, `explanation_source`, `notes`, `tags` (`main.py:285` to `main.py:297`) | `(user_id, created_at desc)`, `(user_id, prediction)` |

Not stored in the saved record: the fact-check verdict, the out-of-domain flag and the Gemini judge verdict (they are in `assessment`, shown as notes in `assessment_notes`).
The code also checks for legacy fields `is_seed` and `is_demo` and a legacy demo account (`LEGACY_SEED_EMAIL`), left from the earlier scaffold.

### 1.5 Security measures that exist in the code

| Measure | Where | Detail |
|---|---|---|
| Password hashing | `app/auth.py:12` | `pwdlib` `PasswordHash.recommended()` = Argon2 (`argon2-cffi 25.1.0`) |
| Password rules | `main.py:105`, `main.py:145` | 8 to 128 characters, must contain a letter and a digit |
| Token | `auth.py:25-27` | JWT, HS256, expires in **7 days**, has `jti`; logout stores the `jti` in `revoked_sessions` (TTL) |
| Secret | `config.py` | Default `DEV_ONLY_CHANGE_ME_BEFORE_DEPLOYMENT` unless `JWT_SECRET` is set in `.env` (it is set in the current `.env`) |
| Input limits | `main.py:259`, `main.py:109`, `main.py:121`, `main.py:303` | Text 25 to 16,000 characters; name 2 to 70; notes up to 3,000; search `q` up to 150; page size 1 to 50 |
| Image checks | `main.py:185-208` | Max 5 MB; JPEG, PNG or WebP only; max 24,000,000 pixels (decompression-bomb check); verified by Pillow |
| EXIF stripping | `main.py:199-203` | `exif_transpose`, `img.info.clear()`, re-encoded as WebP (no metadata kept); `has_exif` is only recorded as a flag |
| SSRF checks | `app/services/article.py:7-30` | http/https only, ports 80/443 only, no credentials in the URL, length 2,048, blocks `localhost`, `.local`, `.internal`, non-global IP literals. The server never fetches the URL itself; extraction goes through Tavily |
| CSV-injection guard | `main.py:415` | Cells starting with `=`, `+`, `-`, `@` get a leading `'` |
| PDF escaping | `main.py` (export_pdf) | User text passed through `xml.sax.saxutils.escape` |
| Ownership checks | `main.py:315-320` | Every analysis query filters by `user_id` |
| CORS | `main.py:94` | Only origins in `FRONTEND_ORIGINS`; methods GET/POST/PUT/PATCH/DELETE; headers Authorization and Content-Type |
| Secrets | `.env` | API keys read from `.env`; never returned to the browser (providers return only a status string) |

**Not found in the code:** rate limiting or login-attempt throttling; HTTPS enforcement; email verification; password reset.

### 1.6 Front-end screens (`frontend/src/App.jsx`, route table at line 68)

| Route | Screen |
|---|---|
| (not signed in) | `AuthScreen`: sign in / register |
| `/` | `Dashboard`: totals, average score, needs-review count, evidence found, category donut, 14-day activity chart, recent analyses |
| `/analyze` | `Analyze`: paste text, URL or image, submit |
| `/analysis/:id` | `AnalysisDetail`: score gauge ("TRUST SCORE" / "ESTIMATE"), label, evidence list with stance badges (fact-check items only), SHAP word chips, Grad-CAM image, assessment notes, explanation, notes, PDF export |
| `/history` | `HistoryPage`: searchable, filterable, paged list; CSV export |
| `/insights` | `Insights`: "Workspace insights", volume and classification distribution of your saved analyses (data from `/api/dashboard/overview`) |
| `/integrations` | `Integrations`: which providers are connected |
| `/settings` | `SettingsPage`: profile |

Libraries: React 19, react-router-dom 7, recharts 2.15, axios, lucide-react, date-fns; build tool Vite 6 (`frontend/package.json`).

### 1.7 Libraries and versions

Declared (`backend/requirements.txt`): fastapi >=0.115,<1; uvicorn[standard] >=0.30,<1; pymongo >=4.10,<5; pydantic >=2.8,<3; pydantic-settings >=2.4,<3; python-multipart >=0.0.9;
PyJWT >=2.8,<3; pwdlib[argon2] >=0.2,<1; httpx >=0.27,<1; beautifulsoup4 >=4.12,<5; Pillow >=10,<12; scikit-learn >=1.5,<2; faiss-cpu >=1.9,<2; numpy >=1.26,<3;
reportlab >=4.2,<5; email-validator >=2.2,<3; python-dotenv >=1.0,<2.
`backend/requirements-ml.txt`: torch >=2.2; transformers >=4.41,<5; shap >=0.45.
Frontend (`frontend/package.json`): axios ^1.7.9, date-fns ^4.1.0, lucide-react ^0.468.0, react ^19.0.0, react-dom ^19.0.0, react-router-dom ^7.1.0, recharts ^2.15.0; dev: @vitejs/plugin-react ^4.3.4, vite ^6.0.5.

**Installed in the working environment** (`backend/.venv`, `pip list`, 2026-11-25): Python 3.13.14; fastapi 0.162.4; uvicorn 0.54.0; pymongo 4.19.0; pydantic 2.14.3; pydantic-settings 2.16.2;
PyJWT 2.15.1; pwdlib 0.3.1; argon2-cffi 25.1.0; httpx 0.28.1; beautifulsoup4 4.15.0; Pillow 11.3.0; scikit-learn 1.9.1; scipy 1.18.1; numpy 2.5.4; pandas 3.1.1; faiss-cpu 1.16.0;
reportlab 4.5.1; torch 2.15.0+cpu; transformers 4.57.6; shap 0.54.1; openpyxl 3.1.5; pytest 9.2.0.
Mismatches to check before submitting: `backend/Dockerfile` uses `python:3.12-slim` and installs only `requirements.txt`, so the Docker image has no torch, transformers or shap and cannot run model mode;
`research/` and `tests/` are not copied into it. `openpyxl` (used to read the Sinhala corpus) is installed but not listed in any requirements file.

### 1.8 `MODEL_MODE`

Current value in `backend/.env`: **`model`** (`.env` is not copied here; checked by reading the key).
Meaning (`app/services/model_adapter.py:1-5` and 266-318): `simulation` or `demo` return random placeholder scores (`mode: "simulated"`, never evidence of credibility); **any other value** runs the real models and needs `requirements-ml.txt`.
The code default in `config.py` is `simulation`. Real mode also uses these settings (all set in the current `.env`): `TEXT_MODEL_PATH`, `MULTIMODAL_MODEL_PATH`, `SINHALA_MODEL_PATH`, `STANCE_MODEL`, `TRUST_MODEL_PATH`, `DOMAIN_MODEL_PATH`, `JUDGE_ENABLED`, `JUDGE_MODEL`, `JUDGE_VARIANT`.
`TAVILY_API_KEY` is **not set** (no news search and no URL-only extraction); Google Fact Check, Gemini and MongoDB are set.

**Leftover "simulated" wording in the code (the app now runs real models):**
`main.py:92` API description; `main.py:395-406` `/api/integrations` (Custom BERT + CLIP shown as not connected, disclaimer says simulated);
`main.py:408-436` CSV header "Simulated classification" / "Simulated trust score"; `main.py:437+` PDF text "MODEL SIMULATION", "Simulated verdict", "Simulated trust score". These can mislead a reader of an exported report. (Not changed; you asked for read-only.)

---

## 2. Models and training

### 2.1 Model IDs

| Role | Model | Source |
|---|---|---|
| Text model | `distilbert-base-uncased` fine-tuned (DistilBERT; the plan's BERT-base was replaced for CPU speed) | `backend/artifacts/*/manifest.json` (`base_model`, `notes`) |
| Image-text model | `openai/clip-vit-base-patch32` (frozen) + trained `FusionHead` | `backend/artifacts/mm-clip-fusion-*/model/fusion_config.json` |
| NLI stance | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 1.3 above |
| Larger NLI tried, not used | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | `R:c1_stance/README.md` |
| Sentence embedder (domain check) | `sentence-transformers/all-MiniLM-L6-v2` | `R:domain/README.md` (artifact `domain.joblib`) |
| Sinhala | char 2 to 5-gram TF-IDF + logistic regression, sigmoid-calibrated (not a neural model) | `R:sinhala/README.md` |
| Explanation/judge LLM | Gemini (`gemini-3.8-flash` for the explanation; judge rotates `gemini-flash-lite-latest, gemini-3.5-flash-lite, gemini-3.5-flash, gemini-3.1-flash-lite`) | `config.py`, `.env` |

**What the running app loads (`backend/.env`):** text model `artifacts/text-distilbert-base-uncased-liar2/model` (trained on LIAR2, political claims);
image-text model `artifacts/mm-clip-fusion-fakeddit_official/model`; Trust Score `results/c2_trust/20261123-111702`.

### 2.2 Fusion head and calibration values

- **FusionHead** (`model_adapter.py:114-121`): input = concat[t, i, t*i, |t-i|] of L2-normalised CLIP text and image embeddings (512 each, so 2,048 inputs) -> Linear(2048, 512) -> ReLU -> Dropout -> Linear(512, 2).
- `fusion_config.json`, `mm-clip-fusion-fakeddit_official`: hidden_dim **512**, dropout **0.1**, real_index 1, temperature **0.988778**.
- `fusion_config.json`, `mm-clip-fusion-fakeddit` (mirror): hidden_dim 512, dropout 0.1, temperature 1.035473.
- `calibration.json` (text model, liar2, seed 42): temperature **1.703973**.
- Temperatures per seed 42/43/44 (`manifest.json`): liar2 1.703973 / 1.689438 / 1.652876; liar2_temporal 1.467943 / 1.498689 / 1.365043; fakeddit mirror 1.404072 / 1.358605 / 1.490155;
  fakeddit_official 1.500471 / 1.468437 / 1.507493; fusion mirror 1.035473 / 0.956459 / 1.028238; fusion official 0.988778 / 0.918546 / 1.108265.
- Inference: DistilBERT max 512 tokens at inference (`model_adapter.py:151`); trained at 128. CLIP text truncated at 77 tokens (`model_adapter.py:197`). SHAP: `max_evals=300`, text cut at 2,000 characters, top 12 words (`model_adapter.py:161`).

### 2.3 Training settings

| | DistilBERT text models | CLIP fusion head |
|---|---|---|
| Epochs | 2 | up to 40, early stopping patience 5 |
| Batch size | 16 | 64 |
| Learning rate | 3e-5 | 1e-3 |
| Weight decay | 0.01 | 1e-4 |
| Warm-up | 0.06 | not recorded |
| Max length | 128 tokens | CLIP's own (77 text tokens) |
| Hidden / dropout | n/a | 512 / 0.1 |
| Seeds | 42, 43, 44 (weights saved for seed 42 only) | 42, 43, 44 |
| Temperature | fitted on that seed's val split | fitted on that seed's val split |

Source: `backend/artifacts/*/manifest.json`. All manifests record `data_loader_sha256` (liar2/fakeddit/liar2_temporal: `399f9da4...`; fakeddit_official: `5db2c306...`).

**Hardware:** CPU only. Intel Core i7-1165G7 @ 2.80 GHz, 16 GB RAM (`systeminfo`/CIM, 2026-11-25); `torch 2.15.0+cpu`; Windows 11 Pro. No GPU was used (`train_text.py:8`, manifests "Trained on CPU"). Colab was planned (`model_adapter.py` docstring) but not used for training.

**Training time (minutes, `predictions/seed*/seed_meta.json`):**

| Model | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| DistilBERT liar2 | 76.5 | 74.0 | 313.3 (note: laptop sleep paused long jobs, see memory note) |
| DistilBERT liar2_temporal | 47.9 | 57.5 | 56.1 |
| DistilBERT fakeddit (mirror) | 17.3 | 17.1 | 20.2 |
| DistilBERT fakeddit_official | 26.0 | 31.4 | 32.7 |
| CLIP fusion (both) | **NOT AVAILABLE** (no minutes recorded for the fusion head; it trains on cached CLIP features `backend/artifacts/cache/*.npz`) | | |

Validation macro-F1 per seed (`manifest.json`): liar2 0.7023 / 0.7016 / 0.7044; liar2_temporal 0.6512 / 0.6567 / 0.6338; fakeddit 0.7324 / 0.7531 / 0.7353; fakeddit_official 0.8137 / 0.8208 / 0.8234;
fusion mirror 0.7875 / 0.7950 / 0.7873; fusion official 0.8837 / 0.8802 / 0.8805.

### 2.4 How the Trust Score is computed now

`app/services/trust.py` (header comment, lines 1 to 20; `trust_score` line 169):

```
Trust = sigmoid( ( w_text * logit(p_text) + w_fused * logit(p_fused) + w_evidence * s_ev + b ) / T )
```
Each signal also has a 0/1 "missing" feature (missing signal contributes 0 through its value). p are P(real); `s_ev` is the evidence score in [-1, 1]. Weights are learned by L2 logistic regression (`fit_weights`, C=1.0, seed 42); T by temperature scaling on out-of-fold margins (5 folds, `fit_trust`, line 136).

**The file loaded by the app** (`R:c2_trust/20261123-111702/trust_config.json`, seed-42 model):
`w_A` (text_value) = **0.901074**, `w_B` (fused_value) = **0.0**, `w_C` (evidence_value) = **0.725678**, text/fused/evidence "missing" weights = 0.0, bias `b` = **0.171734**, temperature `T` = **1.000883**, alpha = **0.1**, qhat = **0.6828248735332343**.
So the saved Trust Score uses text + evidence only; the image signal weight is 0 because LIAR2 has no images (`R:c2_trust/README.md`, "Image signal. Absent").
It was fitted on the LIAR2 **gold** evidence (fact-check justification), not live evidence (`trust_config.json` meta; `R:c2_trust_live_fc/README.md` conclusion 3).

**"Needs review":** decided by **split conformal prediction**, not fixed thresholds. `conformal_qhat` (`trust.py:151`) takes the ceil((n+1)(1-alpha))-th smallest nonconformity score `1 - P(true class)` on a calibration half;
the prediction set holds every class with `1 - P(class) <= qhat`; any set that is not exactly one class returns "Needs review" (`prediction_set`, `trust_score`). Alpha = 0.1, qhat = 0.6828.
Fixed thresholds still exist in `main.py:211` (`categorize`): score >= 67 "Likely authentic", < 40 "Potentially misleading", otherwise "Needs review". They apply only when `TRUST_MODEL_PATH` is not set (`decide`, `main.py:219-239`).

**Final decision order in the app (`decide`, `main.py:219-239`):** (1) calibrated Trust Score and its conformal decision; (2) a same-claim fact-check rating overrides: true = 92, false = 8 (`factcheck_verdict.py:18`);
(3) judge "refuted" caps the score at 20; (4) judge "supported" and out-of-domain sets the score to at least 85; (5) judge "supported" and score under 40 = "Needs review" (50); (6) out-of-domain = "Needs review" (50).
Out-of-domain threshold 0.385 (5th percentile of LIAR2 validation similarity, k = 25) (`R:domain/README.md`).

### 2.5 How EEA is computed (`app/services/explain_audit.py`)

`EEA = |TopK_SHAP ∩ words of evidence with the same stance| / k`, **k = 10** (`eea`, line 69). SHAP value > 0 must be backed by *supporting* evidence, value < 0 by *refuting* evidence ("signed"); the unsigned variant accepts either (ablation).
Matching rule (fixed before experiments, header comment): lowercase, letters and digits only, English stopwords and words under 2 characters dropped, light stemming (`stem`, suffix rules -ing, -ed, -ly, -es, -s, final -e, at least 3 characters left), **exact match on stems**.
Top-k is taken after that filtering. "Unrelated" evidence never counts. Returns None when there is no supporting or refuting evidence or no usable SHAP word.

---

## 3. Data

### 3.1 Datasets (`datasets/README.md`, `backend/research/data.py`)

| Dataset | Used for | Rows train / val / test | Label mapping |
|---|---|---|---|
| LIAR2 (`datasets/liar2`) | Main text model, C1 to C3, Trust Score | 18,369 / 2,297 / 2,296 | 6 labels; binary: 0 to 2 misleading, 3 (half-true) **real**, 4 to 5 real (`data.py:182-186`) |
| LIAR (UKP binary) | `liar_unseen` cross-dataset test only (LIAR claims not in LIAR2; about 75% of LIAR is inside LIAR2) | 10,269 / 1,284 / 1,283 raw; `liar_unseen` test n = 339 | binary true/false |
| Fakeddit **mirror** (`fakeddit_9k`, unofficial) | Building and testing only, **not paper numbers** | 4,800 / 2,405 / 1,441 raw; after removing shared text or image: 4,701 / 2,386 / 1,441 | 6-way; binary: label 0 real, others misleading (`data.py:132-170`) |
| Fakeddit **official v2.0** multimodal sample | Multimodal model for reporting | full files 564,000 / 59,342 / 59,319; sample used **8,841 / 1,777 / 2,807** (stratified, seed 42, images that still download, about 93%) | 6-way; binary: label 0 real, others misleading |
| LIAR2 temporal re-split | C4 time test | past (before 2018) split 85 / 7.5 / 7.5; test n = 1,046; **future** (2019+) n = 7,623; 2018 left out | as LIAR2 |
| Sinhala (LIRNEasia `Corpus.xlsx`) | Sinhala model | 3,000 articles, no split; binary task n = 1,113 (CREDIBLE 1,003 vs FALSE + PARTIAL 110); UNCERTAIN 1,887 excluded from binary | CREDIBLE real; FALSE, PARTIAL misleading |
| Sri Lankan test set (`srilanka_local`) | Test only | 304 FactSeeker fact checks; `labelled.csv` = **212 claims** (14 real, 193 misleading, 5 uncertain); 207 score in binary terms | FactSeeker verdicts |
| `research/judge_facts.csv` | Judge and whole-app checks | **262 rows**: 120 plain facts (60 true + 60 false; splits dev 60, test 60), 40 invented claims (fiction_dev 20, fiction_test 20), 30 true Sri Lankan facts (obscure_dev 15, obscure_test 15), 72 blind rows (24 true, 24 false, 12 invented, 12 true local); written by the team | real = 1 / 0 |

Counts for `judge_facts.csv` were counted from `backend/research/judge_facts.csv`. The mirror's de-duplicated counts (4,701 / 2,386 / 1,441) come from a project memory note about `load_fakeddit(drop_overlap=True)`, not from a repo file; the loader is `data.py:86`. Fakeddit: every subreddit maps to exactly one label (`R:c4_generalise/README.md`, point 5).

**Checks for each item you asked about:**
- **NewsCLIPpings: not used. NOT AVAILABLE.** `datasets/README.md`: "Not downloaded (need manual access ... VisualNews access request)". There is no code that reads it. Needs: access approval from the dataset authors, then a loader in `research/data.py` and a training run.
- **Official Fakeddit: used** (sample above; `research/fetch_fakeddit.py`, `load_fakeddit_official`).
- **Temporal split: used** (`load_liar2_temporal`, `data.py:324`; results in `R:c4_generalise/20261123-200523`). Applied to LIAR2 (text only), not to Fakeddit.
- **Sri Lankan test set: built.** 212 claims, source FactSeeker (factseeker.lk, Sri Lanka Press Institute), collected 2026-11-23 by `research/collect_srilanka.py`; verdicts are FactSeeker's; claim texts extracted with AI assistance (gemini-3.5-flash-lite, verbatim check) and approved by the researcher on 2026-11-23 (`datasets/README.md`). 14 real / 193 misleading / 5 uncertain. Skipped sources: factcheck.lk and AFP (403), Fact Crescendo (robots) (memory note).

---

## 4. Results

Mean ± sd over seeds 42/43/44 where the file gives it. Binary: misleading is the positive class unless stated; "real" = P(real).

### 4.1 Detection

| System | Dataset (test n) | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | ROC-AUC | Source |
|---|---|---|---|---|---|---|---|
| BERT only (DistilBERT, title) | official Fakeddit (2,807) | 0.852 ± 0.002 | 0.844 ± 0.003 | 0.846 ± 0.002 | **0.845 ± 0.002** | 0.922 ± 0.001 | computed from `backend/artifacts/text-distilbert-base-uncased-fakeddit_official/predictions/seed{42,43,44}/fakeddit_official_test.csv` at threshold 0.5 (sd with n-1); F1 and AUROC match `R:c4_generalise/20261123-200523/metrics.json` |
| CLIP fusion (title + image) = DeepShield multimodal | official Fakeddit (2,807) | 0.889 ± 0.003 | 0.883 ± 0.004 | 0.887 ± 0.001 | **0.884 ± 0.002** | 0.955 ± 0.000 | computed from `mm-clip-fusion-fakeddit_official/predictions/...`; F1/AUROC match metrics.json |
| DistilBERT (title) | LIAR2 test (2,296) | 0.706 ± 0.005 | 0.702 ± 0.005 | 0.707 ± 0.006 | **0.703 ± 0.005** | 0.783 ± 0.003 | computed from `text-distilbert-base-uncased-liar2/predictions/...liar2_test.csv`; F1/AUROC match metrics.json |
| DistilBERT (title) | mirror Fakeddit (1,441) | 0.847 ± 0.009 | 0.759 ± 0.017 | 0.743 ± 0.019 | 0.750 ± 0.013 | 0.861 ± 0.001 | same method; building/testing only |
| CLIP fusion | mirror Fakeddit (1,441) | 0.877 ± 0.002 | 0.822 ± 0.006 | 0.760 ± 0.002 | 0.784 ± 0.002 | 0.910 ± 0.001 | same method; building/testing only |
| SpotFake-style baseline | **NOT AVAILABLE** | | | | | | No code or results exist (`grep` over backend, results, datasets finds none). Needs: implement (text encoder + image encoder, concatenation head), train on official Fakeddit with 3 seeds; CPU, about 1 to 2 h |
| Zero-shot CLIP | **NOT AVAILABLE** | | | | | | No code or results. Needs: an evaluation script with text prompts ("real"/"fake") on the official test set; no training; about 15 min on CPU for 2,807 pairs |
| DeepShield full (Trust Score + stance + abstention) on Fakeddit | **NOT AVAILABLE** | | | | | | The Trust Score was fitted on LIAR2 only (text + evidence; image weight 0). Needs: image and evidence on Fakeddit, then a refit |

"DeepShield full" on political claims is in 4.3 (LIAR2): Trust Score macro-F1 0.722.
Caveat that must travel with every Fakeddit number: a one-feature **subreddit-only probe reaches macro-F1 1.000 and AUROC 1.000** on the official test set, because each subreddit has one label (`R:c4_generalise/README.md`). Surface-features-only probe: 0.575 / 0.733.
Image gain: +0.039 macro-F1 and +0.033 AUROC over title-only on official Fakeddit.

### 4.2 RQ1: stance (C1)

| Quantity | Value | Source |
|---|---|---|
| Stance model quality on LIAR2 test (n = 1,565, gold evidence): macro-F1, sentence + decisive | **0.687 [0.664, 0.710]** | `R:c1_stance/20261122-193507/metrics.json`, README |
| Same, similarity-only baseline | 0.567 | same |
| Same, whole-passage NLI | 0.615 | same |
| High-similarity debunk recall: sentence NLI vs similarity | 0.63 vs 0 | same |
| Stance as a veracity signal (AUROC of `s_ev`), gold evidence, 500 claims | 0.683 [0.637, 0.729] | `R:c2_trust_live_fc/README.md` |
| Same, live Wikipedia only | 0.639 [0.588, 0.689] | same |
| Same, + Google Fact Check, strict | **0.663 [0.613, 0.708]** | same |
| Detection macro-F1 **without** stance (DistilBERT, calibrated), LIAR2 test n = 2,296 | 0.703 | `R:c2_trust/README.md` (text, recalibrated) |
| Detection macro-F1 **with** stance (Trust Score = text + stance), same data | **0.722** | same (gold evidence) |
| Same without / with stance on the 500 live claims (strict) | 0.684 / 0.695 | `R:c2_trust_live_fc/README.md` |
| Macro-F1 with and without stance **on the refuting-evidence subset** | **NOT AVAILABLE** | No run splits claims by evidence stance. Needs: a few lines in `eval_trust.py` to take claims whose `s_ev` is negative or whose top evidence refutes, then rerun (the cached `signals.csv` is enough; minutes, no GPU) |
| Larger NLI model | no gain (decisive test F1 0.700 vs 0.697 base) | `R:c1_stance/README.md` |
| Known limitation | sentence scoring leans to "refutes": mostly-true claims (label 4) average s_ev about -0.19 | same |

### 4.3 RQ2: calibration (C2), LIAR2 test n = 2,296, seeds 42/43/44

Run `R:c2_trust/20261123-111702` (`metrics.json`, `summary_over_seeds`); gold evidence.

| System | ECE | Brier | Macro-F1 |
|---|---|---|---|
| Raw DistilBERT (before temperature) | 0.077 | 0.192 | not recorded in this table (the run's `text_raw` row is the temperature-scaled model: 0.0157 / 0.1845 / 0.7068) |
| DistilBERT, temperature-scaled (T = 1.70) | 0.016 | 0.185 | 0.707 |
| DistilBERT, recalibrated on the fit half | 0.021 | 0.185 | 0.703 |
| **DeepShield Trust Score (full)** | **0.025** | **0.178** | **0.722** |
| Evidence only, calibrated | 0.024 | 0.223 | 0.589 |

(`R:c2_trust/README.md`; raw-DistilBERT ECE 0.077 / Brier 0.192 are quoted there from the same test set.) "DeepShield without calibration" has **no exact row** (NOT AVAILABLE as defined). The nearest rows in the same run are the unweighted average of the text and evidence signals: ECE 0.041, Brier 0.192, macro-F1 0.678, and the Trust Score without its own temperature (`trust_no_T` in `metrics.json`, values not copied here). Needs: define the ablation, then read it from `metrics.json`.
Bootstrap (seed 42): Trust minus text Brier -0.007 [-0.011, -0.003]; ECE +0.001 [-0.018, +0.017] (no ECE difference).
Reliability data: `reliability_seed42` inside `R:c2_trust/20261123-111702/metrics.json`. **No reliability-diagram image file exists** (`results/` contains no image for it). Needs: a plotting script from those bins (minutes).
On live evidence the gain disappears: Trust Brier 0.185 vs text 0.184 (strict, n = 500) (`R:c2_trust_live_fc/README.md`).

### 4.4 RQ3: abstention (C2), LIAR2 test n = 2,296

| Policy | Error at 80% coverage | Source |
|---|---|---|
| No abstention (100% coverage) | **0.273** (error at 100% coverage, Trust Score); DistilBERT alone 0.292 | `metrics.json` `error_at_100`, summary over seeds |
| Selective by confidence, error at 80% coverage | **0.227** (Trust); 0.249 (DistilBERT) | `R:c2_trust/README.md` |
| Fixed 40/67 thresholds | **NOT AVAILABLE** | The 40/67 rule is in `main.py:211`, but no experiment measures its error at 80% coverage. Needs: a short script on `test_predictions.csv` (minutes) |
| Conformal, alpha = 0.1 | coverage **0.901** (target 0.90), abstains 43.7% (README rounds to 44%), error on answered 0.175 | `summary_over_seeds.trust.conformal` |
| Conformal, alpha = 0.05 / 0.2 | coverage 0.953 / 0.800; abstain 60.5% / 15.8%; error on answered 0.119 / (README: 0.238) | same |

Risk-coverage curve data: `curve` arrays in `metrics.json` (per seed and summary). **No risk-coverage plot image exists.** Needs: a plot script (minutes).
Live evidence, conformal alpha = 0.1 (strict): coverage 0.915, abstains 49%, error on answered 0.166 (`R:c2_trust_live_fc/README.md`). Abstention follows the text model's uncertainty, not the evidence.

### 4.5 RQ4: explanation faithfulness and EEA (C3)

Data: 500 LIAR2 test claims; DistilBERT accuracy on them 0.694 (`R:c3_explain/README.md`).

**Faithfulness (AOPC; the run measures 1, 5, 10, 20 and 50% of words removed, averaged, not at 10/20/30% separately):**

| | SHAP | Random | SHAP - random [95% CI] |
|---|---|---|---|
| Comprehensiveness (higher better) | 0.142 | 0.027 | +0.116 [+0.109, +0.123] |
| Sufficiency (lower better) | 0.012 | 0.146 | -0.136 [-0.143, -0.130] |

Grad-CAM (100 Fakeddit pairs, accuracy 0.91, mirror data): comprehensiveness 0.013 vs random 0.001 (+0.012 [+0.004, +0.021]); sufficiency 0.014 vs 0.031 (-0.017 [-0.028, -0.009]).
Deleting up to half the image changes P(real) by only about 0.01 to 0.03.
**SHAP and Grad-CAM at 10/20/30% separately: NOT AVAILABLE.** `eval_explain.py` reports the average over 1/5/10/20/50%. Needs: set the fractions and rerun (SHAP text run took about 85 min on CPU for 500 claims; Grad-CAM about 2 h for 200 pairs, so ask before running).

**EEA (gold evidence, 322 of 500 claims have supporting or refuting evidence):** mean signed EEA 0.30 (unsigned 0.51); 16% have EEA 0.
**Correct vs wrong verdicts:** gold evidence mean EEA 0.31 vs 0.27; live Wikipedia 0.26 vs 0.14; live + fact checks (strict) 0.28 vs 0.14 (`R:c3_explain/README.md`, `R:live_evidence/README.md`, `R:c2_trust_live_fc/README.md`).

**AUROC for detecting errors (0.5 = chance):**

| Warning score | gold evidence (322 claims, 97 errors) | live Wikipedia (332 claims, 98 errors) |
|---|---|---|
| Low signed EEA | 0.550 [0.481, 0.620] | **0.656 [0.593, 0.715]** (0.668 strict + fact checks) |
| Low unsigned EEA | 0.473 [0.400, 0.541] | not given |
| Low model confidence | 0.766 [0.714, 0.817] | 0.762 [0.711, 0.813] |
| Both combined (5-fold CV) | 0.750 [0.693, 0.804] | 0.756 [0.698, 0.808] |

H4 (EEA warns of errors beyond confidence) is **not supported** in either setting. SHAP faithfulness above is for the text model only.

### 4.6 RQ5: generalisation (C4), macro-F1 (`R:c4_generalise/20261123-200523/metrics.json`, README)

| Setting | Model | n | Misleading share | Macro-F1 | AUROC |
|---|---|---|---|---|---|
| In-dataset | DistilBERT, LIAR2 | 2,296 | 0.58 | 0.703 ± 0.005 | 0.783 ± 0.003 |
| Cross-dataset (near) | DistilBERT, LIAR2 -> `liar_unseen` | 339 | 0.31 | 0.615 ± 0.020 | 0.705 ± 0.009 |
| Cross-dataset (far) | DistilBERT, LIAR2 -> Fakeddit mirror | 1,441 | 0.80 | 0.461 ± 0.005 | 0.464 ± 0.012 |
| In-period (temporal) | DistilBERT before 2018 | 1,046 | 0.49 | 0.633 ± 0.009 | 0.696 ± 0.007 |
| **Later period** | same model on 2019 to 2023 | 7,623 | 0.79 | **0.675 ± 0.004** | 0.798 ± 0.004 |
| **Sri Lankan set** | LIAR2 model | 207 (14 real, 193 misleading) | 0.93 | 0.558 (seed 42 only) | 0.598 [0.410, 0.772] |
| | LIAR2 before 2018 | 207 | 0.93 | 0.510 | 0.571 [0.423, 0.715] |
| | Fakeddit mirror | 207 | 0.93 | 0.438 | 0.543 [0.374, 0.701] |
| | official Fakeddit | 207 | 0.93 | 0.257 | 0.578 [0.432, 0.724] |
| In-dataset | official Fakeddit, DistilBERT | 2,807 | 0.61 | 0.845 ± 0.002 | 0.922 ± 0.001 |
| In-dataset | official Fakeddit, CLIP fusion | 2,807 | 0.61 | 0.884 ± 0.002 | 0.955 ± 0.000 |
| Cross-dataset | official Fakeddit DistilBERT -> LIAR2 | 2,296 | 0.58 | 0.504 ± 0.017 | 0.559 ± 0.004 |

- **SpotFake-style comparison: NOT AVAILABLE** (no baseline, see 4.1). The table above is DistilBERT / DeepShield only.
- Every Sri Lankan AUROC interval includes 0.5, so the set cannot show transfer (only 14 real claims).
- The "later period" F1 rises because the share of misleading claims rises from about 45% to 79%; it is a data shift, not a better model (README, point 3).
- Shortcut probes on LIAR2 test: metadata only F1 0.694 / AUROC 0.772; year only 0.661 / 0.707; surface features 0.572 / 0.663; the text model beats metadata by about 0.01.
- Sinhala (`R:sinhala/README.md`, 5-fold CV x 3 seeds, binary n = 1,113): char TF-IDF AUROC 0.748 ± 0.008, PR-AUC 0.293, macro-F1 0.628; outlet-only probe AUROC 0.591; fine-tuned XLM-R (one seed) AUROC 0.686. In the app, `SINHALA_MODEL_PATH` routes Sinhala text to the char TF-IDF model.

### 4.7 Significance tests run

| Test | Where | Result |
|---|---|---|
| McNemar, sentence NLI vs similarity and vs passage NLI | C1, `R:c1_stance/README.md` | both p < 1e-6 |
| Paired bootstrap, 1,000 resamples, seed 42 | C2, C3, C4 (`metrics.json` `bootstrap_seed42`, README) | Trust minus text Brier -0.007 [-0.011, -0.003]; Trust minus evidence Brier -0.046 [-0.052, -0.039]; ECE differences include 0 |
| Bootstrap CIs on AUROC / macro-F1 / drops | C1, C3, C4, judge | listed with each number above |
| Paired bootstrap on live evidence | `R:c2_trust_live/README.md` | Trust minus text Brier +0.003 [-0.002, 0.008] (includes 0) |

No t-test, Wilcoxon or p-values other than McNemar are recorded. Seed sd is over three seeds; for the LIAR2 Trust runs it reflects only the val fit/calibration split (the weights seed is 42).

### 4.8 Analysis time

- **NOT AVAILABLE** for average analysis time (text only, text + image) and evidence retrieval alone. There is no timing log and no timing code in the API. Needs: a short script that calls `run_models` and `gather_evidence` N times and records seconds (minutes of CPU).
- Related recorded figures: warm-up of all models at start-up (`main.py:59`, lifespan `main.py:83`); the web UI gives up after 100 s (`main.py:59` docstring); judge limit `JUDGE_DEADLINE = 30` s (`assessment.py`); `DEMO.md:11` says the first analysis after a restart can take 1 to 2 minutes and later ones 5 to 40 seconds.
- Experiment runtimes: C1 stance main run 5,460.5 s, about 91 min, val + test (`R:c1_stance/20261122-193507/config.json`); 500-claim base 1,112 s, large 4,619 s; SHAP text audit about 85 min for 500 claims; live evidence run about 2 h for 500 claims.

### 4.9 Extra results for the app's added layers (beyond C1 to C4)

| Layer | Result | Source |
|---|---|---|
| Out-of-domain check (k = 25, MiniLM) | AUROC "far from LIAR2": Fakeddit titles 0.961, Sri Lankan claims 0.955; 5% of LIAR2 val sent to review by construction; threshold 0.385 | `R:domain/README.md` |
| Same-claim fact-check rating | agrees with the LIAR2 label on 19 of 19 same-claim ratings from non-PolitiFact outlets (about 4% of claims) | `R:domain/README.md` |
| Fair fact-check evidence coverage | 41 of 500 claims (8%) | `R:c2_trust_live_fc/README.md` |
| Gemini judge on LIAR2 test (500 claims, `knowledge` prompt) | commits on 77 (15%); "refuted" right 56 of 64; "supported" right 9 of 13; accuracy 0.696 to 0.700; AUROC 0.780 to 0.782 | `R:judge/README.md` |
| Whole app, plain facts, untouched test half | 60 facts: 58 right, 0 wrong, 2 Needs review (`R:facts_app/20261125-042332`); 20 invented: 20 "not real" (`...-042709`); 15 true Sri Lankan: 15 "real" (`...-042932`) | `R:facts_app/*/metrics.json` |
| Whole app, blind set written after tuning (72 claims, 5 to 13 words) | 66 right, 0 wrong, 6 Needs review (3 had no judge answer, 3 judge "not enough evidence") | `R:facts_app/20261125-065216/metrics.json` |
| Fact sets are small and written by the team | not real-news accuracy | `R:judge/README.md` limits |

---

## 5. Testing

- Folder: `backend/tests/` (23 test files + `conftest.py`, which pins `MODEL_MODE=simulation` and clears model and judge settings so tests ignore `.env`).
- **Run now (2026-11-25, `python -m pytest -q`): 230 passed, 0 failed, 1 warning, 61.75 s.**
- Per file (collected): test_stance 39, test_data 33, test_factcheck_verdict 27, test_explain_audit 18, test_check_export 13, test_judge 10, test_trust 10, test_eval_explain 9, test_eval_generalise 8, test_eval_trust 8, test_claim_and_embeddings 7, test_collect_srilanka 6, test_eval_live 6, test_eval_stance 6, test_sinhala_routing 6, test_assessment 5, test_eval_sinhala 4, test_analysis_contract 3, test_core 3, test_training 3, test_fetch_fakeddit 2, test_merge_factchecks 2, test_model_adapter_gradcam 2 (sum 230).
- What is **not** covered by tests: the API against a live MongoDB (`test_analysis_contract` has 3 tests, no auth/CRUD suite), the front end (no front-end tests exist), PDF and CSV export, the live Gemini and Wikipedia calls.

---

## 6. Gaps: everything marked NOT AVAILABLE, and what it would take

| # | Missing item | What it would take | Cost / GPU |
|---|---|---|---|
| 1 | SpotFake-style baseline (4.1, 4.6) | Write the model (~100 lines), train 3 seeds on official Fakeddit | about 1 to 2 h CPU; no GPU needed. **Ask before running** |
| 2 | Zero-shot CLIP baseline (4.1) | Evaluation script with text prompts on 2,807 official test pairs | about 15 min CPU |
| 3 | NewsCLIPpings (3) | Request VisualNews access, add a loader, train | days (approval); GPU preferable |
| 4 | DeepShield full on Fakeddit (4.1) | Build the evidence + Trust inputs for Fakeddit and refit | hours; needs evidence retrieval for Reddit titles |
| 5 | RQ1 on the refuting-evidence subset (4.2) | Subset analysis from the cached `signals.csv` | minutes, no GPU |
| 6 | Reliability-diagram image (4.3) | Plot from `reliability_seed42` bins | minutes |
| 7 | Risk-coverage plot, fixed 40/67 thresholds row (4.4) | Plot from `curve`; short script for the thresholds | minutes |
| 8 | SHAP / Grad-CAM at 10/20/30% (4.5) | Change fractions in `eval_explain.py`, rerun | SHAP about 85 min, Grad-CAM about 2 h CPU. **Ask before running** |
| 9 | Analysis-time measurements (4.8) | Timing script around `run_models` and `gather_evidence` | minutes |
| 10 | Fusion-head training time (2.3) | Not logged; rerun with timing, or state "trained on cached CLIP features, minutes" only if measured | minutes to rerun |
| 11 | Project plan / proposal file with RQ wording and the original BERT-base plan | Not in the repository; supply it | none |
| 12 | Git history for the development story | The project is not a git repository (`git rev-parse`: not a repository). See `docs/DEV_HISTORY.md` | none |
| 13 | Docker image cannot run real models (1.7) | Add `requirements-ml.txt` to the Dockerfile, copy `artifacts/`, `research/` as needed | small, outside this task |
| 14 | "Simulated" wording left in the PDF, CSV and integrations (1.8) | Edit text in `main.py` | small, outside this task |
| 15 | Live-news accuracy, Sri Lankan transfer, and real-world fact accuracy | More verified-true Sri Lankan claims; a larger independent claim set | days |
| 16 | Tavily key not set | Add a key to `.env` to enable news search and URL-only extraction | none |

Security items for you (not code): narrow the Atlas IP list (currently open to all), rotate the database password and the Gemini keys that appeared in chat, delete `backend/.env.atlas.bak` and check `backend/atlas-credentials.env`.
