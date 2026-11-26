# C4 — Shortcut-aware evaluation protocol: results

Produced by `backend/research/eval_generalise.py`. For each model it reports three kinds of
number (in-dataset, cross-dataset and future), plus shortcut probes trained on the same data.

| Source model trained on | In-dataset test | Cross-dataset / future tests |
|---|---|---|
| `liar2` | LIAR2 test | `liar_unseen` (LIAR claims absent from LIAR2: about 75% of LIAR is inside LIAR2), Fakeddit titles |
| `liar2_temporal` (claims before 2018) | same years, held out | 2019–2023 (`future`); 2018 left out as a gap year; repeated claims removed |
| `fakeddit` (text only) | Fakeddit test | LIAR2 test |
| any | | Sri Lankan set (`datasets/srilanka_local/labelled.csv`: human-labelled: FactSeeker verdicts, researcher-approved claims; format: `research/data.py`, `load_local_labelled`) |

## Run `20261123-200523/`: trained models (main result)

Models: DistilBERT (`distilbert-base-uncased`) fine-tuned on CPU by `research/train_text.py`
(2 epochs, lr 3e-5, max 128 tokens), one export per source in `backend/artifacts/`. All three pass
`check_export`. Values are mean ± sd over seeds 42/43/44; the F1-drop CI is the mean of per-seed
bootstrap CIs. DistilBERT replaces the plan's BERT-base so training fits on a CPU.
Fakeddit rows are from an unofficial mirror: use them for building and testing, not paper numbers.

| Model → target | Kind | n | Misleading share | macro-F1 | AUROC | ECE | F1 drop [95% CI] |
|---|---|---|---|---|---|---|---|
| liar2 → liar2 test | in | 2296 | 0.58 | 0.703 ± 0.005 | 0.783 ± 0.003 | 0.019 | |
| liar2 → liar_unseen | cross | 339 | 0.31 | 0.615 ± 0.020 | 0.705 ± 0.009 | 0.093 | +0.088 [+0.031, +0.140] |
| liar2 → fakeddit | cross | 1441 | 0.80 | 0.461 ± 0.005 | 0.464 ± 0.012 | 0.109 | +0.241 [+0.217, +0.265] |
| temporal → past test | in | 1046 | 0.49 | 0.633 ± 0.009 | 0.696 ± 0.007 | 0.039 | |
| temporal → future | future | 7623 | 0.79 | 0.675 ± 0.004 | 0.798 ± 0.004 | 0.045 | −0.042 [−0.073, −0.009] |
| fakeddit → fakeddit test | in | 1441 | 0.80 | 0.750 ± 0.013 | 0.861 ± 0.001 | 0.033 | |
| fakeddit → liar2 test | cross | 2296 | 0.58 | 0.542 ± 0.024 | 0.563 ± 0.025 | 0.133 | +0.207 [+0.171, +0.243] |
| CLIP fusion (title + image) → fakeddit test | in | 1441 | 0.80 | 0.784 ± 0.002 | 0.910 ± 0.001 | 0.016 | |

Fusion (`research/train_fusion.py`) keeps CLIP ViT-B/32 frozen and trains only the app's FusionHead.
The image adds +0.034 F1 and +0.049 AUROC over the title-only DistilBERT on the same test set. It has no
cross-dataset row because LIAR2 has no images. Part of the gain may come from the publisher-logo
image shortcut described in `research/MODEL_EXPORT.md`, which is not separated out here.

The LIAR2 and mirror-Fakeddit shortcut probes don't depend on the model, so they are the same as in the placeholder table below.

### Official Fakeddit (paper numbers for the multimodal model)

These models are trained on a stratified sample of the **official** Fakeddit v2.0 multimodal splits
(`research/fetch_fakeddit.py`; 8,841 / 1,777 / 2,807 posts whose image still downloads, about 93%).
They replace the mirror models above for reporting.

| Model → target | Kind | n | Misleading share | macro-F1 | AUROC | ECE | F1 drop [95% CI] |
|---|---|---|---|---|---|---|---|
| DistilBERT (title) → official test | in | 2807 | 0.61 | 0.845 ± 0.002 | 0.922 ± 0.001 | 0.025 | |
| CLIP fusion (title + image) → official test | in | 2807 | 0.61 | **0.884 ± 0.002** | **0.955 ± 0.000** | 0.010 | |
| DistilBERT (title) → LIAR2 test | cross | 2296 | 0.58 | 0.504 ± 0.017 | 0.559 ± 0.004 | 0.205 | +0.341 [+0.316, +0.366] |

| Shortcut probe (official test) | macro-F1 | AUROC |
|---|---|---|
| surface features of the title | 0.575 | 0.733 |
| **subreddit only** | **1.000** | **1.000** |

5. **Fakeddit's labels are its subreddits.** In the official data every subreddit maps to exactly one
   label (e.g. psbattle_artwork → false connection, mildlyinteresting → true), because the dataset
   was labelled by source. A one-feature probe on the subreddit is perfect, so the models' 0.92–0.96 AUROC
   is largely *recognising which community a post comes from*, not *checking whether it is true*.
   Fakeddit in-dataset scores must be reported with this caveat. They are useful for comparing text vs
   text + image under the same conditions (the image adds +0.039 F1 and +0.033 AUROC), not as evidence of
   real-world misinformation detection.
6. **Nothing carries over to political claims.** The official-Fakeddit text model is near chance on
   LIAR2 (AUROC 0.56) and badly miscalibrated there (ECE 0.21), consistent with point 5.

### Sri Lankan test set (`srilanka/test`)

**What the set is.**
- **Source:** 212 claims from FactSeeker (Sri Lanka Press Institute) fact checks, collected by `research/collect_srilanka.py`.
- **Human-labelled.** Labels are the verdicts of FactSeeker's human fact-checkers. The claim texts were
  extracted with AI assistance and then **reviewed and approved by the researcher** (2026-11-23).
- **Claim texts:** the text is the circulating claim, not the headline. Rows were dropped where the claim was a fragment,
  contained the verdict, or the verdict tag contradicted the article. 136 kept claims were lightly edited
  (e.g. a speaker added from the title). Every decision is recorded in `datasets/srilanka_local/review.csv`.
- **Size used here:** 207 rows score in binary terms (14 real, 193 misleading; 5 uncertain are excluded), 211 of them in English.
- **Predictions:** only the weights seed (42) has them (`research/predict_export.py`), so these rows have no ± over seeds.

| Model trained on | macro-F1 | AUROC [95% bootstrap CI] | mean P(real) |
|---|---|---|---|
| LIAR2 | 0.558 | 0.598 [0.410, 0.772] | 0.17 |
| LIAR2 before 2018 | 0.510 | 0.571 [0.423, 0.715] | 0.32 |
| Fakeddit mirror (titles) | 0.438 | 0.543 [0.374, 0.701] | 0.45 |
| official Fakeddit (titles) | 0.257 | 0.578 [0.432, 0.724] | 0.65 |

7. **The Sri Lankan set cannot yet show whether models transfer to Sri Lankan claims.** Every CI includes
   chance, because only 14 claims are real: fact-checkers mostly publish debunks. The LIAR2 model is the
   best calibrated to this misleading-heavy set (it rarely says "real"). The official-Fakeddit model calls most claims
   real (mean P(real) 0.65), which gives it the lowest F1. A conclusive test needs more verified-true
   Sri Lankan claims (for example from official announcements).

### What the trained models show

1. **On LIAR2, metadata is barely worse than reading the claim.** DistilBERT reaches F1 0.703 /
   AUROC 0.783. Speaker/context metadata alone reaches 0.694 / 0.772, and the year alone reaches
   0.661 / 0.707. The text model beats the metadata probe by only about 0.01 in both metrics. On the
   temporal split the gap is about as small (0.633 vs 0.628). LIAR2 in-dataset scores therefore say little about
   whether a model reads the claim, which is the case for C4.
2. **Cross-domain transfer is at chance in one direction and close to it in the other.** LIAR2 → Fakeddit:
   AUROC 0.46. Fakeddit → LIAR2: 0.56. Near-domain `liar_unseen` keeps a usable ranking (AUROC 0.705)
   but loses 0.09 F1 and becomes poorly calibrated (ECE 0.019 → 0.093). Part of that comes from the
   different label share (31% vs 58% misleading).
3. **Time drift does not hurt ranking here.** The model trained on claims before 2018 ranks 2019–2023
   claims *better* than its own held-out years (AUROC 0.798 vs 0.696), and F1 rises a little.
   The future set is 79% misleading and is a different mix of claims, so this is a change
   in the data, not evidence that the model got better. The placeholder showed the same pattern, which points to
   the data rather than the model. Report AUROC and the label share next to every out-of-dataset F1.
4. Fine-tuning helps Fakeddit text much more than TF-IDF does (F1 0.525 → 0.750), but the model learns
   nothing that carries over to political claims.

## Run `20261122-232153/`: TF-IDF placeholder (baseline)

The TF-IDF + logistic regression placeholder that set up the protocol before the models were
trained. It has one seed because it is deterministic.

| Model → target | Kind | n | Misleading share | macro-F1 | AUROC | ECE | F1 drop [95% CI] |
|---|---|---|---|---|---|---|---|
| liar2 → liar2 test | in | 2296 | 0.58 | 0.683 | 0.774 | 0.041 | |
| liar2 → liar_unseen | cross | 339 | 0.31 | 0.577 | 0.733 | 0.060 | +0.107 [+0.053, +0.166] |
| liar2 → fakeddit | cross | 1441 | 0.80 | 0.461 | 0.488 | 0.081 | +0.222 [+0.195, +0.249] |
| temporal → past test | in | 1046 | 0.49 | 0.641 | 0.688 | 0.033 | |
| temporal → future | future | 7623 | 0.79 | 0.604 | 0.731 | 0.047 | +0.036 [+0.004, +0.066] |
| fakeddit → fakeddit test | in | 1441 | 0.80 | 0.525 | 0.754 | 0.032 | |
| fakeddit → liar2 test | cross | 2296 | 0.58 | 0.368 | 0.461 | 0.240 | +0.156 [+0.127, +0.185] |

**Shortcut probes** (each sees only one kind of input; scored on the source's in-dataset test):

| Source | Surface (length, quotes, digits…) | Metadata (speaker, context, subject, state) | Year only |
|---|---|---|---|
| liar2 | F1 0.572 / AUROC 0.663 | **F1 0.694 / AUROC 0.772** | F1 0.661 / AUROC 0.707 |
| liar2_temporal | 0.563 / 0.618 | 0.628 / 0.678 | n/a |
| fakeddit | 0.446 / 0.529 | n/a | n/a |

## What the placeholder showed (confirmed by the trained models above)

1. **LIAR2 has strong shortcuts.** Speaker/context metadata *alone* matches the text model (F1 0.694
   vs 0.683), and the claim year alone reaches 0.661, because the misleading share rises from about 45% (2010–16)
   to 87% (2023). Any LIAR2 in-dataset number must be reported next to these probes. A model that
   does not beat them clearly has not shown it reads the claim.
2. **Cross-domain transfer fails completely.** Moving between political claims and Reddit titles drops AUROC to chance
   (0.49 and 0.46). Near-domain transfer (`liar_unseen`) keeps most of the ranking (AUROC 0.73).
   Much of its F1 drop comes from the different label share (31% vs 58% misleading), which a fixed
   0.5 threshold does not adjust for.
3. **Time drift mostly shifts the labels, not the ranking.** On 2019–2023 claims, AUROC holds (0.73 vs 0.69), but F1
   drops slightly (−0.036) because 79% of future claims are misleading. Report AUROC and the label
   share next to F1 for every out-of-dataset test.
4. Fakeddit (Reddit titles, 6-way labels collapsed to 2) is hard for a text-only model (F1 0.525).
   Its image side also has a known shortcut: some publisher logos always appear with one label
   (see `research/MODEL_EXPORT.md`).

Data hygiene behind these numbers: the Fakeddit splits share no text or image (`load_fakeddit`
drops 99 train and 19 val rows). `liar_unseen` and `liar2_temporal` remove repeated claims using
`claim_key()`, which ignores UKP's "<Speaker> says" prefix.
