# Model export spec (Colab → DeepShield)

This spec is for whoever trains the BERT text model or the CLIP fusion model in Colab. If an export
follows it, the app can load the model and the research experiments (C2 Trust Score, C3 explanation
audit, C4 evaluation) can run with no code changes. Before handing an export over, run the checker:

```bash
cd backend
python -m research.check_export path/to/export/text-bert-liar2
```

It prints every problem it finds and exits with code 1 if any of them is an error.

## 1. Folder layout

```
<export>/                         e.g. text-bert-liar2/ or mm-clip-fakeddit/
  manifest.json
  model/                          becomes TEXT_MODEL_PATH or MULTIMODAL_MODEL_PATH
  predictions/
    seed42/<dataset>_<split>.csv
    seed43/<dataset>_<split>.csv
    seed44/<dataset>_<split>.csv
```

- Train with **three seeds** (42, 43, 44); the research plan reports mean ± sd. Export predictions for
  all three, but model weights only for seed 42 (`weights_seed`).
- Never commit `model/` or any weights to git. Share the folder through Drive.

## 2. `model/`: unchanged from the existing app contract

The contract is defined in the docstring of `backend/app/services/model_adapter.py`. In short:

| kind | Required files |
|---|---|
| `text` | `save_pretrained()` model and tokenizer; `config.json` `id2label` must name the real class `real`/`true`/`authentic`/`reliable`/`legit`/`label_1`; optional `calibration.json` = `{"temperature": T}` |
| `multimodal` | `fusion_config.json` (`clip_model`, `hidden_dim`, `dropout`, `real_index`, `temperature`), `fusion_head.pt` (state_dict of `FusionHead`, same architecture), optional fine-tuned CLIP subfolder |

## 3. `manifest.json`

```json
{
  "kind": "text",
  "base_model": "bert-base-uncased",
  "trained_on": "liar2",
  "seeds": [42, 43, 44],
  "weights_seed": 42,
  "temperature_by_seed": {"42": 1.31, "43": 1.27, "44": 1.35},
  "data_loader_sha256": "<sha256 of the research/data.py you used>",
  "hyperparameters": {"lr": 2e-5, "epochs": 3, "batch_size": 16, "max_length": 256},
  "created": "2026-12-01",
  "notes": "free text"
}
```

`kind` is `text` or `multimodal`. `trained_on` is one of the dataset names in section 4.
Each temperature is fitted on that seed's **val** split only (`model_adapter.fit_temperature`).

## 4. Predictions: `<dataset>_<split>.csv`

Use **`backend/research/data.py`** in Colab (copy the file) to load every dataset. This guarantees
the same ids, splits and label mapping as the experiments. Record its hash in the manifest.

| dataset | Loader | Splits to export |
|---|---|---|
| `liar2` | `load_liar2(split)` (half-true counts as real) | `val`, `test` |
| `liar_unseen` | `load_liar_unseen(split)`: LIAR claims that are not in LIAR2 | `val`, `test` |
| `liar` | `load_liar_binary(split)` (75% overlaps LIAR2; train on it only if it is your source) | `val`, `test` |
| `fakeddit` | `load_fakeddit(split)` | `val`, `test` |
| `fakeddit_official` | `load_fakeddit_official(split)`: official v2.0 sample from `research/fetch_fakeddit.py` | `val`, `test` |
| `liar2_temporal` | `load_liar2_temporal(split)`: before 2018 = train/val/test, 2019+ = future | `val`, `test`, `future` |
| `sinhala` | `load_sinhala(split=...)` (seeded split) | `val`, `test` (only for a multilingual model) |
| `srilanka` | `load_local_labelled()`: Sri Lankan FactSeeker set (research/collect_srilanka.py), test only | `test` |

**Required** (`check_export.required_files`): a `text` export must include `liar2_val` and `liar2_test`;
a `multimodal` export must include `fakeddit_val` and `fakeddit_test`. A model trained on
`liar2_temporal` or `fakeddit_official` must also include its own source's val/test files (for multimodal
`fakeddit_official`, those replace the mirror's files). Never export a `train` split.
Targets created after training (e.g. `srilanka_test`) can be added for the saved weights seed with
`python -m research.predict_export <export>`.

**For C4 (`research/eval_generalise.py`), train and export these models:**

| `trained_on` | Also export predictions for |
|---|---|
| `liar2` (the main text model) | `liar_unseen_test`, `fakeddit_test`, `srilanka_test` if it exists |
| `liar2_temporal` (same recipe, temporal train split) | `liar2_temporal_val`, `liar2_temporal_test`, `liar2_temporal_future` |
| `fakeddit` (text-only BERT, same recipe) | `liar2_test` |
| `fakeddit_official` (text-only, same recipe; paper numbers) | `liar2_test` |

Then run `python -m research.eval_generalise --export <dir1> <dir2> <dir3>`.

Columns (one row per example whose `binary_label` is not `None`, every such example included):

| column | Meaning |
|---|---|
| `id` | `Example.id` from the loader, unchanged |
| `label` | `Example.binary_label`: 1 = misleading, 0 = real |
| `misleading_probability` | P(misleading) **after** temperature scaling |
| `margin` | logit(real) − logit(misleading) **before** temperature scaling |

The checker verifies that `1 - misleading_probability ≈ sigmoid(margin / T)` for that seed's T.
The file also works with the existing `backend/evaluate.py`.

```python
import csv, torch
rows = [e for e in load_liar2('test') if e.binary_label is not None]
with torch.no_grad():
    logits = model(**tokenizer([e.text for e in rows], truncation=True, padding=True,
                               return_tensors='pt')).logits          # batch this in practice
margin = logits[:, real_index] - logits[:, 1 - real_index]
p_misleading = 1 - torch.sigmoid(margin / T)
with open('predictions/seed42/liar2_test.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label', 'misleading_probability', 'margin'])
    for e, p, m in zip(rows, p_misleading.tolist(), margin.tolist()):
        w.writerow([e.id, e.binary_label, round(p, 6), round(m, 6)])
```

## 5. Rules

- Train only on `train`; fit the temperature and choose hyperparameters on `val`; never look at `test`.
- Fix all seeds (`random`, `numpy`, `torch`, `transformers.set_seed`).
- **Fakeddit:** the mirror in `datasets/fakeddit_9k` is unofficial, so use it for building and testing only.
  The mirror's splits share images and texts. `load_fakeddit` now drops them by default (train
  4800 → 4701 rows, val 2405 → 2386, test unchanged), so use the loader and don't re-read the CSVs.
  Rows with `meta['image_placeholder']` set use imgur's "image no longer available" picture; treat
  them as having no image. Some images are publisher logos that always come with one label (e.g. The
  Onion's logo → satire). A model can learn the logo instead of the content; C4 tests for this.
- **Official Fakeddit** (`fakeddit_official`): a stratified sample (seed 42) of the official v2.0
  multimodal splits; only rows whose image still downloads are used (about 93%). Fakeddit labels were
  assigned per subreddit (every subreddit maps to one label), so C4 reports a subreddit-only probe.
- Real results must come from these trained models, never from `MODEL_MODE=simulation`/`demo`.
