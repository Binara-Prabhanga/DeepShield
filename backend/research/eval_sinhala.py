"""Sinhala misinformation detection on the LIRNEasia corpus (Corpus.xlsx, CC BY 4.0).

Tasks
  binary   CREDIBLE (real) vs FALSE + PARTIAL (misleading); UNCERTAIN rows left out.
           1,003 real vs 110 misleading, so a single 15% test split would hold ~17 misleading rows.
  3class   CREDIBLE / UNCERTAIN / MISLEADING (= FALSE + PARTIAL), like the LIRNEasia paper.

Protocol (fixed seeds 42, 43, 44)
  random    stratified 5-fold cross-validation, repeated per seed; metrics on the pooled
            out-of-fold predictions, then mean and sd over seeds.
  grouped   stratified 5-fold with whole outlets (domains) held out, so test outlets are never
            seen in training. A drop from random to grouped means the model leans on the outlet.

Models
  domain      one-hot outlet only: a shortcut probe (it sees no text)
  char_tfidf  character 2-5-gram TF-IDF + logistic regression (class-balanced)
  word_tfidf  word 1-2-gram TF-IDF + logistic regression (class-balanced)
  labse       frozen LaBSE sentence embeddings (covers Sinhala) + logistic regression
  xlmr        xlm-roberta-base fine-tuned per fold (--xlmr; slow on CPU, so one seed and, by
              default, only the binary task under the random protocol)

Metrics: binary -> AUROC, average precision (PR-AUC; base rate ~0.10), macro-F1, misleading
recall/precision at 0.5; 3class -> macro-F1, accuracy. Bootstrap 95% CI for AUROC / macro-F1 (seed 42).

Run:   cd backend && python -m research.eval_sinhala [--xlmr]
Writes results/sinhala/<timestamp>/{config.json, metrics.json, oof_seed42.csv}.

App model: python -m research.eval_sinhala --export artifacts/text-sinhala-char-tfidf
  fits char_tfidf on every binary row (seed 42) inside a sigmoid CalibratedClassifierCV, so the app's
  P(real) is calibrated to the corpus base rate, and writes model.joblib + config.json. Point
  SINHALA_MODEL_PATH at that folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from research import data

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'sinhala'
CACHE_DIR = Path(__file__).resolve().parents[1] / 'artifacts' / 'cache'
SEEDS = (42, 43, 44)
FOLDS = 5
LABSE = 'sentence-transformers/LaBSE'
XLMR = 'xlm-roberta-base'
CLASSES_3 = ('CREDIBLE', 'UNCERTAIN', 'MISLEADING')
MODELS = ('domain', 'char_tfidf', 'word_tfidf', 'labse')


def task_rows(rows: list[data.Example], task: str) -> tuple[list[data.Example], np.ndarray]:
    """(examples, y). binary: y = 1 misleading; 3class: index into CLASSES_3."""
    if task == 'binary':
        kept = [e for e in rows if e.binary_label is not None]
        return kept, np.array([e.binary_label for e in kept])
    names = ['MISLEADING' if e.label_name in ('FALSE', 'PARTIAL') else e.label_name for e in rows]
    return rows, np.array([CLASSES_3.index(n) for n in names])


def folds(y: np.ndarray, groups: list[str], seed: int, grouped: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def make_classifier(model: str, seed: int):
    """sklearn pipeline taking a list of feature inputs (texts, domains or embedding rows)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    lr = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=seed)
    if model == 'domain':
        return make_pipeline(OneHotEncoder(handle_unknown='ignore'), lr)
    if model == 'char_tfidf':
        return make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                             max_features=200_000), lr)
    if model == 'word_tfidf':
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr)
    if model == 'labse':
        return lr
    raise ValueError(f'Unknown model {model!r}')


def model_inputs(model: str, rows: list[data.Example], embeddings: np.ndarray | None):
    if model == 'domain':
        return np.array([[e.meta.get('domain', '')] for e in rows], dtype=object)
    if model == 'labse':
        return embeddings
    return [e.text for e in rows]


def oof_probs(model: str, x, y: np.ndarray, splits, seed: int) -> np.ndarray:
    """Out-of-fold class probabilities, shape (N, n_classes)."""
    out = np.zeros((len(y), len(np.unique(y))))
    for train, test in splits:
        clf = make_classifier(model, seed)
        pick = (lambda idx: x[idx]) if isinstance(x, np.ndarray) else (lambda idx: [x[i] for i in idx])
        clf.fit(pick(train), y[train])
        out[test] = clf.predict_proba(pick(test))
    return out


def scores(task: str, y: np.ndarray, probs: np.ndarray) -> dict:
    if task == 'binary':
        p, pred = probs[:, 1], (probs[:, 1] >= 0.5).astype(int)
        return {'auroc': float(roc_auc_score(y, p)), 'pr_auc': float(average_precision_score(y, p)),
                'macro_f1': float(f1_score(y, pred, average='macro')),
                'misleading_recall': float(recall_score(y, pred, zero_division=0)),
                'misleading_precision': float(precision_score(y, pred, zero_division=0))}
    pred = probs.argmax(1)
    return {'macro_f1': float(f1_score(y, pred, average='macro')), 'accuracy': float(accuracy_score(y, pred)),
            'f1_by_class': {c: float(v) for c, v in zip(CLASSES_3, f1_score(y, pred, average=None, labels=[0, 1, 2]))}}


def bootstrap_ci(task: str, y: np.ndarray, probs: np.ndarray, n: int = 1000, seed: int = SEEDS[0]) -> dict:
    rng = np.random.default_rng(seed)
    key = 'auroc' if task == 'binary' else 'macro_f1'
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == len(np.unique(y)):
            values.append(scores(task, y[idx], probs[idx])[key])
    low, high = np.percentile(values, [2.5, 97.5])
    return {'metric': key, 'low': float(low), 'high': float(high)}


def mean_sd(runs: list[dict]) -> dict:
    out = {}
    for key, value in runs[0].items():
        if isinstance(value, dict):
            out[key] = mean_sd([r[key] for r in runs])
        else:
            vals = [r[key] for r in runs]
            out[key] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


def labse_embeddings(rows: list[data.Example], batch_size: int = 16, max_length: int = 256) -> np.ndarray:
    """L2-normalised LaBSE embeddings for every corpus row, cached by row ids."""
    path = CACHE_DIR / 'labse_sinhala.npz'
    ids = np.array([e.id for e in rows])
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if np.array_equal(cached['ids'], ids):
            return cached['emb']
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer, model = AutoTokenizer.from_pretrained(LABSE), AutoModel.from_pretrained(LABSE).eval()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].text))
    emb = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([rows[i].text for i in idx], truncation=True, max_length=max_length, padding=True,
                            return_tensors='pt')
            out = model(**enc).pooler_output
            emb[idx] = (out / out.norm(dim=-1, keepdim=True)).numpy()
            if (start // batch_size) % 25 == 0:
                print(f'  LaBSE {start + len(idx)}/{len(rows)}', flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, emb=emb)
    return emb


def xlmr_oof(texts: list[str], y: np.ndarray, splits, seed: int, epochs: int = 3, lr: float = 2e-5,
             batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    """Fine-tune xlm-roberta-base on each training fold (class-weighted loss); out-of-fold probabilities.
    The 250k-token embedding matrix is frozen so CPU training stays feasible."""
    import random
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from research.train_text import predict_logits
    n_classes = len(np.unique(y))
    out = np.zeros((len(y), n_classes))
    tokenizer = AutoTokenizer.from_pretrained(XLMR)
    for k, (train, test) in enumerate(splits):
        random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(XLMR, num_labels=n_classes)
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        counts = np.bincount(y[train], minlength=n_classes)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (n_classes * counts), dtype=torch.float))
        steps = int(np.ceil(len(train) / batch_size)) * epochs
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        rng = random.Random(seed)
        model.train()
        for epoch in range(epochs):
            order = list(train)
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                                return_tensors='pt')
                loss = loss_fn(model(**enc).logits, torch.tensor(y[idx]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
            print(f'  xlmr fold {k + 1}/{len(splits)} epoch {epoch + 1}/{epochs} done', flush=True)
        if n_classes == 2:
            logits = predict_logits(model, tokenizer, [texts[i] for i in test], max_length, batch_size=32)
        else:
            model.eval()
            with torch.no_grad():
                logits = np.concatenate([model(**tokenizer([texts[i] for i in test[s:s + 32]], truncation=True,
                                                            max_length=max_length, padding=True,
                                                            return_tensors='pt')).logits.numpy()
                                         for s in range(0, len(test), 32)])
        e = np.exp(logits - logits.max(1, keepdims=True))
        out[test] = e / e.sum(1, keepdims=True)
    return out


def run(rows: list[data.Example], embeddings: np.ndarray | None, models=MODELS, seeds=SEEDS,
        n_bootstrap: int = 1000, xlmr: tuple[tuple[str, str], ...] = ()) -> tuple[dict, dict]:
    """Returns (metrics, {task: {protocol: {model: seed-42 out-of-fold probs}}}).
    xlmr: (task, protocol) pairs to also fine-tune XLM-R on, e.g. (('binary', 'random'),)."""
    index = {e.id: i for i, e in enumerate(rows)}
    metrics, oof = {}, {}
    for task in ('binary', '3class'):
        task_examples, y = task_rows(rows, task)
        emb = embeddings[[index[e.id] for e in task_examples]] if embeddings is not None else None
        groups = [e.meta.get('domain', '') for e in task_examples]
        metrics[task] = {'n': len(y), 'class_counts': np.bincount(y).tolist()}
        oof[task] = {}
        for protocol in ('random', 'grouped'):
            metrics[task][protocol], oof[task][protocol] = {}, {}
            for model in models:
                if model == 'labse' and emb is None:
                    continue
                if model == 'domain' and protocol == 'grouped':
                    continue                                    # unseen outlets: the probe has nothing to use
                x = model_inputs(model, task_examples, emb)
                runs = []
                for seed in seeds:
                    probs = oof_probs(model, x, y, folds(y, groups, seed, protocol == 'grouped'), seed)
                    runs.append(scores(task, y, probs))
                    if seed == seeds[0]:
                        oof[task][protocol][model] = probs
                        ci = bootstrap_ci(task, y, probs, n_bootstrap)
                metrics[task][protocol][model] = {**mean_sd(runs), 'ci_seed42': ci}
                print(f'{task:7s} {protocol:8s} {model:11s} ' + ', '.join(
                    f'{k} {v["mean"]:.3f}' for k, v in metrics[task][protocol][model].items()
                    if isinstance(v, dict) and 'mean' in v), flush=True)
            if (task, protocol) in xlmr:
                seed = seeds[0]
                probs = xlmr_oof([e.text for e in task_examples], y, folds(y, groups, seed, protocol == 'grouped'), seed)
                oof[task][protocol]['xlmr'] = probs
                metrics[task][protocol]['xlmr'] = {**mean_sd([scores(task, y, probs)]),
                                                   'ci_seed42': bootstrap_ci(task, y, probs, n_bootstrap),
                                                   'seeds': [seed]}
    return metrics, oof


def export_model(rows: list[data.Example], out_dir: Path, seed: int = SEEDS[0]) -> Path:
    """Fit the app's Sinhala model (calibrated char_tfidf, binary task) on all rows and save it."""
    import joblib
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    examples, y = task_rows(rows, 'binary')
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    model = CalibratedClassifierCV(make_classifier('char_tfidf', seed), method='sigmoid', cv=cv)
    model.fit([e.text for e in examples], y)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'model.joblib')
    config = {'kind': 'sinhala-char-tfidf', 'task': 'binary', 'misleading_index': int(list(model.classes_).index(1)),
              'trained_on': 'datasets/sinhala_lirneasia/Corpus.xlsx (CREDIBLE vs FALSE+PARTIAL)',
              'n': len(y), 'n_misleading': int(y.sum()), 'seed': seed, 'calibration': 'sigmoid, 5-fold',
              'cv_auroc': 0.748, 'cv_results': 'results/sinhala/20261123-203433',
              'sklearn': sklearn.__version__}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xlmr', nargs='*', metavar='TASK:PROTOCOL',
                        help='also fine-tune xlm-roberta-base per fold; default binary:random (about 1-2 h on a free CPU)')
    parser.add_argument('--no-labse', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--out', type=Path, default=RESULTS_DIR)
    parser.add_argument('--export', type=Path, metavar='DIR', help='only fit and save the app model to DIR')
    args = parser.parse_args()
    rows = data.load_sinhala()
    if args.export:
        print(f'Wrote {export_model(rows, args.export)}')
        return
    embeddings = None if args.no_labse else labse_embeddings(rows)
    xlmr = () if args.xlmr is None else tuple(tuple(x.split(':')) for x in (args.xlmr or ['binary:random']))
    metrics, oof = run(rows, embeddings, n_bootstrap=args.bootstrap, xlmr=xlmr)
    out = args.out / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    config = {'data': 'datasets/sinhala_lirneasia/Corpus.xlsx', 'seeds': list(SEEDS), 'folds': FOLDS,
              'models': list(MODELS) + (['xlmr'] if xlmr else []), 'labse': None if args.no_labse else LABSE,
              'xlmr': {'model': XLMR, 'runs': [':'.join(x) for x in xlmr], 'epochs': 3, 'lr': 2e-5,
                       'max_length': 128, 'frozen_embeddings': True, 'seed': SEEDS[0]} if xlmr else None, 'bootstrap': args.bootstrap, 'python': platform.python_version()}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    examples, y = task_rows(rows, 'binary')
    with open(out / 'oof_seed42.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        names = list(oof['binary']['random'])
        writer.writerow(['id', 'domain', 'label', *[f'{m}_p_misleading' for m in names]])
        for i, e in enumerate(examples):
            writer.writerow([e.id, e.meta.get('domain', ''), y[i],
                             *[round(float(oof['binary']['random'][m][i, 1]), 6) for m in names]])
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
