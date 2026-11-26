"""C2 experiments E3 (calibration, RQ2) and E4 (abstention, RQ3) for the Trust Score.

Setting: LIAR2, binary real (labels 3-5; half-true counts as real) vs misleading (0-2), as in
research.data.load_liar2(half_true='real').
Signals per claim:
  evidence  s_ev from the C1 run (sentence-level, decisive) on the fact-checker justification.
            Upper bound for evidence quality: this is curated evidence, not live retrieval.
  text      PLACEHOLDER: TF-IDF + logistic regression on LIAR2 train claims. It stands in for
            the BERT text model until that is trained. Not a DeepShield result.
  fused     not available on LIAR2 (no images); always missing.

Data use: LIAR2 val is split 50/50 (stratified, per seed) into a FIT half (weights, and T on
out-of-fold margins) and a CALIBRATION half (conformal qhat). Test is used only for scoring.
Repeated for seeds 42, 43, 44; metrics.json reports mean and sd over seeds, plus bootstrap CIs
on test for seed 42.

E3 systems (P(real) on test):
  text_raw / evidence_raw   the signal alone, uncalibrated (evidence: (s_ev + 1) / 2)
  average_raw               mean of the available raw signals
  text_cal / evidence_cal   Trust Score fitted on that one signal (calibrated single model)
  trust_no_T                learned weights, T = 1
  trust                     learned weights + temperature (the C2 Trust Score)
E4: risk-coverage (confidence threshold at every coverage) for every system, and split
conformal at alpha in ALPHAS: empirical coverage, abstention rate, error on answered items.

Run:   cd backend && python -m research.eval_trust [--text none | --text-export <export_dir>]
       --text-export uses a trained model's LIAR2 predictions (research/MODEL_EXPORT.md)
       instead of the TF-IDF placeholder.
Writes results/c2_trust/<timestamp>/{config.json, metrics.json, signals.csv,
       test_predictions.csv, trust_config.json}. trust_config.json is the seed-42 model; it
       uses the placeholder text signal, so do not load it in the app.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from app.services import trust
from research.data import load_liar2

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'c2_trust'
DEFAULT_C1 = Path(__file__).resolve().parents[2] / 'results' / 'c1_stance' / '20261122-193507' / 'predictions.csv'
C1_COLUMN = 'sentence_s_ev'
SEEDS = (42, 43, 44)
ALPHAS = (0.05, 0.1, 0.2)
BINS = 15


# --------------------------------------------------------------------------- metrics

def reliability(p_real, y, bins: int = BINS) -> list[dict]:
    """Top-label reliability: per confidence bin, mean confidence vs accuracy."""
    p, y = np.asarray(p_real, float), np.asarray(y)
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    index = np.minimum((conf * bins).astype(int), bins - 1)
    return [{'bin': b, 'n': int((index == b).sum()), 'confidence': float(conf[index == b].mean()),
             'accuracy': float(correct[index == b].mean())} for b in range(bins) if (index == b).any()]


def ece(p_real, y, bins: int = BINS) -> float:
    n = len(y)
    return float(sum(r['n'] / n * abs(r['confidence'] - r['accuracy']) for r in reliability(p_real, y, bins)))


def brier(p_real, y) -> float:
    return float(np.mean((np.asarray(p_real, float) - np.asarray(y)) ** 2))


def log_loss(p_real, y) -> float:
    p = np.clip(np.asarray(p_real, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def macro_f1(p_real, y) -> float:
    return float(f1_score(y, (np.asarray(p_real) >= 0.5).astype(int), average='macro', zero_division=0))


def risk_coverage(p_real, y) -> dict:
    """Answer the most confident items first. AURC = mean error over all coverages."""
    p, y = np.asarray(p_real, float), np.asarray(y)
    order = np.argsort(-np.maximum(p, 1 - p), kind='stable')
    wrong = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    risk = np.cumsum(wrong) / np.arange(1, len(y) + 1)
    at = lambda c: float(risk[max(1, math.ceil(c * len(y))) - 1])
    return {'aurc': float(risk.mean()), 'error_at_80': at(0.8), 'error_at_100': at(1.0),
            'curve': [{'coverage': c / 20, 'error': at(c / 20)} for c in range(1, 21)]}


def conformal_metrics(p_real, y, qhat: float) -> dict:
    sets = [trust.prediction_set(p, qhat) for p in p_real]
    names = [trust.REAL if label else trust.MISLEADING for label in y]
    answered = [(s, t) for s, t in zip(sets, names) if len(s) == 1]
    return {'qhat': float(qhat), 'coverage': float(np.mean([t in s for s, t in zip(sets, names)])),
            'abstain_rate': 1 - len(answered) / len(y),
            'error_on_answered': float(np.mean([s[0] != t for s, t in answered])) if answered else None,
            'empty_sets': sum(len(s) == 0 for s in sets)}


def calibration_metrics(p_real, y) -> dict:
    return {'ece': ece(p_real, y), 'brier': brier(p_real, y), 'log_loss': log_loss(p_real, y),
            'macro_f1': macro_f1(p_real, y), **risk_coverage(p_real, y)}


def bootstrap(metric, y, p_a, p_b=None, n: int = 1000, seed: int = 42) -> dict:
    """95% CI of metric(p_a) or, if p_b is given, of metric(p_a) - metric(p_b) (paired)."""
    rng = np.random.default_rng(seed)
    y, p_a = np.asarray(y), np.asarray(p_a)
    p_b = None if p_b is None else np.asarray(p_b)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        values.append(metric(p_a[i], y[i]) - (metric(p_b[i], y[i]) if p_b is not None else 0.0))
    point = metric(p_a, y) - (metric(p_b, y) if p_b is not None else 0.0)
    low, high = np.percentile(values, [2.5, 97.5])
    return {'value': float(point), 'low': float(low), 'high': float(high)}


# --------------------------------------------------------------------------- signals

def tfidf_text_model(train, seed: int = 42):
    """PLACEHOLDER text model: TF-IDF + logistic regression on train claims.
    Returns predict(texts) -> array of P(real)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = [e for e in train if e.binary_label is not None]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(vectorizer.fit_transform([e.text for e in train]), [1 - e.binary_label for e in train])
    return lambda texts: clf.predict_proba(vectorizer.transform(list(texts)))[:, 1]


def tfidf_text_probs(train, others, seed: int = 42) -> dict[str, float]:
    """PLACEHOLDER text signal: P(real) for `others` from tfidf_text_model()."""
    probs = tfidf_text_model(train, seed)([e.text for e in others])
    return {e.id: float(p) for e, p in zip(others, probs)}


def read_c1_scores(path: Path, column: str = C1_COLUMN) -> dict[str, float]:
    with open(path, newline='', encoding='utf-8') as handle:
        return {row['id']: float(row[column]) for row in csv.DictReader(handle) if row.get(column)}


def build_signals(val, test, s_ev: dict[str, float], text_probs: dict[str, float] | None) -> list[dict]:
    rows = []
    for e in [*val, *test]:
        if e.binary_label is None:
            continue
        rows.append({'id': e.id, 'split': e.split, 'liar2_label': e.label, 'y_real': 1 - e.binary_label,
                     'text': text_probs.get(e.id) if text_probs else None, 'fused': None,
                     'evidence': s_ev.get(e.id)})
    return rows


# --------------------------------------------------------------------------- systems

def _only(rows, signal):
    return [{name: (row[name] if name == signal else None) for name in trust.SIGNALS} for row in rows]


def _signals(rows):
    return [{name: row[name] for name in trust.SIGNALS} for row in rows]


def system_probs(fit, cal, test, seed: int) -> dict[str, tuple[list[float], list[float]]]:
    """name -> (P(real) on cal, P(real) on test). Fitting uses `fit` only."""
    have_text = any(row['text'] is not None for row in fit)
    raw = {'evidence': lambda r: 0.5 if r['evidence'] is None else (r['evidence'] + 1) / 2,
           'text': lambda r: 0.5 if r['text'] is None else r['text']}
    out = {'evidence_raw': ([raw['evidence'](r) for r in cal], [raw['evidence'](r) for r in test])}
    if have_text:
        out['text_raw'] = ([raw['text'](r) for r in cal], [raw['text'](r) for r in test])
        avg = lambda r: float(np.mean([f(r) for name, f in raw.items() if r[name] is not None] or [0.5]))
        out['average_raw'] = ([avg(r) for r in cal], [avg(r) for r in test])
    y_fit = [r['y_real'] for r in fit]
    singles = ['evidence', 'text'] if have_text else ['evidence']
    for signal in singles:
        model = trust.fit_trust(_only(fit, signal), y_fit, seed)
        out[f'{signal}_cal'] = ([model.probability(s) for s in _only(cal, signal)],
                                [model.probability(s) for s in _only(test, signal)])
    model = trust.fit_trust(_signals(fit), y_fit, seed)
    out['trust'] = ([model.probability(s) for s in _signals(cal)], [model.probability(s) for s in _signals(test)])
    untempered = trust.TrustModel(model.weights, model.bias)
    out['trust_no_T'] = ([untempered.probability(s) for s in _signals(cal)],
                         [untempered.probability(s) for s in _signals(test)])
    return out


def split_val(val_rows, seed: int):
    fit, cal = train_test_split(val_rows, test_size=0.5, random_state=seed, stratify=[r['y_real'] for r in val_rows])
    return fit, cal


def evaluate_seed(rows, seed: int, alphas=ALPHAS) -> dict:
    fit, cal = split_val([r for r in rows if r['split'] == 'val'], seed)
    test = [r for r in rows if r['split'] == 'test']
    y_cal, y_test = [r['y_real'] for r in cal], [r['y_real'] for r in test]
    result = {}
    for name, (p_cal, p_test) in system_probs(fit, cal, test, seed).items():
        result[name] = calibration_metrics(p_test, y_test)
        result[name]['conformal'] = {str(a): conformal_metrics(p_test, y_test, trust.conformal_qhat(p_cal, y_cal, a))
                                     for a in alphas}
    return result


def summarise(per_seed: dict[int, dict]) -> dict:
    """Mean and sd over seeds for every numeric leaf (curves and counts included)."""
    def merge(values):
        first = values[0]
        if isinstance(first, dict):
            return {k: merge([v[k] for v in values]) for k in first}
        if isinstance(first, list):
            return [merge([v[i] for v in values]) for i in range(len(first))]
        if first is None or any(v is None for v in values):
            return None
        return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return merge(list(per_seed.values()))


# --------------------------------------------------------------------------- run

def run(rows, seeds=SEEDS, alphas=ALPHAS, n_bootstrap: int = 1000) -> tuple[dict, trust.TrustModel, dict]:
    per_seed = {seed: evaluate_seed(rows, seed, alphas) for seed in seeds}
    seed = seeds[0]
    fit, cal = split_val([r for r in rows if r['split'] == 'val'], seed)
    test = [r for r in rows if r['split'] == 'test']
    y = [r['y_real'] for r in test]
    probs = {name: p_test for name, (_, p_test) in system_probs(fit, cal, test, seed).items()}

    singles = [name for name in ('evidence_cal', 'text_cal') if name in probs]
    ci = {'trust': {m: bootstrap(f, y, probs['trust'], n=n_bootstrap, seed=seed)
                    for m, f in (('ece', ece), ('brier', brier))}}
    for name in singles:
        ci[f'trust_minus_{name}'] = {m: bootstrap(f, y, probs['trust'], probs[name], n=n_bootstrap, seed=seed)
                                     for m, f in (('ece', ece), ('brier', brier))}

    model = trust.fit_trust(_signals(fit), [r['y_real'] for r in fit], seed)
    model.alpha = 0.1
    model.qhat = trust.conformal_qhat([model.probability(s) for s in _signals(cal)], [r['y_real'] for r in cal], 0.1)
    by_label = {}
    for label in sorted({r['liar2_label'] for r in test}):
        p = [pr for pr, r in zip(probs['trust'], test) if r['liar2_label'] == label]
        by_label[label] = {'n': len(p), 'mean_p_real': float(np.mean(p))}
    metrics = {
        'n': {'val_fit': len(fit), 'val_cal': len(cal), 'test': len(test),
              'test_evidence_missing': sum(r['evidence'] is None for r in test)},
        'summary_over_seeds': summarise(per_seed),
        'per_seed': {str(s): v for s, v in per_seed.items()},
        'bootstrap_seed42': ci,
        'reliability_seed42': {name: reliability(p, y) for name, p in probs.items()},
        'trust_p_real_by_liar2_label_seed42': by_label,
        'trust_model_seed42': model.to_dict(),
    }
    predictions = {'id': [r['id'] for r in test], 'liar2_label': [r['liar2_label'] for r in test], 'y_real': y, **probs}
    return metrics, model, predictions


def print_table(metrics: dict) -> None:
    s = metrics['summary_over_seeds']
    fmt = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}"
    print(f"{'system':14s} {'ECE':>13s} {'Brier':>13s} {'macro-F1':>13s} {'err@80%':>13s} {'AURC':>13s}")
    for name, m in s.items():
        print(f"{name:14s} {fmt(m['ece']):>13s} {fmt(m['brier']):>13s} {fmt(m['macro_f1']):>13s} "
              f"{fmt(m['error_at_80']):>13s} {fmt(m['aurc']):>13s}")
    print('\ntrust, split conformal:')
    for alpha, c in s['trust']['conformal'].items():
        err = fmt(c['error_on_answered']) if c['error_on_answered'] else 'n/a'
        print(f"  alpha={alpha:<5s} coverage {fmt(c['coverage'])}  abstain {fmt(c['abstain_rate'])}  error on answered {err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--c1-predictions', type=Path, default=DEFAULT_C1)
    parser.add_argument('--text', choices=('tfidf', 'none'), default='tfidf')
    parser.add_argument('--text-export', type=Path, help='trained text model export; overrides --text')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--only-scored', action='store_true',
                        help='keep only claims listed in --c1-predictions (e.g. the live-evidence sample from '
                             'eval_live --val-scores), so unscored claims do not enter as "evidence missing"')
    parser.add_argument('--out', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    val, test = load_liar2('val'), load_liar2('test')
    s_ev = read_c1_scores(args.c1_predictions)
    if args.only_scored:
        with open(args.c1_predictions, newline='', encoding='utf-8') as handle:
            listed = {row['id'] for row in csv.DictReader(handle)}
        val, test = [e for e in val if e.id in listed], [e for e in test if e.id in listed]
    if args.text_export:
        from research.check_export import read_export_probs
        text_probs = {**read_export_probs(args.text_export, 'liar2', 'val'),
                      **read_export_probs(args.text_export, 'liar2', 'test')}
        text_name = f'export:{args.text_export}'
    elif args.text == 'tfidf':
        text_probs, text_name = tfidf_text_probs(load_liar2('train'), [*val, *test], SEEDS[0]), 'tfidf_placeholder'
    else:
        text_probs, text_name = None, None
    rows = build_signals(val, test, s_ev, text_probs)
    metrics, model, predictions = run(rows, n_bootstrap=args.bootstrap)
    model.meta = {'text_signal': text_name, 'evidence': str(args.c1_predictions),
                  'note': 'Research artifact fitted on LIAR2; check the text signal before loading it in the app.'}

    out = args.out / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    config = {'dataset': 'liar2', 'half_true': 'real', 'c1_predictions': str(args.c1_predictions),
              'c1_column': C1_COLUMN, 'text_signal': text_name, 'only_scored': args.only_scored,
              'seeds': list(SEEDS), 'alphas': list(ALPHAS), 'bins': BINS, 'bootstrap': args.bootstrap,
              'python': platform.python_version()}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    model.save(out)
    with open(out / 'signals.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / 'test_predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(list(predictions))
        writer.writerows(zip(*predictions.values()))
    print(f'Wrote {out}')
    print_table(metrics)


if __name__ == '__main__':
    main()
