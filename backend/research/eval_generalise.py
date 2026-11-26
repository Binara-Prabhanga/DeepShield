"""C4 experiment E6 (RQ5): do results travel? A shortcut-aware evaluation protocol.

For every model, three kinds of number (plan, C4):
  in-dataset     the test split of the dataset it was trained on
  cross-dataset  liar2 -> liar_unseen (LIAR claims absent from LIAR2), liar2 -> fakeddit (Reddit titles),
                 fakeddit -> liar2, fakeddit_official -> liar2 (official sample, when fetched)
  future         liar2_temporal: trained on claims before 2018, tested on 2019-2023 ('future'),
                 with an in-period test from the same years for comparison
plus the Sri Lankan test set (srilanka: FactSeeker verdicts, research/collect_srilanka.py) when labelled.csv exists.
Reported per target: macro-F1, AUROC (unaffected by the label-share shift), ECE, Brier, the
misleading share, and the macro-F1 drop from in-dataset with a 95% bootstrap CI.

Shortcut probes: small logistic regressions trained on the same train split that see ONLY
  surface   length, digits, quotes, '!', '?', capitals, a leading 'Says', '%'
  metadata  speaker, context, subject, state (LIAR2 sources)
  year      claim year (LIAR2 random split only; the temporal split makes it meaningless)
  subreddit the post's subreddit (official Fakeddit; its labels were assigned per subreddit)
If a probe comes close to the model, the model's in-dataset score may come from a shortcut.

Models:
  --export DIR [DIR ...]  trained exports (research/MODEL_EXPORT.md); mean and sd over their seeds
  default                 PLACEHOLDER TF-IDF + logistic regression trained here on each source
                          (deterministic, so one seed). Not a DeepShield result.

Run:   cd backend && python -m research.eval_generalise [--export DIR ...]
Writes results/c4_generalise/<timestamp>/{config.json, metrics.json}.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from research import data
from research.eval_trust import brier, ece

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'c4_generalise'
PROTOCOL = {
    'liar2': {'in': ('liar2', 'test'), 'targets': [('liar_unseen', 'test'), ('fakeddit', 'test')]},
    'liar2_temporal': {'in': ('liar2_temporal', 'test'), 'targets': [('liar2_temporal', 'future')]},
    'fakeddit': {'in': ('fakeddit', 'test'), 'targets': [('liar2', 'test')]},
    'fakeddit_official': {'in': ('fakeddit_official', 'test'), 'targets': [('liar2', 'test')]},
}
LOCAL_TARGET = ('srilanka', 'test')
SEED = 42


def load_split(dataset: str, split: str) -> list[data.Example]:
    """Labelled examples (binary_label not None) for any dataset name used in the protocol."""
    if dataset == 'liar2':
        rows = data.load_liar2(split)
    elif dataset == 'liar2_temporal':
        rows = data.load_liar2_temporal(split)
    elif dataset == 'liar':
        rows = data.load_liar_binary(split)
    elif dataset == 'liar_unseen':
        rows = data.load_liar_unseen(split)
    elif dataset == 'fakeddit':
        rows = data.load_fakeddit(split, check_images=False)
    elif dataset == 'fakeddit_official':
        rows = data.load_fakeddit_official(split)
    elif dataset == 'srilanka':
        rows = data.load_local_labelled()
    else:
        raise ValueError(f'Unknown dataset {dataset!r}')
    return [e for e in rows if e.binary_label is not None]


def local_set_available() -> bool:
    return (data.data_dir() / 'srilanka_local' / 'labelled.csv').is_file()


def source_available(source: str) -> bool:
    """Every source ships with datasets/ except the official Fakeddit sample (research/fetch_fakeddit.py)."""
    return source != 'fakeddit_official' or (data.data_dir() / 'fakeddit_official' / 'sample_test.csv').is_file()


# --------------------------------------------------------------------------- metrics

def target_metrics(p_real, y_real) -> dict:
    p, y = np.asarray(p_real, float), np.asarray(y_real)
    return {'n': int(len(y)), 'misleading_share': float(1 - y.mean()),
            'macro_f1': float(f1_score(y, (p >= 0.5).astype(int), average='macro', zero_division=0)),
            'auroc': float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
            'ece': ece(p, y), 'brier': brier(p, y)}


def bootstrap_drop(p_in, y_in, p_out, y_out, n: int = 1000, seed: int = SEED) -> dict:
    """macro-F1(in) - macro-F1(out), resampling both test sets independently."""
    def f1(p, y):
        return f1_score(y, (p >= 0.5).astype(int), average='macro', zero_division=0)
    p_in, y_in, p_out, y_out = map(np.asarray, (p_in, y_in, p_out, y_out))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i, j = rng.integers(0, len(y_in), len(y_in)), rng.integers(0, len(y_out), len(y_out))
        values.append(f1(p_in[i], y_in[i]) - f1(p_out[j], y_out[j]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {'value': float(f1(p_in, y_in) - f1(p_out, y_out)), 'low': float(low), 'high': float(high)}


def mean_sd(values: list[dict]) -> dict:
    """Mean and sd over seeds for each numeric field."""
    out = {}
    for key in values[0]:
        vals = [v[key] for v in values]
        if all(isinstance(x, int) for x in vals) and len(set(vals)) == 1:     # counts such as n
            out[key] = vals[0]
        elif any(x is None for x in vals) or not isinstance(vals[0], (int, float)):
            out[key] = vals[0] if len(set(map(str, vals))) == 1 else None
        else:
            out[key] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out


# --------------------------------------------------------------------------- shortcut probes

def surface_features(text: str) -> list[float]:
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return [len(text), len(words), sum(c.isdigit() for c in text), text.count('"') + text.count('“'),
            text.count('!'), text.count('?'), sum(c.isupper() for c in letters) / max(1, len(letters)),
            float(bool(re.match(r'\s*says\b', text, re.I))), float('%' in text or 'percent' in text.lower())]


def metadata_features(e: data.Example) -> dict:
    return {f'{k}={str(e.meta.get(k, "")).strip().lower()}': 1.0 for k in ('speaker', 'context', 'subject', 'state_info')}


def year_features(e: data.Example) -> dict:
    return {f'year={e.date.year if e.date else "none"}': 1.0}


def subreddit_features(e: data.Example) -> dict:
    return {f'subreddit={e.meta.get("subreddit", "")}': 1.0}


def probe(kind: str, train: list[data.Example], test: list[data.Example], seed: int = SEED) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y_train = [1 - e.binary_label for e in train]
    if kind == 'surface':
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        x_train, x_test = [surface_features(e.text) for e in train], [surface_features(e.text) for e in test]
    else:
        features = {'metadata': metadata_features, 'year': year_features, 'subreddit': subreddit_features}[kind]
        model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, random_state=seed))
        x_train, x_test = [features(e) for e in train], [features(e) for e in test]
    model.fit(x_train, y_train)
    return target_metrics(model.predict_proba(x_test)[:, 1], [1 - e.binary_label for e in test])


PROBES = {'liar2': ('surface', 'metadata', 'year'), 'liar2_temporal': ('surface', 'metadata'), 'fakeddit': ('surface',),
          'fakeddit_official': ('surface', 'subreddit')}


# --------------------------------------------------------------------------- models

def placeholder_predictions(source: str, targets: list[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """{target: [ {id: P(real)} ]} from the TF-IDF placeholder trained on the source's train split."""
    from research.eval_trust import tfidf_text_model
    predict = tfidf_text_model(load_split(source, 'train'), SEED)
    out = {}
    for target in targets:
        rows = load_split(*target)
        out[target] = [dict(zip([e.id for e in rows], map(float, predict([e.text for e in rows]))))]
    return out


def export_predictions(export_dir: Path, targets) -> tuple[str, dict]:
    from research.check_export import read_export_probs
    manifest = json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))
    out = {}
    for target in targets:
        # Seeds that have this file, weights seed first (targets added later by predict_export.py exist only
        # for the weights seed, so their index 0 pairs with the in-dataset run of the same seed).
        seeds = [s for s in manifest['seeds']
                 if (export_dir / 'predictions' / f'seed{s}' / f'{target[0]}_{target[1]}.csv').is_file()]
        if seeds and seeds[0] == manifest['seeds'][0]:
            out[target] = [read_export_probs(export_dir, *target, seed) for seed in seeds]
    return manifest['trained_on'], out


def evaluate_source(source: str, predictions: dict, n_bootstrap: int) -> dict:
    """predictions: {target: [ {id: P(real)} per seed ]}, including the in-dataset target."""
    spec = PROTOCOL[source]
    gold = {t: {e.id: 1 - e.binary_label for e in load_split(*t)} for t in predictions}
    in_target = spec['in']
    if in_target not in predictions:
        raise ValueError(f'{source}: in-dataset predictions {in_target} are missing')
    result = {}
    for target, per_seed in predictions.items():
        runs, drops = [], []
        for seed_index, probs in enumerate(per_seed):
            ids = [i for i in gold[target] if i in probs]
            p, y = [probs[i] for i in ids], [gold[target][i] for i in ids]
            runs.append(target_metrics(p, y))
            if target != in_target:
                in_probs = predictions[in_target][seed_index]
                in_ids = [i for i in gold[in_target] if i in in_probs]
                drops.append(bootstrap_drop([in_probs[i] for i in in_ids], [gold[in_target][i] for i in in_ids],
                                            p, y, n_bootstrap))
        name = f'{target[0]}/{target[1]}'
        result[name] = {'kind': 'in-dataset' if target == in_target else ('future' if target[1] == 'future' else 'cross-dataset'),
                        **mean_sd(runs)}
        if drops:
            result[name]['macro_f1_drop'] = drops[0] if len(drops) == 1 else mean_sd(drops)
    return result


def run(exports: list[Path] | None, n_bootstrap: int = 1000, sources=None) -> dict:
    metrics = {'models': {}, 'shortcut_probes': {}}
    sources = sources or tuple(s for s in PROTOCOL if source_available(s))
    local = [LOCAL_TARGET] if local_set_available() else []
    if exports:
        for export_dir in exports:
            source, preds = export_predictions(export_dir, [PROTOCOL[s]['in'] for s in PROTOCOL] +
                                               [t for s in PROTOCOL for t in PROTOCOL[s]['targets']] + local)
            if source not in PROTOCOL:
                raise ValueError(f'{export_dir}: trained_on {source!r} has no protocol')
            wanted = {PROTOCOL[source]['in'], *PROTOCOL[source]['targets'], *local}
            metrics['models'][f'{source} ({export_dir.name})'] = evaluate_source(
                source, {t: p for t, p in preds.items() if t in wanted}, n_bootstrap)
    else:
        for source in sources:
            targets = [PROTOCOL[source]['in'], *PROTOCOL[source]['targets'], *local]
            metrics['models'][f'{source} (tfidf placeholder)'] = evaluate_source(
                source, placeholder_predictions(source, targets), n_bootstrap)
    for source in sources:
        train, test = load_split(source, 'train'), load_split(*PROTOCOL[source]['in'])
        metrics['shortcut_probes'][source] = {kind: probe(kind, train, test) for kind in PROBES[source]}
    return metrics


def print_table(metrics: dict) -> None:
    def num(x):                                                     # plain number or {'mean', 'sd'} over seeds
        return x['mean'] if isinstance(x, dict) else x

    def val(d, key):
        x = d.get(key)
        return '   n/a' if x is None else f'{num(x):.3f}'
    for model, targets in metrics['models'].items():
        print(f'\n{model}')
        print(f"  {'target':26s} {'kind':14s} {'n':>6s} {'mis%':>6s} {'F1':>6s} {'AUROC':>6s} {'ECE':>6s}  F1 drop [95% CI]")
        for name, m in targets.items():
            drop = m.get('macro_f1_drop')
            drop_text = (f"{num(drop['value']):+.3f} [{num(drop['low']):+.3f}, {num(drop['high']):+.3f}]"
                         if drop and 'value' in drop else '')
            n = m['n']['mean'] if isinstance(m['n'], dict) else m['n']
            print(f"  {name:26s} {m['kind']:14s} {n:6.0f} {val(m, 'misleading_share'):>6s} {val(m, 'macro_f1'):>6s} "
                  f"{val(m, 'auroc'):>6s} {val(m, 'ece'):>6s}  {drop_text}")
    print('\nshortcut probes (in-dataset test):')
    for source, probes in metrics['shortcut_probes'].items():
        print('  ' + source + ': ' + ', '.join(f"{k} F1 {v['macro_f1']:.3f} / AUROC {v['auroc']:.3f}"
                                              for k, v in probes.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', type=Path, nargs='*')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--out', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    metrics = run(args.export, args.bootstrap)
    out = args.out / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    config = {'protocol': {s: {'in': list(p['in']), 'targets': [list(t) for t in p['targets']]} for s, p in PROTOCOL.items()},
              'models': [str(e) for e in args.export] if args.export else 'tfidf_placeholder',
              'local_set': local_set_available(), 'temporal': {'past_end': str(data.TEMPORAL_PAST_END),
                                                               'future_start': str(data.TEMPORAL_FUTURE_START)},
              'seed': SEED, 'bootstrap': args.bootstrap, 'python': platform.python_version()}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Wrote {out}')
    print_table(metrics)


if __name__ == '__main__':
    main()
