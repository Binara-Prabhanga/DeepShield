"""C1 experiment: does NLI stance beat similarity-only evidence scoring? (RQ1 / G1)

Setting: LIAR2 claim + its fact-checker justification as the one evidence item.
Gold stance: labels 4-5 (mostly-true, true) -> supports; 0-1 (pants-fire, false) -> refutes.
Labels 2-3 are left out of the scored task; their mean s_ev is reported as a sanity check.

Justifications are curated "gold" evidence written by the fact-checker, so this tests the
stance step on its own, not live retrieval.

Systems (thresholds are tuned on val, scored on test):
  similarity       TF-IDF relevance from rank_evidence(); relevance >= t -> supports
  <v>_argmax       NLI zero-shot; P_support > P_refute -> supports
  <v>_tuned        NLI; s_ev = P_support - P_refute >= t -> supports
where <v> is an NLI variant (see stance.premise_scores / stance.aggregate):
  passage              the whole justification is one premise
  sentence_decisive    per-sentence NLI, restatements skipped, most decisive sentence wins
  sentence_max         per-sentence; strongest support and strongest refute both count
  sentence_mean        per-sentence; average

Choose a variant by its VAL macro-F1 (metrics.json -> val_macro_f1), never by test.

Run:      cd backend && python -m research.eval_stance [--limit 500]
Re-score: python -m research.eval_stance --from-scores results/c1_stance/<run>/premise_scores.jsonl
          (no model needed; use after changing an aggregation rule)
Writes results/c1_stance/<timestamp>/{config.json, metrics.json, predictions.csv, premise_scores.jsonl}.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from app.services.evidence import rank_evidence
from app.services.stance import AGGREGATIONS, aggregate, premise_scores, transformers_nli
from research.data import load_liar2

DEFAULT_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'c1_stance'
SUPPORTS, REFUTES = 1, 0   # gold encoding inside this script
GOLD = {0: REFUTES, 1: REFUTES, 4: SUPPORTS, 5: SUPPORTS}
VARIANTS = ('passage',) + tuple(f'sentence_{method}' for method in AGGREGATIONS)


# --------------------------------------------------------------------------- metrics (pure)

def tune_threshold(scores, gold) -> float:
    """Threshold t maximising macro-F1 for 'score >= t -> supports'. Ties pick the smallest t."""
    scores, gold = np.asarray(scores, dtype=float), np.asarray(gold)
    candidates = np.unique(scores)
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        f1 = f1_score(gold, (scores >= t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def binary_metrics(gold, pred) -> dict:
    gold, pred = np.asarray(gold), np.asarray(pred)
    p, r, f, n = precision_recall_fscore_support(gold, pred, labels=[REFUTES, SUPPORTS], zero_division=0)
    return {'n': int(len(gold)), 'macro_f1': float(f.mean()), 'accuracy': float((gold == pred).mean()),
            'refutes': {'precision': float(p[0]), 'recall': float(r[0]), 'f1': float(f[0]), 'support': int(n[0])},
            'supports': {'precision': float(p[1]), 'recall': float(r[1]), 'f1': float(f[1]), 'support': int(n[1])},
            'confusion_rows_gold_refutes_supports': confusion_matrix(gold, pred, labels=[REFUTES, SUPPORTS]).tolist()}


def bootstrap_macro_f1(gold, pred_a, pred_b=None, n: int = 1000, seed: int = 42) -> dict:
    """95% percentile CI of macro-F1 for pred_a, or of (a - b) when pred_b is given (paired)."""
    gold, pred_a = np.asarray(gold), np.asarray(pred_a)
    pred_b = None if pred_b is None else np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(gold), len(gold))
        value = f1_score(gold[idx], pred_a[idx], average='macro', zero_division=0)
        if pred_b is not None:
            value -= f1_score(gold[idx], pred_b[idx], average='macro', zero_division=0)
        values.append(value)
    low, high = np.percentile(values, [2.5, 97.5])
    return {'low': float(low), 'high': float(high), 'resamples': n, 'seed': seed}


def mcnemar_exact(gold, pred_a, pred_b) -> dict:
    """Exact McNemar test on the items where exactly one system is correct."""
    gold = np.asarray(gold)
    a_right, b_right = np.asarray(pred_a) == gold, np.asarray(pred_b) == gold
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {'only_a_correct': only_a, 'only_b_correct': only_b, 'p_value': float(p)}


# --------------------------------------------------------------------------- scoring

_PROB_KEYS = {'supports': 'p_support', 'refutes': 'p_refute', 'unrelated': 'p_unrelated'}


def nli_argmax_label(row: dict, variant: str) -> str:
    """3-way NLI label (how often the model calls gold evidence 'unrelated' is worth reporting)."""
    return max(_PROB_KEYS, key=lambda label: row[f'{variant}_{_PROB_KEYS[label]}'])


def score_split(examples, nli) -> list[dict]:
    """Relevance plus raw per-premise NLI scores ('passage' and 'sentences') for each
    (claim, justification) pair. Uses stance.premise_scores, the app's code path."""
    usable = [e for e in examples if e.meta['justification']]
    pairs = [(e.text, e.meta['justification']) for e in usable]
    passage = premise_scores(pairs, nli, 'passage')
    sentences = premise_scores(pairs, nli, 'sentence')
    rows = []
    for e, p, sents in zip(usable, passage, sentences, strict=True):
        item = {'title': '', 'snippet': e.meta['justification'], 'url': f'liar2://{e.id}', 'source_type': 'web_search'}
        rows.append({'id': e.id, 'split': e.split, 'liar2_label': e.label, 'gold': GOLD.get(e.label),
                     'relevance': rank_evidence(e.text, [item])[0]['relevance'],
                     'passage': p[0][1], 'sentences': [[text, probs] for text, probs in sents]})
    return rows


def add_variants(rows: list[dict]) -> list[dict]:
    """Add <variant>_p_support / _p_refute / _p_unrelated / _s_ev columns from the raw scores."""
    for row in rows:
        combined = {'passage': row['passage']}
        for method in AGGREGATIONS:
            combined[f'sentence_{method}'] = aggregate([probs for _, probs in row['sentences']], method)
        for variant, p in combined.items():
            row |= {f'{variant}_p_support': p['entailment'], f'{variant}_p_refute': p['contradiction'],
                    f'{variant}_p_unrelated': p['neutral'], f'{variant}_s_ev': p['entailment'] - p['contradiction']}
    return rows


def sample(examples, limit: int | None, seed: int):
    if not limit or limit >= len(examples):
        return examples
    return sorted(random.Random(seed).sample(examples, limit), key=lambda e: e.id)


def _predict(rows, thresholds):
    preds = {'similarity': np.array([int(r['relevance'] >= thresholds['similarity']) for r in rows])}
    for v in VARIANTS:
        preds[f'{v}_argmax'] = np.array([int(r[f'{v}_p_support'] > r[f'{v}_p_refute']) for r in rows])
        preds[f'{v}_tuned'] = np.array([int(r[f'{v}_s_ev'] >= thresholds[v]) for r in rows])
    return preds


def evaluate(val_rows: list[dict], test_rows: list[dict], bootstrap: int, seed: int) -> dict:
    val = [r for r in val_rows if r['gold'] is not None]
    test = [r for r in test_rows if r['gold'] is not None]
    gold_val = np.array([r['gold'] for r in val])
    gold = np.array([r['gold'] for r in test])
    thresholds = {'similarity': tune_threshold([r['relevance'] for r in val], gold_val)}
    for v in VARIANTS:
        thresholds[v] = tune_threshold([r[f'{v}_s_ev'] for r in val], gold_val)
    val_preds, preds = _predict(val, thresholds), _predict(test, thresholds)
    for name, pred in preds.items():
        for row, value in zip(test, pred):
            row[f'pred_{name}'] = int(value)

    # G1 subset: refuted claims whose evidence is most similar (top quarter of test relevance).
    q75 = float(np.percentile([r['relevance'] for r in test], 75))
    g1 = np.array([r['gold'] == REFUTES and r['relevance'] >= q75 for r in test])

    systems = {}
    for name, pred in preds.items():
        systems[name] = binary_metrics(gold, pred) | {
            'macro_f1_ci95': bootstrap_macro_f1(gold, pred, n=bootstrap, seed=seed),
            'g1_refute_recall': float((pred[g1] == REFUTES).mean()) if g1.any() else None}

    def paired(a: str, b: str) -> dict:
        return {'macro_f1_diff_ci95': bootstrap_macro_f1(gold, preds[a], preds[b], bootstrap, seed),
                'mcnemar': mcnemar_exact(gold, preds[a], preds[b])}

    comparisons = {f'{name}_vs_similarity': paired(name, 'similarity') for name in preds if name != 'similarity'}
    for v in VARIANTS[1:]:
        comparisons[f'{v}_tuned_vs_passage_tuned'] = paired(f'{v}_tuned', 'passage_tuned')

    per_variant = {}
    for v in VARIANTS:
        by_label = {}
        for label in range(6):
            values = [r[f'{v}_s_ev'] for r in test_rows if r['liar2_label'] == label]
            by_label[label] = {'n': len(values), 'mean_s_ev': float(np.mean(values)) if values else None}
        rho = spearmanr([r['liar2_label'] for r in test_rows], [r[f'{v}_s_ev'] for r in test_rows])
        per_variant[v] = {
            'nli_label_share': {label: float(np.mean([nli_argmax_label(r, v) == label for r in test_rows]))
                                for label in _PROB_KEYS},
            's_ev_by_liar2_label': by_label,
            'spearman_label_vs_s_ev': {'rho': float(rho.statistic), 'p_value': float(rho.pvalue)}}
    return {
        'thresholds_tuned_on_val': thresholds,
        'val_macro_f1': {name: float(f1_score(gold_val, pred, average='macro', zero_division=0))
                         for name, pred in val_preds.items()},
        'val_note': 'tuned systems are scored on the split their threshold was fit on (optimistic); '
                    'use val_macro_f1 to choose a variant, test_systems to report it',
        'test_systems': systems,
        'g1_subset': {'definition': 'gold refutes and relevance >= test 75th percentile',
                      'relevance_q75': q75, 'n': int(g1.sum())},
        'paired_comparisons': comparisons,
        'test_diagnostics_by_variant': per_variant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--limit', type=int, default=None, help='random rows per split (quick runs)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--out', type=Path, default=RESULTS_DIR)
    parser.add_argument('--from-scores', type=Path, default=None,
                        help='premise_scores.jsonl from an earlier run: re-aggregate without the model')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    config = {'seed': args.seed, 'bootstrap': args.bootstrap, 'dataset': 'liar2 val (tune) / test (score)',
              'gold_mapping': 'labels 4-5 supports, 0-1 refutes, 2-3 excluded from scoring',
              'evidence': 'cleaned LIAR2 justification (gold evidence, not live retrieval)',
              'variants': list(VARIANTS), 'python': platform.python_version()}
    if args.from_scores:
        with open(args.from_scores, encoding='utf-8') as handle:
            all_rows = [json.loads(line) for line in handle if line.strip()]
        source_config = json.loads((args.from_scores.parent / 'config.json').read_text(encoding='utf-8'))
        config |= {key: source_config.get(key) for key in ('model', 'model_revision', 'limit', 'torch')}
        config['rescored_from'] = str(args.from_scores)
        rows = {split: [r for r in all_rows if r['split'] == split] for split in ('val', 'test')}
    else:
        import torch
        torch.manual_seed(args.seed)
        from app.services.stance import _nli_bundle
        nli = transformers_nli(args.model, batch_size=args.batch_size)
        rows = {}
        for split in ('val', 'test'):
            examples = sample(load_liar2(split), args.limit, args.seed)
            print(f'{split}: scoring {len(examples)} pairs ...', flush=True)
            rows[split] = score_split(examples, nli)
        config |= {'model': args.model, 'limit': args.limit, 'torch': torch.__version__,
                   'model_revision': getattr(_nli_bundle(args.model)[1].config, '_commit_hash', None)}
    for split_rows in rows.values():
        add_variants(split_rows)
    metrics = evaluate(rows['val'], rows['test'], args.bootstrap, args.seed)
    config['runtime_seconds'] = round(time.time() - started, 1)

    out = args.out / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    raw_keys = ('id', 'split', 'liar2_label', 'gold', 'relevance', 'passage', 'sentences')
    with open(out / 'premise_scores.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows['val'] + rows['test']:
            handle.write(json.dumps({key: row[key] for key in raw_keys}, ensure_ascii=False) + '\n')
    fields = ['id', 'split', 'liar2_label', 'gold', 'relevance'] + [
        f'{v}_{key}' for v in VARIANTS for key in ('p_support', 'p_refute', 'p_unrelated', 's_ev')] + [
        f'pred_{name}' for name in metrics['test_systems']]
    with open(out / 'predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows['val'] + rows['test'])

    print(f'\nWrote {out}')
    print(f"{'system':28s} {'val F1':>7s} {'test F1':>8s}  test 95% CI      refR  supR  G1")
    for name, m in metrics['test_systems'].items():
        ci = m['macro_f1_ci95']
        print(f"{name:28s} {metrics['val_macro_f1'][name]:7.3f} {m['macro_f1']:8.3f}  [{ci['low']:.3f}, {ci['high']:.3f}]"
              f"  {m['refutes']['recall']:.2f}  {m['supports']['recall']:.2f}  {m['g1_refute_recall']:.2f}")


if __name__ == '__main__':
    main()
