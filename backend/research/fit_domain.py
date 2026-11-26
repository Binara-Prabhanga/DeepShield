"""Fit the out-of-domain check (app/services/domain.py) and measure what it does.

  python -m research.fit_domain --out artifacts/domain-liar2 --results ../results/domain

Fit: sentence embeddings (--model) of the LIAR2 train claims. A claim's similarity = mean cosine to its --k nearest
train claims (k=25 separated other datasets better than k=1: AUROC 0.966/0.955 vs 0.955/0.932). The threshold is
the --quantile (default 5%) of the LIAR2 *validation* claims' similarities, so about 5% of in-domain claims are
sent to review. Measured on held-out data: how different the other datasets look (AUROC of "far from LIAR2"),
and whether the text model really is worse on the claims the check rejects (LIAR2 test, 500 claims from the
live run's items.csv, if given). Fixed seed: nothing here is random.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from research.data import (load_fakeddit_official, load_liar2, load_local_labelled)


def embed(texts: list[str], model: str, batch: int = 128) -> np.ndarray:
    from app.services.evidence import embed_texts
    return np.vstack([embed_texts(texts[i:i + batch], model) for i in range(0, len(texts), batch)])


def fit(train_texts: list[str], model: str, k: int) -> dict:
    return {'model': model, 'k': k, 'matrix': embed(train_texts, model).astype(np.float16)}


def similarities(bundle: dict, texts: list[str]) -> np.ndarray:
    """Mean similarity to the bundle's k nearest training claims."""
    sims = embed(texts, bundle['model']) @ bundle['matrix'].astype(np.float32).T
    k = int(bundle['k'])
    return np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('artifacts/domain-liar2'))
    parser.add_argument('--results', type=Path, default=Path('../results/domain'))
    parser.add_argument('--quantile', type=float, default=0.05)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--k', type=int, default=25, help='average the similarity of the k nearest training claims')
    parser.add_argument('--items', type=Path, default=Path('../results/live_evidence/20261124-factcheck-all/items.csv'),
                        help='live-run items.csv (id, y_real, p_real) to check accuracy on rejected claims')
    args = parser.parse_args()
    from sklearn.metrics import roc_auc_score

    train = [e.text for e in load_liar2('train')]
    bundle = fit(train, args.model, args.k)
    val = [e.text for e in load_liar2('val')]
    val_sim = similarities(bundle, val)
    threshold = float(np.quantile(val_sim, args.quantile))
    bundle.update({'threshold': threshold, 'quantile': args.quantile, 'n_train': len(train)})
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out / 'domain.joblib')

    test = load_liar2('test')
    test_sim = similarities(bundle, [e.text for e in test])
    metrics = {'n_train': len(train), 'threshold': threshold, 'quantile': args.quantile,
               'liar2_val_sim_median': float(np.median(val_sim)), 'liar2_test_rejected': float((test_sim < threshold).mean())}
    others = {'fakeddit_official_test (Reddit titles)': [e.text for e in load_fakeddit_official('test')][:1000]}
    try:
        others['sri_lanka_fact_checks'] = [e.text for e in load_local_labelled()]
    except Exception as exc:   # optional dataset
        metrics['sri_lanka_error'] = type(exc).__name__
    others['examples'] = [
        'The 2020 Summer Olympics were held in Tokyo, Japan, in July and August 2021 after being postponed for a year because of the COVID-19 pandemic.',
        'The Sri Lankan government announced that the Moon will be renamed after a local cricketer, and every school in the country will close for a full year to celebrate.',
        'Unemployment doubled under Governor Walker in Wisconsin.']
    metrics['other_datasets'] = {}
    for name, texts in others.items():
        sim = similarities(bundle, texts)
        entry = {'n': len(texts), 'median_sim': float(np.median(sim)), 'rejected': float((sim < threshold).mean())}
        if len(texts) > 10:
            labels = np.r_[np.zeros(len(test_sim)), np.ones(len(sim))]
            entry['auroc_far_from_liar2'] = float(roc_auc_score(labels, -np.r_[test_sim, sim]))
        else:
            entry['similarities'] = [round(float(s), 3) for s in sim]
        metrics['other_datasets'][name] = entry

    if args.items.is_file():   # is the text model worse where the check rejects?
        by_id = {e.id: e for e in test}
        sim_by_id = dict(zip([e.id for e in test], test_sim))
        rows = [r for r in csv.DictReader(open(args.items, encoding='utf-8')) if r['id'] in by_id]
        y = np.array([int(r['y_real']) for r in rows]); p = np.array([float(r['p_real']) for r in rows])
        s = np.array([sim_by_id[r['id']] for r in rows]); wrong = ((p >= 0.5) != (y == 1))
        low = s < threshold
        metrics['text_model_error_rate'] = {'n': len(rows), 'rejected_n': int(low.sum()),
                                           'error_when_rejected': float(wrong[low].mean()) if low.any() else None,
                                           'error_when_kept': float(wrong[~low].mean())}
        order = np.argsort(s)
        q = len(order) // 5
        metrics['text_model_error_rate']['lowest_20pct_similarity'] = float(wrong[order[:q]].mean())
        metrics['text_model_error_rate']['highest_80pct_similarity'] = float(wrong[order[q:]].mean())
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run = args.results / stamp
    run.mkdir(parents=True, exist_ok=True)
    (run / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print('Wrote', args.out, 'and', run)


if __name__ == '__main__':
    main()
