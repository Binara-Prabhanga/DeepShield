"""Tests for research/eval_sinhala.py on synthetic rows (no model download)."""
import random

import numpy as np

from research import eval_sinhala as ev
from research.data import Example

LABELS = {'CREDIBLE': 0, 'UNCERTAIN': None, 'PARTIAL': 1, 'FALSE': 1}


def corpus(n=300, seed=0, leak_domain=False):
    """Misleading rows contain 'බොරු'; with leak_domain the outlet also gives the label away."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        name = rng.choice(['CREDIBLE'] * 5 + ['UNCERTAIN'] * 3 + ['FALSE', 'PARTIAL'])
        word = 'බොරු' if LABELS[name] == 1 else ('සමහර' if name == 'UNCERTAIN' else 'සත්‍ය')
        domain = (f'{name.lower()}{rng.randrange(5)}.lk' if leak_domain else f'site{rng.randrange(10)}.lk')
        rows.append(Example(id=f's{i}', text=f'{word} පුවත {rng.randrange(1000)}', label=0, label_name=name,
                            binary_label=LABELS[name], split='all', meta={'domain': domain}))
    return rows


def test_task_rows():
    rows = corpus()
    binary, y = ev.task_rows(rows, 'binary')
    assert all(e.label_name != 'UNCERTAIN' for e in binary) and set(y) == {0, 1}
    three, y3 = ev.task_rows(rows, '3class')
    assert len(three) == len(rows) and set(y3) == {0, 1, 2}
    assert all(ev.CLASSES_3[c] == 'MISLEADING' for e, c in zip(three, y3) if e.label_name in ('FALSE', 'PARTIAL'))


def test_grouped_folds_hold_out_whole_domains():
    rows, y = ev.task_rows(corpus(), 'binary')
    groups = [e.meta['domain'] for e in rows]
    for train, test in ev.folds(y, groups, seed=42, grouped=True):
        assert not {groups[i] for i in train} & {groups[i] for i in test}
    again = ev.folds(y, groups, seed=42, grouped=False)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(again, ev.folds(y, groups, seed=42, grouped=False)))


def test_run_learns_text_signal_and_flags_domain_shortcut():
    metrics, oof = ev.run(corpus(leak_domain=True), None, models=('domain', 'char_tfidf'), seeds=(42, 43), n_bootstrap=50)
    random_b = metrics['binary']['random']
    assert random_b['char_tfidf']['auroc']['mean'] > 0.95
    assert random_b['domain']['auroc']['mean'] > 0.95                     # outlet gives the label away
    assert 'domain' not in metrics['binary']['grouped']                  # probe is undefined on unseen outlets
    assert metrics['binary']['grouped']['char_tfidf']['auroc']['mean'] > 0.9
    assert metrics['3class']['random']['char_tfidf']['macro_f1']['mean'] > 0.9
    assert oof['binary']['random']['char_tfidf'].shape == (metrics['binary']['n'], 2)
    ci = random_b['char_tfidf']['ci_seed42']
    assert ci['metric'] == 'auroc' and ci['low'] <= ci['high']


def test_labse_model_uses_given_embeddings():
    rows = corpus()
    emb = np.array([[1.0, 0.0] if e.binary_label == 1 else [0.0, 1.0] for e in rows]) + np.random.default_rng(0).normal(0, 0.1, (len(rows), 2))
    metrics, _ = ev.run(rows, emb, models=('labse',), seeds=(42,), n_bootstrap=20)
    assert metrics['binary']['random']['labse']['auroc']['mean'] > 0.95
