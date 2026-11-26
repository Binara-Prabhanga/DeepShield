"""Tests for research/train_text.py and research/train_fusion.py helpers (no model download)."""
import csv
import math

import numpy as np
import pytest

from research import check_export, train_fusion, train_text
from research.data import Example


def test_text_targets_cover_required_export_files(monkeypatch):
    monkeypatch.setattr(train_text, 'local_set_available', lambda: False)
    for source in ('liar2', 'liar2_temporal', 'fakeddit', 'fakeddit_official'):
        names = {f'{d}_{s}' for d, s in train_text.targets_for(source)}
        required = check_export.required_files({'trained_on': source, 'kind': 'text'})
        assert set(required) <= names and not any(n.endswith('_train') for n in names)
    assert ('liar2_temporal', 'future') in train_text.targets_for('liar2_temporal')
    monkeypatch.setattr(train_text, 'local_set_available', lambda: True)
    assert ('srilanka', 'test') in train_text.targets_for('liar2')


def read(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_text_predictions_satisfy_checker_formula(tmp_path):
    rows = [Example(id=f'x{i}', text='t', label=0, label_name='', binary_label=i % 2, split='test') for i in range(4)]
    logits = np.array([[0.0, 2.0], [1.0, -1.0], [0.5, 0.5], [-3.0, 3.0]], dtype=np.float32)
    train_text.write_predictions(tmp_path / 'p.csv', rows, logits, temperature=1.5)
    out = read(tmp_path / 'p.csv')
    for row, (fake, real) in zip(out, logits):
        assert float(row['margin']) == pytest.approx(real - fake)
        assert 1 - float(row['misleading_probability']) == pytest.approx(1 / (1 + math.exp(-(real - fake) / 1.5)), abs=1e-5)
    assert [int(r['label']) for r in out] == [0, 1, 0, 1]


def synthetic_embeddings(n, seed, dim=16):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    text = rng.normal(size=(n, dim)) + y[:, None] * 1.5
    image = rng.normal(size=(n, dim))
    norm = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)
    return {'ids': np.array([f'f{seed}-{i}' for i in range(n)]), 'labels_real': y, 'text': norm(text), 'image': norm(image)}


def test_fusion_head_learns_and_is_reproducible(tmp_path):
    train, val = synthetic_embeddings(400, 1), synthetic_embeddings(200, 2)
    head = train_fusion.train_head(train, val, seed=7, hidden=32, dropout=0.0, epochs=30)
    margin = train_fusion.margins(head, val)
    assert ((margin > 0).astype(int) == val['labels_real']).mean() > 0.85
    again = train_fusion.margins(train_fusion.train_head(train, val, seed=7, hidden=32, dropout=0.0, epochs=30), val)
    assert np.allclose(margin, again)
    train_fusion.write_predictions(tmp_path / 'f.csv', val, margin, 2.0)
    out = read(tmp_path / 'f.csv')
    assert [int(r['label']) for r in out] == [int(1 - y) for y in val['labels_real']]   # 1 = misleading
    assert 1 - float(out[0]['misleading_probability']) == pytest.approx(1 / (1 + math.exp(-margin[0] / 2)), abs=1e-5)
