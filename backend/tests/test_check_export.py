"""Tests for research/check_export.py using a synthetic export and an injected loader."""
import csv
import json
import math

import pytest

from research import check_export as ce

GOLD = {'val': {f'liar2-{i}': i % 2 for i in range(10)}, 'test': {f'liar2-{i}': i % 2 for i in range(10, 20)}}
TEMPS = {'42': 1.5, '43': 1.2, '44': 1.0}


def loader(dataset, split):
    assert dataset == 'liar2'
    return GOLD[split]


def write_predictions(path, gold, temperature, mutate=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (rid, label) in enumerate(gold.items()):
        margin = (-2.0 if label else 2.0) + 0.1 * i
        p_real = 1 / (1 + math.exp(-margin / temperature))
        rows.append({'id': rid, 'label': label, 'misleading_probability': round(1 - p_real, 6), 'margin': margin})
    if mutate:
        rows = mutate(rows)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ce.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def make_export(root, mutate=None, manifest_changes=None):
    manifest = {'kind': 'text', 'base_model': 'bert-base-uncased', 'trained_on': 'liar2', 'seeds': [42, 43, 44],
                'weights_seed': 42, 'temperature_by_seed': TEMPS, 'data_loader_sha256': ce.data_loader_sha256(),
                'hyperparameters': {}, 'created': '2026-12-01', **(manifest_changes or {})}
    (root / 'model').mkdir(parents=True)
    (root / 'model' / 'config.json').write_text(json.dumps({'id2label': {'0': 'fake', '1': 'real'}}))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    for seed in (42, 43, 44):
        for split in ('val', 'test'):
            write_predictions(root / 'predictions' / f'seed{seed}' / f'liar2_{split}.csv', GOLD[split],
                              TEMPS[str(seed)], mutate if (seed, split) == (43, 'test') else None)
    return root


def test_valid_export_passes(tmp_path):
    errors, warnings = ce.check_export(make_export(tmp_path), loader)
    assert errors == [] and warnings == []


def test_read_export_probs_uses_weights_seed(tmp_path):
    probs = ce.read_export_probs(make_export(tmp_path), 'liar2', 'val')
    assert set(probs) == set(GOLD['val']) and all(0 < p < 1 for p in probs.values())
    assert probs['liar2-0'] > 0.5 > probs['liar2-1']                  # label 0 = real


@pytest.mark.parametrize('mutate, expected', [
    (lambda rows: [dict(r, label=1 - int(r['label'])) if i == 0 else r for i, r in enumerate(rows)], 'data.py says'),
    (lambda rows: rows + [dict(rows[0])], 'duplicate id'),
    (lambda rows: rows[:5], 'covers 50.0%'),
    (lambda rows: [dict(r, id='liar2-999') if i == 0 else r for i, r in enumerate(rows)], 'id not in this split'),
    (lambda rows: [dict(r, misleading_probability=1.5) if i == 0 else r for i, r in enumerate(rows)], 'outside [0, 1]'),
    (lambda rows: [dict(r, misleading_probability=round(1 - float(r['misleading_probability']), 6)) for r in rows],
     'sigmoid(margin'),                                               # flipped class
])
def test_bad_predictions_are_reported(tmp_path, mutate, expected):
    errors, _ = ce.check_export(make_export(tmp_path, mutate), loader)
    assert any(expected in e and 'seed43/liar2_test.csv' in e for e in errors), errors


def test_manifest_problems(tmp_path):
    errors, _ = ce.check_export(make_export(tmp_path, manifest_changes={'seeds': [42]}), loader)
    assert any('at least 3 seeds' in e for e in errors)
    (tmp_path / 'b').mkdir()
    errors, _ = ce.check_export(make_export(tmp_path / 'b', manifest_changes={'temperature_by_seed': {'42': 1}}), loader)
    assert any('temperature_by_seed["43"]' in e for e in errors)
    (tmp_path / 'c').mkdir()
    _, warnings = ce.check_export(make_export(tmp_path / 'c', manifest_changes={'data_loader_sha256': 'x'}), loader)
    assert any('data_loader_sha256' in w for w in warnings)
    assert ce.check_export(tmp_path / 'missing', loader)[0] == ['manifest.json: missing']


def test_model_dir_and_required_files(tmp_path):
    root = make_export(tmp_path)
    (root / 'model' / 'config.json').write_text(json.dumps({'id2label': {'0': 'LABEL_0', '1': 'LABEL_X'}}))
    (root / 'predictions' / 'seed44' / 'liar2_val.csv').unlink()
    (root / 'predictions' / 'seed44' / 'liar2_train.csv').write_text('id\n')
    errors, _ = ce.check_export(root, loader)
    assert any('no real-class name' in e for e in errors)
    assert any('seed44/liar2_val.csv: required file missing' in e for e in errors)
    assert any('liar2_train.csv: name must be' in e for e in errors)


def test_multimodal_model_dir(tmp_path):
    errors = []
    (tmp_path / 'fusion_config.json').write_text(json.dumps({'clip_model': 'x', 'hidden_dim': 512}))
    ce.check_model_dir(tmp_path, 'multimodal', errors)
    assert "model/fusion_config.json: missing 'real_index'" in errors and 'model/fusion_head.pt: missing' in errors


def test_temporal_export_requires_future_and_rejects_future_elsewhere(tmp_path):
    temporal_gold = {'val': {'liar2-1': 0}, 'test': {'liar2-2': 1}, 'future': {'liar2-3': 1}}

    def temporal_loader(dataset, split):
        return temporal_gold[split] if dataset == 'liar2_temporal' else GOLD[split]
    root = make_export(tmp_path, manifest_changes={'trained_on': 'liar2_temporal'})
    errors, _ = ce.check_export(root, temporal_loader)
    assert any('liar2_temporal_future.csv: required file missing' in e for e in errors)
    for seed in (42, 43, 44):
        for split, gold in temporal_gold.items():
            write_predictions(root / 'predictions' / f'seed{seed}' / f'liar2_temporal_{split}.csv', gold, TEMPS[str(seed)])
    errors, _ = ce.check_export(root, temporal_loader)
    assert errors == []
    (root / 'predictions' / 'seed42' / 'liar2_future.csv').write_text('id\n')
    errors, _ = ce.check_export(root, temporal_loader)
    assert any('liar2_future.csv: name must be' in e for e in errors)


def test_required_files_per_source_and_kind():
    req = ce.required_files
    assert req({'trained_on': 'liar2', 'kind': 'text'}) == ('liar2_val', 'liar2_test')
    assert req({'trained_on': 'fakeddit', 'kind': 'multimodal'}) == ('fakeddit_val', 'fakeddit_test')
    assert req({'trained_on': 'fakeddit_official', 'kind': 'multimodal'}) == ('fakeddit_official_val', 'fakeddit_official_test')
    assert req({'trained_on': 'fakeddit_official', 'kind': 'text'}) == (
        'fakeddit_official_val', 'fakeddit_official_test', 'liar2_val', 'liar2_test')
    assert 'liar2_val' not in req({'trained_on': 'liar2_temporal', 'kind': 'text'})
