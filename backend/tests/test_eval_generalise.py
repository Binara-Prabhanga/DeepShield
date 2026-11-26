"""Tests for research/eval_generalise.py with a fake loader (no real datasets needed)."""
import csv
import json
import random
from datetime import date

import pytest

from research import eval_generalise as ev
from research.data import Example


def make(prefix, n, seed, shift=0.0):
    """Examples whose text says 'good' more often when real; label share set by shift."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        real = rng.random() < 0.5 - shift
        words = ['good' if rng.random() < (0.8 if real else 0.2) else 'bad', 'news', 'today']
        rows.append(Example(id=f'{prefix}-{i}', text=' '.join(words), label=0, label_name='', split='test',
                            binary_label=0 if real else 1, date=date(2015 + i % 5, 1, 1),
                            meta={'speaker': f's{i % 7}', 'context': 'tweet', 'subject': '', 'state_info': ''}))
    return rows


FAKE = {('liar2', 'train'): make('tr', 400, 1), ('liar2', 'test'): make('te', 300, 2),
        ('liar_unseen', 'test'): make('lu', 200, 3, shift=0.2), ('fakeddit', 'test'): make('fk', 200, 4)}


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(ev, 'load_split', lambda dataset, split: FAKE[(dataset, split)])
    monkeypatch.setattr(ev, 'local_set_available', lambda: False)


def test_target_metrics_and_bootstrap_drop():
    m = ev.target_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert m['macro_f1'] == 1.0 and m['auroc'] == 1.0 and m['misleading_share'] == 0.5 and m['n'] == 4
    assert ev.target_metrics([0.9, 0.8], [1, 1])['auroc'] is None
    drop = ev.bootstrap_drop([0.9, 0.1] * 20, [1, 0] * 20, [0.9, 0.1] * 20, [0, 1] * 20, n=100)
    assert drop['value'] == 1.0 and drop['low'] <= drop['value'] <= drop['high']


def test_mean_sd_handles_numbers_none_and_text():
    out = ev.mean_sd([{'a': 1.0, 'b': None, 'kind': 'x'}, {'a': 3.0, 'b': None, 'kind': 'x'}])
    assert out['a'] == {'mean': 2.0, 'sd': pytest.approx(1.414, abs=1e-3)} and out['b'] is None and out['kind'] == 'x'


def test_surface_features():
    f = ev.surface_features('Says "90 percent" of US jobs are gone!')
    assert f[7] == 1.0 and f[8] == 1.0 and f[3] == 2 and f[4] == 1 and f[2] == 2


def test_probes_learn_the_shortcut_they_see():
    train, test = FAKE[('liar2', 'train')], FAKE[('liar2', 'test')]
    for kind in ('surface', 'metadata', 'year'):
        m = ev.probe(kind, train, test)
        assert 0 <= m['macro_f1'] <= 1 and m['n'] == len(test)
    # make the speaker decide the label: the metadata probe must find it
    leaky = [Example(**{**e.__dict__, 'meta': {'speaker': f'liar{e.binary_label}'}}) for e in train + test]
    assert ev.probe('metadata', leaky[:400], leaky[400:])['macro_f1'] == 1.0


def test_placeholder_run_produces_protocol_table(fake_loader):
    metrics = ev.run(None, n_bootstrap=50, sources=('liar2',))
    table = metrics['models']['liar2 (tfidf placeholder)']
    assert [table[t]['kind'] for t in table] == ['in-dataset', 'cross-dataset', 'cross-dataset']
    assert table['liar2/test']['macro_f1']['mean'] > 0.7
    assert 'macro_f1_drop' in table['liar_unseen/test'] and 'macro_f1_drop' not in table['liar2/test']
    assert set(metrics['shortcut_probes']['liar2']) == {'surface', 'metadata', 'year'}


def write_export(root, source, files, seeds=(42, 43, 44)):
    (root / 'predictions').mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps({'trained_on': source, 'seeds': list(seeds), 'weights_seed': seeds[0]}))
    for seed in seeds:
        folder = root / 'predictions' / f'seed{seed}'
        folder.mkdir()
        for (dataset, split), rows in files.items():
            with open(folder / f'{dataset}_{split}.csv', 'w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'label', 'misleading_probability', 'margin'])
                for e in rows:
                    p_mis = 0.2 if 'good' in e.text else 0.8
                    writer.writerow([e.id, e.binary_label, p_mis + (seed - 42) * 0.01, 0.0])


def test_export_run_averages_over_seeds(tmp_path, fake_loader):
    write_export(tmp_path / 'bert', 'liar2', {k: FAKE[k] for k in (('liar2', 'test'), ('liar_unseen', 'test'), ('fakeddit', 'test'))})
    metrics = ev.run([tmp_path / 'bert'], n_bootstrap=30, sources=('liar2',))
    table = metrics['models']['liar2 (bert)']
    assert set(table) == {'liar2/test', 'liar_unseen/test', 'fakeddit/test'}
    assert table['liar2/test']['macro_f1']['sd'] >= 0 and table['liar2/test']['n'] == 300
    assert set(table['fakeddit/test']['macro_f1_drop']) == {'value', 'low', 'high'}
    ev.print_table(metrics)                                         # seed-averaged drops print without error


def test_target_added_for_weights_seed_only(tmp_path, fake_loader):
    write_export(tmp_path / 'bert', 'liar2', {k: FAKE[k] for k in (('liar2', 'test'), ('liar_unseen', 'test'))})
    extra = tmp_path / 'bert' / 'predictions'
    (extra / 'seed42' / 'fakeddit_test.csv').write_text(
        (extra / 'seed42' / 'liar_unseen_test.csv').read_text(encoding='utf-8').replace('liar_unseen', 'fakeddit'),
        encoding='utf-8')
    _, preds = ev.export_predictions(tmp_path / 'bert', [('liar2', 'test'), ('fakeddit', 'test')])
    assert len(preds[('liar2', 'test')]) == 3 and len(preds[('fakeddit', 'test')]) == 1


def test_export_without_in_dataset_predictions_fails(tmp_path, fake_loader):
    write_export(tmp_path / 'bad', 'liar2', {('fakeddit', 'test'): FAKE[('fakeddit', 'test')]})
    with pytest.raises(ValueError, match='in-dataset predictions'):
        ev.run([tmp_path / 'bad'], n_bootstrap=10, sources=('liar2',))
