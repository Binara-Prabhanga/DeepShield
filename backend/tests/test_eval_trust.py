"""Tests for research/eval_trust.py metrics and the end-to-end run on synthetic signals."""
import random

import pytest

from app.services import trust
from research import eval_trust as ev


def test_ece_by_hand():
    # confidences 0.9, 0.9 (bin 13), 0.6, 0.6 (bin 9); accuracy 1/2 and 1/2
    p, y = [0.9, 0.1, 0.6, 0.4], [1, 1, 1, 1]
    assert ev.ece(p, y) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert ev.ece([1.0, 0.0], [1, 0]) == 0.0


def test_brier_log_loss_f1():
    assert ev.brier([0.8, 0.3], [1, 0]) == pytest.approx((0.04 + 0.09) / 2)
    assert ev.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(0.6931, abs=1e-4)
    assert ev.macro_f1([0.9, 0.2, 0.7, 0.1], [1, 0, 1, 0]) == 1.0


def test_risk_coverage_orders_by_confidence():
    p, y = [0.95, 0.9, 0.6, 0.55], [1, 1, 0, 1]                      # the 0.6 item is wrong
    rc = ev.risk_coverage(p, y)
    assert rc['error_at_100'] == 0.25
    assert rc['curve'][9]['error'] == 0.0                            # 50% coverage: two right
    assert rc['aurc'] == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4)


def test_conformal_metrics_counts():
    m = ev.conformal_metrics([0.9, 0.6, 0.2], [1, 0, 0], qhat=0.65)
    # sets: {real}, {real, misleading}, {misleading}
    assert m['coverage'] == 1.0 and m['abstain_rate'] == pytest.approx(1 / 3)
    assert m['error_on_answered'] == 0.0 and m['empty_sets'] == 0


def test_bootstrap_repeatable_and_paired_zero():
    y, p = [1, 0] * 30, [0.7, 0.4] * 30
    a = ev.bootstrap(ev.brier, y, p, n=100, seed=1)
    assert a == ev.bootstrap(ev.brier, y, p, n=100, seed=1)
    assert a['low'] <= a['value'] <= a['high']
    assert ev.bootstrap(ev.brier, y, p, p, n=50)['high'] == 0.0


def synthetic_rows(n_val=400, n_test=400, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n_val + n_test):
        label = rng.randrange(6)
        y = int(label >= 2)
        evidence = None if rng.random() < 0.05 else max(-1.0, min(1.0, rng.gauss(0.4 if y else -0.4, 0.5)))
        rows.append({'id': f'liar2-{i}', 'split': 'val' if i < n_val else 'test', 'liar2_label': label,
                     'y_real': y, 'text': trust.sigmoid(rng.gauss(0.3 if y else -0.3, 1)), 'fused': None,
                     'evidence': evidence})
    return rows


def test_run_end_to_end_on_synthetic_signals():
    metrics, model, predictions = ev.run(synthetic_rows(), seeds=(1, 2), alphas=(0.1,), n_bootstrap=50)
    summary = metrics['summary_over_seeds']
    assert set(summary) == {'evidence_raw', 'text_raw', 'average_raw', 'evidence_cal', 'text_cal', 'trust', 'trust_no_T'}
    assert summary['trust']['brier']['mean'] < summary['text_raw']['brier']['mean']
    assert summary['trust']['conformal']['0.1']['coverage']['mean'] > 0.8
    assert metrics['n']['val_fit'] + metrics['n']['val_cal'] == 400
    assert 'trust_minus_evidence_cal' in metrics['bootstrap_seed42']
    assert model.qhat is not None and model.alpha == 0.1
    assert len(predictions['trust']) == len(predictions['id']) == 400
    assert set(metrics['trust_p_real_by_liar2_label_seed42']) <= set(range(6))


def test_run_without_text_signal():
    rows = [dict(r, text=None) for r in synthetic_rows(200, 200, seed=3)]
    metrics, _, _ = ev.run(rows, seeds=(1,), alphas=(0.2,), n_bootstrap=20)
    assert set(metrics['summary_over_seeds']) == {'evidence_raw', 'evidence_cal', 'trust', 'trust_no_T'}


def test_build_signals_joins_and_skips_unlabelled():
    from research.data import Example
    val = [Example(id='liar2-1', text='a', label=0, label_name='pants_fire', binary_label=1, split='val'),
           Example(id='liar2-2', text='b', label=3, label_name='half_true', binary_label=None, split='val')]
    rows = ev.build_signals(val, [], {'liar2-1': -0.8}, {'liar2-1': 0.3})
    assert rows == [{'id': 'liar2-1', 'split': 'val', 'liar2_label': 0, 'y_real': 0, 'text': 0.3,
                     'fused': None, 'evidence': -0.8}]
