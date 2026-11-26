"""Unit tests for app/services/trust.py (C2 Trust Score)."""
import math
import random

import pytest

from app.services import trust


def test_features_logit_probabilities_and_flag_missing():
    row = trust.features({'text': 0.5, 'fused': None, 'evidence': -0.4})
    assert row == pytest.approx([0.0, 0.0, 0.0, 1.0, -0.4, 0.0])
    assert trust.features({}) == [0.0, 1.0] * 3
    assert trust.features({'text': float('nan')})[:2] == [0.0, 1.0]
    assert math.isfinite(trust.features({'text': 1.0})[0])            # clipped, not infinite


def test_model_margin_probability_and_temperature():
    model = trust.TrustModel(weights={'text_value': 2.0, 'evidence_value': 1.0, 'fused_missing': -0.5}, bias=0.25)
    signals = {'text': trust.sigmoid(0.5), 'evidence': 0.3}           # logit(text) = 0.5
    z = 2.0 * 0.5 + 1.0 * 0.3 - 0.5 + 0.25
    assert model.margin(signals) == pytest.approx(z)
    model.temperature = 2.0
    assert model.probability(signals) == pytest.approx(trust.sigmoid(z / 2))


def test_trust_model_round_trip_and_rejects_unknown_feature(tmp_path):
    model = trust.TrustModel(weights={'text_value': 1.0}, bias=0.1, temperature=1.5, alpha=0.1, qhat=0.4)
    model.save(tmp_path)
    assert trust.load_trust_model(tmp_path) == model
    with pytest.raises(ValueError):
        trust.TrustModel.from_dict({'weights': {'source_value': 1.0}, 'bias': 0.0})


def test_missing_setting_raises(monkeypatch):
    monkeypatch.setattr(trust, 'get_settings', lambda: type('S', (), {'trust_model_path': ''})())
    with pytest.raises(RuntimeError, match='TRUST_MODEL_PATH'):
        trust.load_trust_model()


def synthetic(n, seed=0):
    """Evidence is informative, text weakly so, fused always missing; 10% have no evidence."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        y = rng.random() < 0.5
        evidence = None if rng.random() < 0.1 else max(-1.0, min(1.0, rng.gauss(0.5 if y else -0.5, 0.4)))
        text = trust.sigmoid(rng.gauss(0.4 if y else -0.4, 1.0))
        rows.append({'text': text, 'fused': None, 'evidence': evidence})
        labels.append(int(y))
    return rows, labels


def test_fit_trust_learns_sensible_weights():
    rows, labels = synthetic(600)
    model = trust.fit_trust(rows, labels, seed=1)
    assert model.weights['evidence_value'] > model.weights['text_value'] > 0
    assert model.weights['fused_value'] == model.weights['fused_missing'] == 0.0   # never seen
    assert 0.2 < model.temperature < 5
    accuracy = sum((model.probability(r) >= 0.5) == bool(y) for r, y in zip(rows, labels)) / len(rows)
    assert accuracy > 0.8


def test_fit_temperature_recovers_overconfidence():
    rng = random.Random(3)
    true_margins = [rng.gauss(0, 2) for _ in range(3000)]
    labels = [int(rng.random() < trust.sigmoid(m)) for m in true_margins]
    t = trust.fit_temperature([3 * m for m in true_margins], labels)     # margins 3x too confident
    assert t == pytest.approx(3, rel=0.15)


def test_conformal_qhat_quantile_and_small_n():
    probs, labels = [0.9, 0.8, 0.3, 0.6], [1, 1, 0, 0]               # scores 0.1, 0.2, 0.3, 0.6
    assert trust.conformal_qhat(probs, labels, 0.5) == pytest.approx(0.3)   # ceil(5 * 0.5) = 3rd
    assert trust.conformal_qhat(probs, labels, 0.1) == 1.0                 # k = 5 > n
    with pytest.raises(ValueError):
        trust.conformal_qhat(probs, labels, 0)


def test_prediction_set_cases():
    assert trust.prediction_set(0.9, 0.3) == ('real',)
    assert trust.prediction_set(0.6, 0.65) == ('real', 'misleading')     # scores 0.4 and 0.6
    assert trust.prediction_set(0.55, 0.2) == ()
    assert trust.prediction_set(0.4, None) == ('misleading',)


def test_conformal_coverage_holds_on_fresh_data():
    rows, labels = synthetic(4000, seed=5)
    model = trust.fit_trust(rows[:1000], labels[:1000], seed=0)
    cal = [model.probability(r) for r in rows[1000:2000]]
    qhat = trust.conformal_qhat(cal, labels[1000:2000], 0.1)
    covered = [('real' if y else 'misleading') in trust.prediction_set(model.probability(r), qhat)
               for r, y in zip(rows[2000:], labels[2000:])]
    assert sum(covered) / len(covered) >= 0.88


def test_trust_score_breakdown_and_decisions():
    model = trust.TrustModel(weights={'text_value': 1.0, 'evidence_value': 2.0, 'evidence_missing': -0.4},
                             bias=0.0, temperature=2.0, alpha=0.1, qhat=0.3)
    result = trust.trust_score({'text': trust.sigmoid(1.0), 'evidence': 1.0}, model)
    assert result['probability_real'] == pytest.approx(trust.sigmoid(1.5), abs=1e-4)
    assert result['decision'] == 'Likely authentic' and not result['abstained']
    parts = {p['signal']: p for p in result['breakdown']}
    assert parts['text']['contribution'] == pytest.approx(0.5)
    assert parts['evidence']['contribution'] == pytest.approx(1.0)
    assert parts['fused'] == {'signal': 'fused', 'available': False, 'value': None, 'contribution': 0.0, 'share': 0.0}
    assert parts['evidence']['share'] == pytest.approx(2 / 3, abs=1e-4)

    unsure = trust.trust_score({'text': 0.5}, model)                # evidence missing -> -0.2
    assert unsure['decision'] == 'Needs review' and unsure['abstained']
    assert {p['signal']: p['available'] for p in unsure['breakdown']} == {'text': True, 'fused': False, 'evidence': False}
    assert trust.trust_score({'evidence': -1.0}, model)['decision'] == 'Potentially misleading'
