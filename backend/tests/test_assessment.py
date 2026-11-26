"""Tests for app/services/assessment.py: each research part runs only when configured."""
import pytest

from app.services import assessment
from app.services.model_adapter import ModelOutput
from app.services.trust import TrustModel


def settings(stance='', trust=''):
    return type('S', (), {'stance_model': stance, 'trust_model_path': trust})()


def output(mode='model', image=False, tokens=()):
    return ModelOutput(text_probability_real=0.7, image_text_alignment=0.4 if image else None,
                       fused_probability_real=0.6, mode=mode, explanation_tokens=list(tokens))


EVIDENCE = [{'title': 'Jobs report', 'snippet': 'Jobs rose sharply last year.', 'relevance': 0.5}]


@pytest.fixture
def fake_stance(monkeypatch):
    def classify(claim, evidence):
        return [dict(e, stance={'label': 'supports', 'p_support': 0.9, 'p_refute': 0.05, 'p_unrelated': 0.05,
                                'source': 'nli'}) for e in evidence]
    monkeypatch.setattr(assessment, 'classify_stance', classify)


def test_nothing_configured_changes_nothing(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings())
    evidence, out = assessment.assess('claim', EVIDENCE, output())
    assert evidence is EVIDENCE
    assert out == {'evidence_score': None, 'trust': None, 'explanation_audit': None, 'notes': [],
                   'fact_check': None, 'out_of_domain': False, 'judge': None}


def test_stance_trust_and_eea_when_configured(monkeypatch, fake_stance):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings('nli-model', 'trust-dir'))
    captured = {}
    model = TrustModel(weights={'text_value': 1.0, 'evidence_value': 1.0}, bias=0.0, alpha=0.1, qhat=0.5)

    def load():
        return model
    monkeypatch.setattr(assessment, 'load_trust_model', load)
    real_trust = assessment.trust_score
    monkeypatch.setattr(assessment, 'trust_score', lambda s, m: captured.setdefault('signals', s) and real_trust(s, m))
    tokens = [{'token': 'jobs', 'value': 0.4}, {'token': 'rose', 'value': 0.2}]
    evidence, out = assessment.assess('Jobs rose', EVIDENCE, output(tokens=tokens))
    assert evidence[0]['stance']['label'] == 'supports' and 'stance' not in EVIDENCE[0]
    assert out['evidence_score'] == pytest.approx(0.85)
    assert captured['signals'] == {'text': 0.7, 'fused': None, 'evidence': out['evidence_score']}  # no image
    assert out['trust']['decision'] in ('Likely authentic', 'Needs review', 'Potentially misleading')
    assert out['explanation_audit']['eea'] == 1.0 and len(out['notes']) == 1 and 'No fact-checker' in out['notes'][0]


def test_simulated_models_never_get_a_calibrated_score(monkeypatch, fake_stance):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings('nli-model', 'trust-dir'))
    monkeypatch.setattr(assessment, 'load_trust_model', lambda: pytest.fail('must not load'))
    _, out = assessment.assess('claim', EVIDENCE, output(mode='simulated'))
    assert out['trust'] is None and out['evidence_score'] is not None


def test_failures_become_notes(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings('nli-model', 'missing-dir'))

    def boom(*args):
        raise RuntimeError('model download failed')
    monkeypatch.setattr(assessment, 'classify_stance', boom)
    monkeypatch.setattr(assessment, 'load_trust_model', boom)
    evidence, out = assessment.assess('claim', EVIDENCE, output(image=True))
    assert evidence is EVIDENCE and out['evidence_score'] is None and out['trust'] is None
    assert out['notes'][:2] == ['Evidence stance unavailable: RuntimeError', 'Calibrated trust score unavailable: RuntimeError']


def test_judge_with_no_answer_adds_a_note(monkeypatch):
    s = type('S', (), {'stance_model': '', 'trust_model_path': '', 'judge_enabled': True, 'judge_model': '',
                       'judge_variant': 'plausibility'})()
    monkeypatch.setattr(assessment, 'get_settings', lambda: s)

    async def silent(*args, **kwargs):
        return None
    monkeypatch.setattr(assessment, '_judge', silent)
    _, out = assessment.assess('claim', EVIDENCE, output())
    assert out['judge'] is None
    assert any('gave no answer' in n for n in out['notes'])
