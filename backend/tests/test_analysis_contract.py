"""API contract for POST /api/analyses after wiring C1-C3 (app/services/assessment.py).
create_analysis is called directly; MongoDB, evidence search and the LLM are faked."""
import asyncio

import pytest

from app import main
from app.services import assessment
from app.services.trust import TrustModel

TEXT = 'Officials said the new bridge in the capital collapsed during heavy rain on Monday.'
EVIDENCE = [{'title': 'Bridge stands', 'snippet': 'The bridge did not collapse, engineers said.', 'url': 'https://example.org',
             'publisher': 'example.org', 'source_type': 'web_search', 'relevance': 0.6}]
OLD_FIELDS = {'id', 'created_at', 'title', 'text', 'claim', 'url', 'source_domain', 'image_data', 'image_metadata',
              'metadata', 'prediction', 'trust_score', 'model_simulated', 'model', 'signals', 'evidence', 'providers',
              'explanation', 'explanation_source', 'notes', 'tags'}
NEW_FIELDS = {'evidence_score', 'trust', 'explanation_audit', 'assessment_notes'}


@pytest.fixture
def fake_backend(monkeypatch):
    stored = []

    class Collection:
        async def insert_one(self, record):
            stored.append(record)

    async def gather(claim):
        return [dict(e) for e in EVIDENCE], {'tavily': 'ok'}

    async def explain(claim, evidence):
        return 'explanation', 'local_fallback'
    monkeypatch.setattr(main, 'database', lambda: type('DB', (), {'analyses': Collection()})())
    monkeypatch.setattr(main, 'gather_evidence', gather)
    monkeypatch.setattr(main, 'explain', explain)
    return stored


def analyse():
    return asyncio.run(main.create_analysis(text=TEXT, url='', image=None, user={'_id': 'user-1'}))


def set_settings(monkeypatch, stance='', trust=''):
    monkeypatch.setattr(assessment, 'get_settings', lambda: type('S', (), {'stance_model': stance, 'trust_model_path': trust})())


def test_unconfigured_response_is_old_contract_plus_empty_fields(fake_backend, monkeypatch):
    set_settings(monkeypatch)
    result = analyse()
    assert set(result) == OLD_FIELDS | NEW_FIELDS
    assert result['evidence_score'] is None and result['trust'] is None and result['explanation_audit'] is None
    assert result['assessment_notes'] == [] and 'stance' not in result['evidence'][0]
    assert result['trust_score'] == round(result['model']['fused_probability_real'] * 100)   # raw score kept
    assert result['model_simulated'] is True


def test_stance_runs_but_simulated_scores_are_never_calibrated(fake_backend, monkeypatch):
    set_settings(monkeypatch, stance='nli', trust='dir')

    def classify(claim, evidence):
        return [dict(e, stance={'label': 'refutes', 'p_support': 0.05, 'p_refute': 0.9, 'p_unrelated': 0.05,
                                'source': 'nli'}) for e in evidence]
    monkeypatch.setattr(assessment, 'classify_stance', classify)
    monkeypatch.setattr(assessment, 'load_trust_model', lambda: TrustModel(weights={'evidence_value': 3.0}, bias=0.0))
    result = analyse()
    assert result['evidence'][0]['stance']['label'] == 'refutes'
    assert result['evidence_score'] == pytest.approx(-0.85)
    assert result['trust'] is None                                   # MODEL_MODE is simulation
    assert fake_backend[0]['evidence_score'] == result['evidence_score']   # stored as returned


def test_calibrated_score_replaces_raw_score_with_real_models(fake_backend, monkeypatch):
    set_settings(monkeypatch, stance='nli', trust='dir')
    monkeypatch.setattr(assessment, 'classify_stance', lambda c, ev: [
        dict(e, stance={'label': 'refutes', 'p_support': 0.0, 'p_refute': 1.0, 'p_unrelated': 0.0, 'source': 'nli'})
        for e in ev])
    monkeypatch.setattr(assessment, 'load_trust_model', lambda: TrustModel(weights={'evidence_value': 3.0}, bias=0.0,
                                                                            alpha=0.1, qhat=0.5))
    real_output = main.run_models(TEXT, False)
    real_output.mode = 'model'
    monkeypatch.setattr(main, 'run_models', lambda *args: real_output)
    result = analyse()
    assert result['trust']['decision'] == 'Potentially misleading'
    assert result['trust_score'] == result['trust']['score'] == round(100 / (1 + 2.718281828 ** 3))
    assert result['prediction'] == 'Potentially misleading'
    assert [p['signal'] for p in result['trust']['breakdown']] == ['text', 'fused', 'evidence']
