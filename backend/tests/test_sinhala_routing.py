"""Sinhala input is scored by the Sinhala model, and English-only parts are skipped for it."""
import numpy as np

from app.services import assessment, model_adapter as ma
from app.services.model_adapter import ModelOutput
from research import eval_sinhala as ev
from tests.test_eval_sinhala import corpus

SINHALA = 'ශ්‍රී ලංකාවේ ආර්ථිකය පසුගිය වසරේ සියයට දහයකින් වර්ධනය වූ බව රජය පවසයි.'


def settings(**kw):
    base = {'model_mode': 'model', 'sinhala_model_path': '', 'stance_model': '', 'trust_model_path': ''}
    return type('S', (), {**base, **kw})()


def test_is_sinhala_uses_share_of_letters():
    assert ma.is_sinhala(SINHALA)
    assert not ma.is_sinhala('The economy grew by ten percent last year, the government says.')
    assert not ma.is_sinhala('Officials said ශ්‍රී ලංකා grew strongly in the last quarter of the year.')
    assert not ma.is_sinhala('2024 — 10%')


def test_sinhala_text_goes_to_the_sinhala_model(monkeypatch):
    class Stub:
        def predict_proba(self, texts):
            return np.array([[0.8, 0.2]])
    monkeypatch.setattr(ma, 'get_settings', lambda: settings(sinhala_model_path='dir'))
    monkeypatch.setattr(ma, '_sinhala_bundle', lambda: (Stub(), 1))
    monkeypatch.setattr(ma, '_text_probs', lambda texts: (_ for _ in ()).throw(AssertionError('English model used')))
    out = ma.run_models(SINHALA, has_image=True, image_bytes=b'x')
    assert out.mode == 'model-sinhala' and out.text_probability_real == out.fused_probability_real == 0.8
    assert out.image_text_alignment is None and out.explanation_tokens == [] and len(out.notes) == 2


def test_without_sinhala_model_the_score_is_flagged(monkeypatch):
    monkeypatch.setattr(ma, 'get_settings', lambda: settings())
    monkeypatch.setattr(ma, '_text_probs', lambda texts: [0.6])
    monkeypatch.setattr(ma, '_shap_tokens', lambda text: [])
    out = ma.run_models(SINHALA, has_image=False)
    assert out.mode == 'model' and 'SINHALA_MODEL_PATH is not set' in out.notes[0]


def test_english_text_is_unchanged(monkeypatch):
    monkeypatch.setattr(ma, 'get_settings', lambda: settings(sinhala_model_path='dir'))
    monkeypatch.setattr(ma, '_text_probs', lambda texts: [0.6])
    monkeypatch.setattr(ma, '_shap_tokens', lambda text: [])
    out = ma.run_models('The economy grew by ten percent last year, the government says.', has_image=False)
    assert out.mode == 'model' and out.notes == []


def test_assessment_skips_english_only_parts_for_sinhala(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings(stance_model='nli', trust_model_path='dir'))
    monkeypatch.setattr(assessment, 'classify_stance', lambda c, e: (_ for _ in ()).throw(AssertionError('NLI ran')))
    model = ModelOutput(0.8, None, 0.8, 'model-sinhala', [])
    evidence = [{'title': 't', 'snippet': 's'}]
    kept, out = assessment.assess('claim', evidence, model)
    assert kept is evidence and out['trust'] is None and out['evidence_score'] is None and len(out['notes']) == 1


def test_exported_model_loads_in_the_app(tmp_path, monkeypatch):
    ev.export_model(corpus(n=200), tmp_path)
    monkeypatch.setattr(ma, 'get_settings', lambda: settings(sinhala_model_path=str(tmp_path)))
    ma._sinhala_bundle.cache_clear()
    try:
        lie, truth = ma._sinhala_p_real('බොරු පුවත 5'), ma._sinhala_p_real('සත්‍ය පුවත 5')
    finally:
        ma._sinhala_bundle.cache_clear()
    assert 0 <= lie < truth <= 1
