"""Pure unit tests. Run: cd backend && python -m pytest -q"""
import pytest
from app.services.article import valid_public_url
from app.services.model_adapter import run_models
from app.services.evidence import rank_evidence


def test_url_rejection():
    assert not valid_public_url('http://localhost/private')
    assert not valid_public_url('http://127.0.0.1/foo')
    assert not valid_public_url('http://192.168.1.1/')
    assert not valid_public_url('file:///etc/passwd')
    assert valid_public_url('https://example.org/news')


def test_simulated_models_explicit_and_bounded():
    result = run_models('A sample story headline written for a local UI test', True)
    assert result.mode == 'simulated'
    assert 0 <= result.text_probability_real <= 1
    assert result.image_text_alignment is not None
    assert 0 <= result.fused_probability_real <= 1
    assert result.gradcam_available is False
    assert result.shap_available is False


def test_retrieval_order_and_real_urls():
    docs = [
        {'title':'Space weather report', 'snippet':'Solar winds and particles', 'url':'https://example.org/one', 'source_type':'web_search', 'publisher':'example.org'},
        {'title':'Ocean temperatures', 'snippet':'Ocean temperatures are increasing', 'url':'https://example.org/two', 'source_type':'fact_check', 'publisher':'example.org'},
    ]
    res = rank_evidence('ocean temperatures increasing', docs)
    assert len(res) == 2
    assert res[0]['source_type'] == 'fact_check'
    assert all('relevance' in item for item in res)
