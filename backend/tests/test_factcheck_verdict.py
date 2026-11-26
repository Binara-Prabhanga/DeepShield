"""Fact-check ratings decide the result; claims unlike the training data are sent to review."""
import numpy as np
import pytest

from app.services import assessment, domain
from app.services.factcheck_verdict import fact_check_verdict, rating_verdict
from app.services.model_adapter import ModelOutput

CLAIM = 'Unemployment doubled under Governor Walker in Wisconsin'


def review(rating, snippet=CLAIM, publisher='AP', source_type='fact_check'):
    return {'title': 't', 'url': 'https://x.org/' + rating.replace(' ', ''), 'publisher': publisher, 'snippet': snippet,
            'rating': rating, 'source_type': source_type}


@pytest.mark.parametrize('rating,expected', [
    ('False', 'false'), ('Mostly False', 'false'), ('Pants on Fire', 'false'), ('Not true', 'false'),
    ('Misleading', 'false'), ('True', 'true'), ('Mostly True', 'true'), ('Correct', 'true'),
    ('Half True', 'mixed'), ('Partly false', 'mixed'), ('Needs context', 'mixed'), ('', None), ('Unrated', None)])
def test_rating_verdict(rating, expected):
    assert rating_verdict(rating) == expected


def test_same_claim_false_rating_gives_a_low_score():
    fc = fact_check_verdict(CLAIM, [review('Mostly False')])
    assert fc['verdict'] == 'false' and fc['score'] == 8 and fc['publisher'] == 'AP'


def test_same_claim_true_rating_gives_a_high_score():
    assert fact_check_verdict(CLAIM, [review('True')])['score'] == 92


def test_other_claims_mixed_ratings_and_disagreement_do_not_override():
    other = review('False', snippet='The moon landing was staged in a film studio in Nevada')
    assert fact_check_verdict(CLAIM, [other]) is None
    assert fact_check_verdict(CLAIM, [review('Half True')]) is None
    assert fact_check_verdict(CLAIM, [review('True'), review('False', publisher='Snopes')]) is None
    assert fact_check_verdict(CLAIM, [review('False', source_type='web_search')]) is None


def settings(**kw):
    base = {'model_mode': 'model', 'stance_model': '', 'trust_model_path': '', 'domain_model_path': ''}
    return type('S', (), {**base, **kw})()


MODEL = ModelOutput(0.2, None, 0.2, 'model', [])


def test_assess_reports_the_fact_check_and_a_note(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings())
    _, out = assessment.assess(CLAIM, [review('False')], MODEL)
    assert out['fact_check']['score'] == 8 and not out['out_of_domain']
    assert 'Fact-checked by AP' in out['notes'][0]


def test_out_of_domain_claim_is_flagged_only_without_a_fact_check(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings())
    monkeypatch.setattr(assessment, 'out_of_domain', lambda claim: (True, 0.31))
    _, out = assessment.assess('Water boils at 100 degrees Celsius.', [], MODEL)
    assert out['out_of_domain'] and out['fact_check'] is None and 'not reliable' in out['notes'][0]
    _, out = assessment.assess(CLAIM, [review('False')], MODEL)
    assert not out['out_of_domain'] and out['fact_check']


def test_simulated_scores_are_never_declared_out_of_domain(monkeypatch):
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings())
    monkeypatch.setattr(assessment, 'out_of_domain', lambda claim: (True, 0.1))
    _, out = assessment.assess('anything', [], ModelOutput(0.5, None, 0.5, 'simulated', []))
    assert not out['out_of_domain']


def test_domain_similarity_threshold(monkeypatch):
    bundle = {'model': 'stub', 'matrix': np.array([[1, 0], [0, 1]], dtype=np.float32), 'threshold': 0.5}
    monkeypatch.setattr(domain, 'get_settings', lambda: settings(domain_model_path='dir'))
    monkeypatch.setattr(domain, '_bundle', lambda path: bundle)
    from app.services import evidence
    monkeypatch.setattr(evidence, 'embed_texts', lambda texts, name: np.array([[0.6, 0.8]], dtype=np.float32))
    assert domain.out_of_domain('x') == (False, 0.8)
    monkeypatch.setattr(evidence, 'embed_texts', lambda texts, name: np.array([[0.707, 0.707]], dtype=np.float32))
    assert domain.out_of_domain('x')[0] is True or domain.out_of_domain('x')[1] >= 0.5


def test_domain_check_never_fails_the_analysis(monkeypatch):
    monkeypatch.setattr(domain, 'get_settings', lambda: settings(domain_model_path='missing'))
    monkeypatch.setattr(domain, '_bundle', lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    assert domain.out_of_domain('x') == (False, None)
    monkeypatch.setattr(domain, 'get_settings', lambda: settings())
    assert domain.out_of_domain('x') == (False, None)


# ---- final decision rules (app.main.decide) ----
from app.main import decide


def A(**kw):
    return {'trust': None, 'fact_check': None, 'judge': None, 'out_of_domain': False, **kw}


def test_decide_default_uses_the_model_score():
    assert decide(80, A()) == (80, 'Likely authentic')
    assert decide(13, A()) == (13, 'Potentially misleading')


def test_decide_priority_fact_check_over_judge_over_out_of_domain():
    assert decide(90, A(fact_check={'score': 8}, judge={'verdict': 'supported'})) == (8, 'Potentially misleading')
    assert decide(90, A(judge={'verdict': 'refuted'}, out_of_domain=True)) == (20, 'Potentially misleading')
    assert decide(13, A(judge={'verdict': 'supported'})) == (50, 'Needs review')
    assert decide(70, A(judge={'verdict': 'supported'})) == (70, 'Likely authentic')   # agreement: no change
    assert decide(70, A(out_of_domain=True)) == (50, 'Needs review')
    assert decide(70, A(judge={'verdict': 'not_enough_evidence'})) == (70, 'Likely authentic')


def test_decide_judge_refuted_never_raises_a_score():
    assert decide(10, A(judge={'verdict': 'refuted'})) == (10, 'Potentially misleading')


def test_assess_runs_the_judge_only_when_enabled_and_no_fact_check(monkeypatch):
    calls = []

    async def fake(claim, evidence, **kw):
        calls.append(kw)
        return {'verdict': 'refuted', 'reason': 'contradicted'}
    monkeypatch.setattr(assessment, 'judge', fake)
    ev = [review('Mixed', snippet='something else entirely about Mars rovers', source_type='web_search')]
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings())
    assert assessment.assess(CLAIM, ev, MODEL)[1]['judge'] is None and not calls
    monkeypatch.setattr(assessment, 'get_settings', lambda: settings(judge_enabled=True, judge_model='m'))
    _, out = assessment.assess(CLAIM, ev, MODEL)
    assert out['judge']['verdict'] == 'refuted' and calls[0]['model'] == 'm' and 'Gemini evidence judge: refuted' in out['notes'][0]
    calls.clear()
    _, out = assessment.assess(CLAIM, [review('False')], MODEL)    # a fact check decides; the judge is not asked
    assert out['judge'] is None and not calls


def test_decide_supported_raises_the_score_only_for_out_of_domain_claims():
    assert decide(13, A(judge={'verdict': 'supported'}, out_of_domain=True)) == (85, 'Likely authentic')   # plain fact
    assert decide(90, A(judge={'verdict': 'supported'}, out_of_domain=True)) == (90, 'Likely authentic')
    assert decide(13, A(judge={'verdict': 'supported'})) == (50, 'Needs review')   # political claim: judge too unreliable to raise
    assert decide(70, A(judge={'verdict': 'refuted'}, out_of_domain=True)) == (20, 'Potentially misleading')


def test_short_claims_need_a_much_closer_match_to_a_review():
    fuji = review('False', snippet='Japan Mount Fuji has erupted for the first time.')
    assert fact_check_verdict('Mount Fuji is in Japan.', [fuji]) is None           # a review of a different claim
    burj = review('This image is fake', snippet='An image shows the Burj Khalifa in Dubai, UAE, on fire.')
    assert fact_check_verdict('The Burj Khalifa is in Dubai.', [burj]) is None
    same = review('False', snippet='Mount Fuji is in Japan.')
    assert fact_check_verdict('Mount Fuji is in Japan.', [same])['verdict'] == 'false'   # identical text still matches
