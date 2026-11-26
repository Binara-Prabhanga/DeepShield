"""Gemini evidence judge: prompt, parsing and the evaluation summary (no network)."""
import numpy as np

from app.services import judge as jd
from research import eval_judge as ev


def test_prompt_contains_claim_evidence_and_rating_and_treats_them_as_data():
    text = jd.build_prompt('Claim X', [{'title': 'T', 'publisher': 'AP', 'snippet': 'S1', 'rating': 'False'}])
    assert 'CLAIM: Claim X' in text and 'Fact-check rating: False' in text and 'untrusted data' in text
    assert '(none)' in jd.build_prompt('Claim X', [])


def test_parse_accepts_only_known_verdicts():
    assert jd.parse('{"verdict": "refuted", "reason": "r"}') == {'verdict': 'refuted', 'reason': 'r'}
    assert jd.parse('{"verdict": "true", "reason": "r"}') is None
    assert jd.parse('not json') is None and jd.parse('{}') is None


def test_judge_without_key_or_evidence_does_nothing(monkeypatch):
    import asyncio
    monkeypatch.setattr(jd, 'get_settings', lambda: type('S', (), {'gemini_api_key': '', 'gemini_api_keys': '', 'gemini_model': 'm'})())
    assert asyncio.run(jd.judge('c', [{'title': 't'}])) is None
    monkeypatch.setattr(jd, 'get_settings', lambda: type('S', (), {'gemini_api_key': 'k', 'gemini_api_keys': '', 'gemini_model': 'm'})())
    assert asyncio.run(jd.judge('c', [])) is None


def rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = int(i % 2)
        verdict = ('supported' if y else 'refuted') if i % 3 else 'not_enough_evidence'
        out.append({'id': str(i), 'y_real': y, 'p_real': float(np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.2), 0.01, 0.99)),
                    's_ev': None if i % 5 == 0 else float(rng.normal()), 'verdict': verdict})
    return out


def test_summarise_reports_coverage_accuracy_and_cv():
    m = ev.summarise(rows())
    assert m['n_answered'] == 60 and abs(m['coverage_decided'] - 40 / 60) < 1e-9
    assert m['decided']['judge_accuracy'] == 1.0
    assert m['auroc_all_answered']['judge']['value'] > 0.7
    assert set(m['cv_auroc']) == {'text', 'text+judge', 'text+judge+s_ev'}


def test_summarise_ignores_claims_gemini_never_answered():
    data = rows()
    data[0]['verdict'] = None
    assert ev.summarise(data)['n_answered'] == 59


# ---- key rotation ----
import asyncio
import httpx


class FakeResponse:
    def __init__(self, status, text=''):
        self.status_code = status
        self._text = text
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('x', request=None, response=None)

    def json(self):
        return {'candidates': [{'content': {'parts': [{'text': self._text}]}}]}


def fake_client(statuses_by_key, seen):
    class C:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            key = headers['x-goog-api-key']
            seen.append(key)
            status = statuses_by_key[key]
            return FakeResponse(status, '{"verdict": "refuted", "reason": "r"}')
    return C


def setup(monkeypatch, keys, statuses, seen):
    settings = type('S', (), {'gemini_api_key': keys[0], 'gemini_api_keys': ','.join(keys[1:]), 'gemini_model': 'm'})()
    monkeypatch.setattr(jd, 'get_settings', lambda: settings)
    monkeypatch.setattr(jd.httpx, 'AsyncClient', fake_client(statuses, seen))
    monkeypatch.setattr(jd, '_resting', {})


def test_api_keys_lists_primary_then_extras_without_blanks_or_repeats():
    s = type('S', (), {'gemini_api_key': 'a', 'gemini_api_keys': ' b, ,a,c '})()
    assert jd.api_keys(s) == ['a', 'b', 'c']


def test_judge_moves_to_the_next_key_when_one_hits_its_quota(monkeypatch):
    seen = []
    setup(monkeypatch, ['k1', 'k2'], {'k1': 429, 'k2': 200}, seen)
    out = asyncio.run(jd.judge('c', [{'title': 't'}]))
    assert out['verdict'] == 'refuted' and seen == ['k1', 'k2']
    asyncio.run(jd.judge('c', [{'title': 't'}]))   # k1 is resting now: straight to k2
    assert seen[2:] == ['k2']


def test_judge_gives_up_without_waiting_when_every_key_is_resting(monkeypatch):
    seen = []
    setup(monkeypatch, ['k1', 'k2'], {'k1': 429, 'k2': 429}, seen)
    assert asyncio.run(jd.judge('c', [{'title': 't'}])) is None
    assert seen == ['k1', 'k2']


def test_judge_falls_back_to_the_second_model_on_a_busy_server(monkeypatch):
    seen = []

    class C:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            model = url.split('/models/')[1].split(':')[0]
            seen.append(model)
            return FakeResponse(503 if model == 'm1' else 200, '{"verdict": "supported", "reason": "r"}')

    settings = type('S', (), {'gemini_api_key': 'k', 'gemini_api_keys': '', 'gemini_model': 'x'})()
    monkeypatch.setattr(jd, 'get_settings', lambda: settings)
    monkeypatch.setattr(jd.httpx, 'AsyncClient', C)
    monkeypatch.setattr(jd, '_resting', {})
    out = asyncio.run(jd.judge('c', [{'title': 't'}], model='m1, m2'))
    assert out['verdict'] == 'supported' and seen == ['m1', 'm2']
    asyncio.run(jd.judge('c', [{'title': 't'}], model='m1,m2'))   # m1 is resting: straight to m2
    assert seen[2:] == ['m2']


def test_the_judging_step_has_a_hard_deadline(monkeypatch):
    from app.services import assessment

    async def hang(*a, **kw):
        await asyncio.sleep(60)
    monkeypatch.setattr(assessment, 'judge', hang)
    monkeypatch.setattr(assessment, 'JUDGE_DEADLINE', 0.2)

    async def no_intros(evidence):
        return evidence
    monkeypatch.setattr(assessment, 'with_wikipedia_intros', no_intros)
    assert asyncio.run(assessment._judge('c', [{'title': 't'}], 'm', 'knowledge')) is None
