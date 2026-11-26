"""Tests for research/eval_live.py helpers (no network, no models)."""
import random

from research import eval_live as ev


def test_is_leak_matches_fact_checker_domains_only():
    assert ev.is_leak({'url': 'https://www.politifact.com/factchecks/2020/x/'})
    assert ev.is_leak({'url': 'https://politifact.com/a'})
    assert not ev.is_leak({'url': 'https://en.wikipedia.org/wiki/PolitiFact'})
    assert not ev.is_leak({'url': 'https://notpolitifact.com/a'}) and not ev.is_leak({})


def synthetic_items(n=200, seed=0):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        y = rng.random() < 0.45
        p = min(0.99, max(0.01, (0.65 if y else 0.35) + rng.uniform(-0.3, 0.3)))
        has_ev = rng.random() < 0.6
        s_ev = (rng.uniform(0, 1) if y else rng.uniform(-1, 0)) if has_ev else None
        items.append({'id': f'x{i}', 'liar2_label': 4 if y else 1, 'y_real': int(y), 'p_real': p,
                      'n_evidence': 3 if has_ev else 0, 'n_leak_dropped': 0, 'n_decisive': 1 if has_ev else 0,
                      's_ev': s_ev, 'gold_s_ev': rng.uniform(-1, 1), 'eea_signed': rng.random() if has_ev else None,
                      'gold_eea_signed': rng.random(), 'trust_p_real': p, 'trust_abstained': abs(p - 0.5) < 0.1})
    return items


def test_summarise_reports_coverage_stance_h4_and_trust():
    m = ev.summarise(synthetic_items(), n_bootstrap=50)
    assert m['n'] == 200 and 0.45 < m['coverage']['any_evidence'] < 0.75
    assert m['c1_stance']['live']['auroc']['value'] > 0.95            # s_ev separates the classes by construction
    assert m['c1_stance']['live_decisive_only']['n'] == sum(r['n_decisive'] > 0 for r in synthetic_items())
    assert 0.3 < m['c1_stance']['gold_same_claims']['auroc']['value'] < 0.7
    assert m['c3_h4']['live']['n'] == sum(r['eea_signed'] is not None for r in synthetic_items())
    trust = m['c2_trust']['trust_live']
    assert trust['brier'] == m['c2_trust']['text_only']['brier'] and 0 < trust['abstain_rate'] < 1


def test_stratified_sample_keeps_label_share():
    from research.data import Example
    rows = [Example(id=f'v{i:03d}', text='t', label=0, label_name='', binary_label=int(i < 40), split='val')
            for i in range(100)]
    sample = ev.stratified_sample(rows, 50)
    assert len(sample) == 50 and sum(e.binary_label for e in sample) == 20
    assert [e.id for e in sample] == [e.id for e in ev.stratified_sample(rows, 50)]
    assert len(ev.stratified_sample(rows, 500)) == 100


def test_live_s_ev_drops_leaks_and_handles_no_evidence(monkeypatch):
    from app.services import stance
    seen = {}

    def fake_classify(claim, evidence):
        seen['urls'] = [e['url'] for e in evidence]
        return [dict(e, stance={'label': 'refutes', 'p_support': 0.0, 'p_refute': 1.0, 'p_unrelated': 0.0,
                                'source': 'nli'}) for e in evidence]
    monkeypatch.setattr(stance, 'classify_stance', fake_classify)
    snapshot = {'evidence': [{'url': 'https://en.wikipedia.org/wiki/A', 'relevance': 1.0},
                             {'url': 'https://www.politifact.com/x', 'relevance': 1.0}]}
    assert ev.live_s_ev('claim', snapshot) == -1.0 and seen['urls'] == ['https://en.wikipedia.org/wiki/A']
    assert ev.live_s_ev('claim', {'evidence': [{'url': 'https://politifact.com/y'}]}) is None


def test_strict_drops_fact_checks_of_the_same_claim_only(monkeypatch):
    claim = 'Says the unemployment rate in Wisconsin doubled under Governor Walker.'
    same = {'url': 'https://www.factcheck.org/a', 'source_type': 'fact_check',
            'snippet': 'The unemployment rate in Wisconsin doubled under Walker.'}
    other = {'url': 'https://www.factcheck.org/b', 'source_type': 'fact_check',
             'snippet': 'Walker cut funding for public schools in Wisconsin by a billion dollars.'}
    wiki = {'url': 'https://en.wikipedia.org/wiki/Scott_Walker', 'source_type': 'web_search',
            'snippet': 'The unemployment rate in Wisconsin doubled under Walker.'}
    assert ev.same_claim(same, claim) and not ev.same_claim(other, claim) and not ev.same_claim(wiki, claim)
    monkeypatch.setattr(ev, 'STRICT', False)
    assert ev.keep(same, claim)
    monkeypatch.setattr(ev, 'STRICT', True)
    assert not ev.keep(same, claim) and ev.keep(other, claim) and ev.keep(wiki, claim)
    assert not ev.keep({'url': 'https://www.politifact.com/x'}, claim)


def test_summarise_tolerates_decisive_items_without_s_ev():
    items = synthetic_items()
    items[0].update(n_decisive=1, s_ev=None)
    m = ev.summarise(items, n_bootstrap=20)
    assert m['c1_stance']['agreement_where_both_decisive']['n'] < m['n']
