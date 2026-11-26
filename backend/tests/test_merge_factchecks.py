"""Tests for research/merge_factchecks.py (no network)."""
from research import merge_factchecks as mf

CLAIM = 'Wisconsin unemployment doubled under Governor Walker'


def wiki_item(i):
    return {'title': f'Wisconsin page {i}', 'url': f'https://en.wikipedia.org/wiki/W{i}', 'snippet': 'Wisconsin economy',
            'source_type': 'web_search', 'relevance': 0.1}


FACT = {'title': 'Did unemployment double?', 'url': 'https://www.factcheck.org/x', 'snippet': CLAIM,
        'source_type': 'fact_check', 'rating': 'False'}


def test_merge_keeps_wikipedia_adds_fact_checks_first_and_caps_at_eight():
    wiki = {'id': 'c1', 'providers': {'wikipedia': 'connected', 'google_factcheck': 'not_configured'},
            'evidence': [wiki_item(i) for i in range(8)]}
    facts = {'id': 'c1', 'providers': {'google_factcheck': 'connected'},
             'evidence': [FACT, dict(wiki_item(99), url='https://en.wikipedia.org/wiki/Other')]}
    merged = mf.merge_one(CLAIM, wiki, facts)
    assert merged['providers'] == {'wikipedia': 'connected', 'google_factcheck': 'connected'}
    assert len(merged['evidence']) == 8 and merged['evidence'][0]['url'] == FACT['url']
    assert all(e['url'] != 'https://en.wikipedia.org/wiki/Other' for e in merged['evidence'])   # only fact checks taken


def test_merge_without_fact_checks_is_wikipedia_only():
    wiki = {'id': 'c1', 'providers': {'wikipedia': 'connected'}, 'evidence': [wiki_item(0)]}
    merged = mf.merge_one(CLAIM, wiki, None)
    assert [e['url'] for e in merged['evidence']] == [wiki_item(0)['url']]
    assert merged['providers']['google_factcheck'] == 'missing'
