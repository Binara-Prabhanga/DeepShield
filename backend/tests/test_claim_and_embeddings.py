"""Claim extraction for long articles, and the optional embedding ranking (no model download)."""
import numpy as np

from app.services import evidence
from app.services.preprocess import extract_claim

ARTICLE = (
    'Local residents gathered on Saturday for the annual spring fair in the town centre. '
    'There was music, food and a children\'s parade through the main street. '
    'Mayor Jane Smith said unemployment in the county doubled to 12 percent last year. '
    'Organisers thanked the volunteers who helped set up the stalls. '
    'The fair will return next April. ') * 2


def test_short_text_is_the_claim_itself():
    assert extract_claim('Vaccines cause autism. Share this now!') == 'Vaccines cause autism.'
    assert extract_claim('Headline here that is long enough\nBody line') == 'Headline here that is long enough'


def test_long_article_gives_the_check_worthy_sentence():
    assert len(ARTICLE) > 400
    assert 'doubled to 12 percent' in extract_claim(ARTICLE)


def test_long_text_without_usable_sentences_falls_back_to_the_first_one():
    assert extract_claim('word ' * 200) == ('word ' * 200).strip()[:300]


DOCS = [
    {'title': 'Ocean warming', 'snippet': 'x', 'url': 'https://e.org/a', 'source_type': 'web_search'},
    {'title': 'Space weather', 'snippet': 'y', 'url': 'https://e.org/b', 'source_type': 'web_search'},
]


def settings(name):
    return type('S', (), {'embedding_model': name})()


def test_embedding_model_ranks_by_cosine_of_vectors(monkeypatch):
    monkeypatch.setattr(evidence, 'get_settings', lambda: settings('stub'))
    monkeypatch.setattr(evidence, '_embedder', lambda name: (_ for _ in ()).throw(AssertionError('not used')), raising=True)
    vectors = np.array([[1, 0], [0.2, 0.98], [0.9, 0.44]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    monkeypatch.setattr(evidence, '_semantic_vectors', lambda documents: vectors)
    ranked = evidence.rank_evidence('claim', [dict(d) for d in DOCS])
    assert [r['url'] for r in ranked] == ['https://e.org/b', 'https://e.org/a']


def test_unloadable_embedding_model_falls_back_to_tfidf(monkeypatch):
    monkeypatch.setattr(evidence, 'get_settings', lambda: settings('no/such-model'))
    monkeypatch.setattr(evidence, '_embedder', lambda name: (_ for _ in ()).throw(OSError('offline')))
    assert evidence._semantic_vectors(['a', 'b']) is None
    ranked = evidence.rank_evidence('ocean warming', [dict(d) for d in DOCS])
    assert ranked[0]['url'] == 'https://e.org/a'


def test_default_is_tfidf(monkeypatch):
    monkeypatch.setattr(evidence, 'get_settings', lambda: settings(''))
    assert evidence._semantic_vectors(['a']) is None


def test_wikipedia_intro_replaces_the_search_fragment_only_where_available():
    results = [{'title': 'Tokyo Olympics', 'snippet': 'fragment ...'}, {'title': 'Other', 'snippet': 'keep me'}]
    out = evidence.apply_intros(results, {'Tokyo Olympics': 'The 2020 Summer Olympics were held in Tokyo in 2021.', 'Other': ''})
    assert out[0]['snippet'].startswith('The 2020 Summer Olympics') and out[1]['snippet'] == 'keep me'
