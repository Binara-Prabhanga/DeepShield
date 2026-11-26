"""Tests for research/collect_srilanka.py parsing (no network)."""
from research import collect_srilanka as cs

PAGE = ('<html><head><script>var x=1;</script></head><body><h1>Viral Posts Falsely Claim X</h1>'
        '<p>Claim: Independent MP said salaries rose by Rs. 60,000 .</p><p>Verdict: MISLEADING</p>'
        '<h2>The Background</h2><p>Base salaries are unchanged &amp; fixed.</p></body></html>')


def test_parse_claim_takes_claim_not_headline():
    claim, verdict, body = cs.parse_claim(cs.page_text(PAGE))
    assert claim == 'Independent MP said salaries rose by Rs. 60,000.'
    assert verdict == 'misleading' and 'Falsely' not in claim
    assert body.startswith('The Background') and '&' in body and 'var x' not in body


def test_reported_claim_from_narrative_page():
    text = ('Viral claim that 8 died is completely FAKE False A viral post claiming eight died is fake. '
            'by Anonymous | June 15, 2026 A post accompanied by a photograph is being shared on Facebook claiming '
            'that a massive rock fell onto a passenger bus on the A26 road, killing eight people. '
            'When FactSeeker investigated this claim, the image was AI-generated.')
    claim, verdict, body = cs.parse_claim(text)
    assert claim == 'A massive rock fell onto a passenger bus on the A26 road, killing eight people.'
    assert verdict is None and body.startswith('A post accompanied')
    leaky = 'by X | May 1, 2024 A fake video circulating on TikTok claims that the bridge collapsed yesterday.'
    assert cs.parse_claim(leaky)[0] is None                         # only candidate contains a verdict word
    assert cs.reported_claim('Police said that the road is open today for everyone.') is None   # not a report of a claim


def test_parse_claim_partial_verdict_and_missing_parts():
    _, verdict, _ = cs.parse_claim('Claim: A thing happened. Verdict: Partially Factual More text')
    assert verdict == 'partially factual'
    assert cs.parse_claim('An explainer without a verdict') == (None, None, '')


def test_labels_tags_and_language():
    assert cs.label_for('fake', None) == ('misleading', 'page')
    assert cs.label_for(None, 'factual') == ('real', 'tags')
    assert cs.label_for('partially factual', 'false') == ('uncertain', 'page')
    assert cs.label_for('explainer', None) == (None, 'none')
    assert cs.verdict_from_tags(['all', 'partially-factual-politics', 'politics']) == 'partially factual'
    assert cs.verdict_from_tags(['factual-society']) == 'factual' and cs.verdict_from_tags(['society']) is None
    assert cs.language('ශ්‍රී ලංකාව') == 'si' and cs.language('இலங்கை') == 'ta' and cs.language('Sri Lanka') == 'en'


def test_verified_span_must_be_copied_from_article():
    article = 'A post   claims that the  Central Bank will issue a new 10,000 rupee note next month. It is not true.'
    assert cs.verified_span('"the Central Bank will issue a new 10,000 rupee note next month."', article) == \
        'The Central Bank will issue a new 10,000 rupee note next month.'
    assert cs.verified_span('the Central Bank is issuing a new 10,000 rupee note', article) is None   # paraphrase
    assert cs.verified_span('NONE', article) is None and cs.verified_span(None, article) is None
    assert cs.verified_span('rupee note next month', article) is None                                 # too short
    assert cs.verified_span('A post claims that the Central Bank will issue', 'A post claims that the Central Bank will issue fake notes') \
        == 'A post claims that the Central Bank will issue.'
    assert cs.verified_span('the Central Bank will issue fake notes', 'the Central Bank will issue fake notes') is None  # leak word


def test_review_sheet_round_trip(tmp_path):
    import csv
    import pytest
    raw = [{'post_id': 1, 'url': 'u1', 'date': '2026-01-02', 'title': 'T1', 'claim': 'A bus fell into the river at Kandy.',
            'page_verdict': None, 'tag_verdict': 'false', 'label': 'misleading'},
           {'post_id': 2, 'url': 'u2', 'date': '2026-01-03', 'title': 'T2', 'claim': None,
            'page_verdict': 'factual', 'tag_verdict': None, 'label': 'real'},
           {'post_id': 3, 'url': 'u3', 'date': '2026-01-04', 'title': 'Explainer', 'claim': None,
            'page_verdict': None, 'tag_verdict': None, 'label': None}]
    path = tmp_path / 'review.csv'
    rows = cs.write_review_sheet(raw, path)
    assert [r['post_id'] for r in rows] == [1, 2]                      # explainer without a verdict is left out
    with pytest.raises(SystemExit):
        cs.write_review_sheet(raw, path)                               # never overwrite a sheet being reviewed
    with open(path, newline='', encoding='utf-8-sig') as handle:
        sheet = list(csv.DictReader(handle))
    sheet[0]['keep'] = 'y'
    sheet[1].update(keep='yes', final_claim='VAT on medicines was removed in 2025.', final_label='Real')
    out = cs.labelled_from_review(sheet)
    assert [(r['id'], r['label'], r['text']) for r in out] == [
        ('factseeker-1', 'misleading', 'A bus fell into the river at Kandy.'),
        ('factseeker-2', 'real', 'VAT on medicines was removed in 2025.')]
    sheet[1]['final_claim'], sheet[1]['suggested_claim'] = '', ''
    with pytest.raises(ValueError, match='post 2'):
        cs.labelled_from_review(sheet)
