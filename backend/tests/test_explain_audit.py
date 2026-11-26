"""Unit tests for app/services/explain_audit.py (EEA, C3)."""
import pytest

from app.services import explain_audit as ea


@pytest.mark.parametrize('word, expected', [
    ('taxes', 'tax'), ('tax', 'tax'), ('rates', 'rat'), ('rated', 'rat'), ('rating', 'rat'), ('rate', 'rat'),
    ('voters', 'voter'), ('class', 'class'), ('quickly', 'quick'), ('pushes', 'push'), ('bus', 'bus'),
    ('red', 'red'), ('used', 'used'), ('2020', '2020'),
])
def test_stem(word, expected):
    assert ea.stem(word) == expected


def test_normalize_drops_stopwords_punctuation_and_possessives():
    assert ea.normalize('The') is None and ea.normalize('.') is None and ea.normalize('a') is None
    assert ea.normalize('Obama’s') == 'obama' and ea.normalize("Taxes,") == 'tax'


def test_top_words_filters_and_keeps_strongest_occurrence():
    tokens = [{'token': 'the', 'value': 0.9}, {'token': 'tax', 'value': 0.2}, {'token': 'Taxes', 'value': -0.5},
              {'token': 'jobs', 'value': 0.3}, {'token': '.', 'value': 0.8}, {'token': 'zero', 'value': 0.0}]
    assert ea.top_words(tokens, 5) == [('tax', -0.5), ('job', 0.3)]


def item(text, label):
    return {'snippet': text, 'stance': {'label': label}}


TOKENS = [{'token': 'unemployment', 'value': -0.6}, {'token': 'doubled', 'value': -0.4},
          {'token': 'jobs', 'value': 0.3}, {'token': 'Wisconsin', 'value': 0.1}]


def test_eea_signed_counts_only_same_direction():
    evidence = [item('Unemployment did not double in Wisconsin.', 'refutes'),
                item('Job numbers rose last year.', 'supports'),
                item('Doubled claims appear online.', 'unrelated')]
    result = ea.eea(TOKENS, evidence, k=10)
    # unemployment(-) in refutes yes; doubled(-) -> 'doubl', refutes has 'doubl' ('double' -> 'doubl') yes;
    # jobs(+) in supports yes; wisconsin(+) only in refutes -> no
    assert result['matched'] == ['unemployment', 'doubl', 'job'] and result['eea'] == 0.75 and result['k'] == 4
    assert ea.eea(TOKENS, evidence, k=10, signed=False)['eea'] == 1.0


def test_eea_k_limits_and_none_cases():
    evidence = [item('Unemployment rose.', 'refutes')]
    assert ea.eea(TOKENS, evidence, k=1) == {'eea': 1.0, 'k': 1, 'matched': ['unemployment'], 'signed': True}
    assert ea.eea(TOKENS, [item('Unemployment rose.', 'unrelated')]) is None      # no stance evidence
    assert ea.eea([{'token': 'the', 'value': 1.0}], evidence) is None              # no usable word
    assert ea.eea(TOKENS, []) is None
