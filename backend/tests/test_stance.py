"""Unit tests for app/services/stance.py. A fake NLI backend is injected, so no model is downloaded."""
import pytest

from app.services import stance


def fake_nli(table):
    """Backend returning fixed probabilities per premise; records the pairs it was called with."""
    calls = []

    def run(pairs):
        calls.extend(pairs)
        return [table[premise] for premise, _ in pairs]
    run.calls = calls
    return run


SUPPORT = {'entailment': 0.9, 'neutral': 0.08, 'contradiction': 0.02}
REFUTE = {'entailment': 0.05, 'neutral': 0.15, 'contradiction': 0.8}
NEUTRAL = {'entailment': 0.1, 'neutral': 0.85, 'contradiction': 0.05}


@pytest.mark.parametrize('rating, expected', [
    ('False', 'refutes'), ('Pants on Fire!', 'refutes'), ('Mostly False', 'refutes'), ('Not true', 'refutes'),
    ('Incorrect', 'refutes'), ('Inaccurate', 'refutes'), ('Misleading', 'refutes'),
    ('True', 'supports'), ('Mostly true', 'supports'), ('Correct', 'supports'),
    ('Half True', None), ('Mixture', None), ('Missing context', None), ('Unproven', None), ('', None),
    ('Four Pinocchios', None),
])
def test_rating_stance(rating, expected):
    assert stance.rating_stance(rating) == expected


def test_nli_label_indices_any_order():
    assert stance.nli_label_indices({0: 'ENTAILMENT', 1: 'NEUTRAL', 2: 'CONTRADICTION'}) == {
        'entailment': 0, 'neutral': 1, 'contradiction': 2}
    assert stance.nli_label_indices({'0': 'contradiction', '1': 'neutral', '2': 'entailment'}) == {
        'contradiction': 0, 'neutral': 1, 'entailment': 2}


def test_nli_label_indices_rejects_two_class_model():
    with pytest.raises(RuntimeError):
        stance.nli_label_indices({0: 'entailment', 1: 'not_entailment'})


def test_classify_uses_evidence_as_premise_and_claim_as_hypothesis():
    nli = fake_nli({'Debunk Officials denied it': REFUTE})
    [item] = stance.classify_stance('The bridge collapsed', [{'title': 'Debunk', 'snippet': 'Officials denied it'}], nli)
    assert nli.calls == [('Debunk Officials denied it', 'The bridge collapsed')]
    assert item['stance'] == {'label': 'refutes', 'p_support': 0.05, 'p_refute': 0.8, 'p_unrelated': 0.15,
                              'source': 'nli'}


def test_rating_overrides_nli_and_skips_model_call():
    nli = fake_nli({'other text': NEUTRAL})
    items = stance.classify_stance('claim', [
        {'title': 'Fact check', 'snippet': 'x', 'rating': 'False'},
        {'title': 'other text', 'rating': 'Half true'},     # ambiguous rating -> NLI
    ], nli)
    assert items[0]['stance']['source'] == 'fact_check_rating' and items[0]['stance']['label'] == 'refutes'
    assert items[1]['stance']['source'] == 'nli' and items[1]['stance']['label'] == 'unrelated'
    assert [premise for premise, _ in nli.calls] == ['other text']


def test_classify_does_not_mutate_input_and_handles_empty():
    evidence = [{'title': '', 'snippet': ''}]
    items = stance.classify_stance('claim', evidence, fake_nli({}))
    assert 'stance' not in evidence[0]
    assert items[0]['stance']['source'] == 'empty'
    assert stance.classify_stance('claim', [], fake_nli({})) == []


def test_missing_model_setting_raises(monkeypatch):
    monkeypatch.setattr(stance, 'get_settings', lambda: type('S', (), {'stance_model': ''})())
    with pytest.raises(RuntimeError, match='STANCE_MODEL'):
        stance.classify_stance('claim', [{'title': 'some evidence'}])


def test_evidence_score_is_relevance_weighted():
    items = stance.classify_stance('claim', [
        {'title': 'a', 'relevance': 0.6}, {'title': 'b', 'relevance': 0.2}, {'title': 'c', 'relevance': 0.2},
    ], fake_nli({'a': REFUTE, 'b': SUPPORT, 'c': NEUTRAL}))
    expected = (0.6 * (0.05 - 0.8) + 0.2 * (0.9 - 0.02) + 0.2 * (0.1 - 0.05)) / 1.0
    assert stance.evidence_score(items) == pytest.approx(expected, abs=1e-4)


def test_evidence_score_bounds_and_none():
    support = stance.classify_stance('c', [{'title': 'x', 'rating': 'True'}], fake_nli({}))
    refute = stance.classify_stance('c', [{'title': 'x', 'rating': 'False'}], fake_nli({}))
    assert stance.evidence_score(support) == 1.0
    assert stance.evidence_score(refute) == -1.0
    assert stance.evidence_score([]) is None
    assert stance.evidence_score([dict(support[0], relevance=0.0)]) is None   # zero relevance -> no signal
    assert stance.evidence_score(stance.classify_stance('c', [{'title': ''}], fake_nli({}))) is None


# --------------------------------------------------------------------------- sentence-level scoring

CLAIM = '90 percent of Americans support universal background checks for gun purchases.'


@pytest.mark.parametrize('sentence, expected', [
    ('Chris Abele said 90 percent of Americans support universal background checks.', True),
    ('A viral post claimed that Americans support universal background checks at 90 percent.', True),
    ('Abele said 90 percent of Americans support universal background checks, but polls show less.', False),
    ('Polls show 90 percent of Americans support universal background checks.', False),   # no attribution
    ('He said the weather was nice in Milwaukee that week.', False),                      # low overlap
])
def test_is_restatement(sentence, expected):
    assert stance.is_restatement(sentence, CLAIM) is expected


def test_evidence_sentences_drops_restatement_and_short_fragments():
    text = ('Abele said 90 percent of Americans support universal background checks. Ok then. '
            'Surveys found support closer to 70 percent nationwide. The poll had a small sample.')
    assert stance.evidence_sentences(text, CLAIM) == [
        'Surveys found support closer to 70 percent nationwide.', 'The poll had a small sample.']


def test_evidence_sentences_falls_back_to_whole_text():
    text = 'Abele said 90 percent of Americans support universal background checks.'
    assert stance.evidence_sentences(text, CLAIM) == [text]


def test_stance_probs_most_decisive_sentence_wins_and_batches_once():
    text = 'The first sentence is neutral here. The second sentence refutes the claim.'
    calls = []

    def nli(pairs):
        calls.append(list(pairs))
        return [NEUTRAL if 'neutral' in premise else REFUTE for premise, _ in pairs]
    [sentence] = stance.stance_probs([('claim', text)], nli, 'sentence')
    [passage] = stance.stance_probs([('claim', text)], lambda pairs: [SUPPORT] * len(pairs), 'passage')
    assert sentence == REFUTE and passage == SUPPORT
    assert len(calls) == 1 and len(calls[0]) == 2


def test_stance_probs_rejects_unknown_granularity():
    with pytest.raises(ValueError):
        stance.stance_probs([('c', 'text')], lambda pairs: [], 'word')


# --------------------------------------------------------------------------- aggregation rules

STRONG_SUPPORT = {'entailment': 0.7, 'neutral': 0.2, 'contradiction': 0.1}
STRONG_REFUTE = {'entailment': 0.05, 'neutral': 0.15, 'contradiction': 0.8}


def test_aggregate_decisive_lets_one_sentence_decide():
    assert stance.aggregate([STRONG_SUPPORT, STRONG_REFUTE], 'decisive') == STRONG_REFUTE


def test_aggregate_max_keeps_both_directions_and_normalises():
    p = stance.aggregate([STRONG_SUPPORT, STRONG_REFUTE], 'max')
    total = 0.7 + 0.8 + 0.15
    assert p == pytest.approx({'entailment': 0.7 / total, 'contradiction': 0.8 / total, 'neutral': 0.15 / total})
    assert sum(p.values()) == pytest.approx(1.0)


def test_aggregate_mean():
    p = stance.aggregate([STRONG_SUPPORT, STRONG_REFUTE], 'mean')
    assert p == pytest.approx({'entailment': 0.375, 'neutral': 0.175, 'contradiction': 0.45})


def test_aggregate_single_sentence_is_unchanged_for_every_rule():
    for method in stance.AGGREGATIONS:
        assert stance.aggregate([STRONG_SUPPORT], method) == pytest.approx(STRONG_SUPPORT)


def test_aggregate_rejects_bad_input():
    with pytest.raises(ValueError):
        stance.aggregate([], 'max')
    with pytest.raises(ValueError):
        stance.aggregate([STRONG_SUPPORT], 'median')


def test_premise_scores_returns_each_sentence():
    text = 'The first sentence is neutral here. The second sentence refutes the claim.'
    [group] = stance.premise_scores([('claim', text)], lambda pairs: [NEUTRAL, REFUTE], 'sentence')
    assert group == [('The first sentence is neutral here.', NEUTRAL), ('The second sentence refutes the claim.', REFUTE)]
