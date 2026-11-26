"""Tests for the pure parts of research/eval_stance.py (no model download)."""
import pytest

from research import eval_stance as ev
from research.data import Example


def test_tune_threshold_finds_separating_point():
    assert ev.tune_threshold([0.1, 0.2, 0.7, 0.9], [0, 0, 1, 1]) == 0.7


def test_binary_metrics_perfect_and_confusion():
    m = ev.binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m['accuracy'] == 0.75
    assert m['refutes']['recall'] == 0.5 and m['supports']['recall'] == 1.0
    assert m['confusion_rows_gold_refutes_supports'] == [[1, 1], [0, 2]]


def test_bootstrap_is_repeatable_and_brackets_value():
    gold, pred = [0, 1] * 20, [0, 1] * 15 + [1, 0] * 5
    a = ev.bootstrap_macro_f1(gold, pred, n=200, seed=1)
    assert a == ev.bootstrap_macro_f1(gold, pred, n=200, seed=1)
    assert a['low'] <= 0.75 <= a['high']
    assert ev.bootstrap_macro_f1(gold, pred, pred, n=50)['high'] == 0.0   # paired diff with itself


def test_mcnemar_counts_discordant_pairs():
    result = ev.mcnemar_exact([1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1])
    assert (result['only_a_correct'], result['only_b_correct']) == (2, 1)
    assert 0 < result['p_value'] <= 1


def test_score_split_pairs_and_gold():
    examples = [Example(id='liar2-1', text='claim one', label=0, label_name='pants_fire', binary_label=1,
                        split='test', meta={'justification': 'evidence one is short'}),
                Example(id='liar2-2', text='claim two', label=3, label_name='half_true', binary_label=0,
                        split='test', meta={'justification': ''})]          # empty justification skipped
    calls = []

    def nli(pairs):
        calls.extend(pairs)
        return [{'entailment': 0.1, 'neutral': 0.2, 'contradiction': 0.7}] * len(pairs)
    [row] = ev.score_split(examples, nli)
    assert calls == [('evidence one is short', 'claim one')] * 2      # passage + sentence
    assert row['gold'] == ev.REFUTES
    assert row['sentences'] == [['evidence one is short', {'entailment': 0.1, 'neutral': 0.2, 'contradiction': 0.7}]]
    ev.add_variants([row])
    for variant in ev.VARIANTS:
        assert row[f'{variant}_s_ev'] == pytest.approx(-0.6)
    assert 0 <= row['relevance'] <= 1


def probs(e, n, c):
    return {'entailment': e, 'neutral': n, 'contradiction': c}


def synthetic_rows(n, split):
    """Raw rows as score_split returns them. Relevance is uninformative (as the G1 claim predicts).
    Passage scores lean to support for refuted claims. True claims have one strongly supporting
    sentence plus a weaker "but..." sentence, which fools the 'decisive' rule only when the
    "but..." sentence is the more confident one."""
    rows = []
    for i in range(n):
        label = [0, 1, 4, 5, 2, 3][i % 6]
        good = label >= 4
        if good:
            passage = probs(0.8, 0.1, 0.1)
            sentences = [['supports', probs(0.7, 0.2, 0.1)], ['but', probs(0.05, 0.15, 0.8)]]
        else:
            passage = probs(0.6, 0.1, 0.3)
            sentences = [['refutes', probs(0.05, 0.15, 0.8)], ['filler', probs(0.1, 0.8, 0.1)]]
        rows.append({'id': f'{split}-{i}', 'split': split, 'liar2_label': label, 'gold': ev.GOLD.get(label),
                     'relevance': 0.3 + (i % 5) * 0.1, 'passage': passage, 'sentences': sentences})
    return ev.add_variants(rows)


def test_evaluate_end_to_end_on_synthetic_rows():
    m = ev.evaluate(synthetic_rows(60, 'val'), synthetic_rows(60, 'test'), bootstrap=50, seed=0)
    systems = m['test_systems']
    assert set(systems) == {'similarity'} | {f'{v}_{kind}' for v in ev.VARIANTS for kind in ('argmax', 'tuned')}
    assert systems['sentence_decisive_argmax']['supports']['recall'] == 0.0   # "but..." wins under decisive
    assert systems['sentence_max_tuned']['macro_f1'] == 1.0
    assert systems['passage_argmax']['macro_f1'] < 1.0
    assert systems['similarity']['macro_f1'] < 1.0
    assert systems['sentence_max_tuned']['n'] == 40                 # labels 2-3 excluded from scoring
    assert set(m['val_macro_f1']) == set(systems)
    assert m['test_diagnostics_by_variant']['sentence_mean']['s_ev_by_liar2_label'][2]['n'] == 10
    assert 'sentence_max_tuned_vs_passage_tuned' in m['paired_comparisons']
    assert m['g1_subset']['n'] > 0
