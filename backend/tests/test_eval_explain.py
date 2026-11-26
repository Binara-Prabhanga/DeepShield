"""Tests for research/eval_explain.py with toy models whose important inputs are known."""
import math
import random

import numpy as np
import pytest

from research import eval_explain as ev


def toy_predict(texts):
    """P(real) rises with 'good' and falls with 'bad'; every other word is ignored."""
    out = []
    for t in texts:
        words = t.lower().split()
        z = 2.0 * words.count('good') - 2.0 * words.count('bad') + 0.1
        out.append(1 / (1 + math.exp(-z)))
    return np.array(out)


UNITS = ['the ', 'good ', 'news ', 'is ', 'good ', 'and ', 'not ', 'bad ', 'at ', 'all']


def test_shap_ranking_puts_predicted_class_drivers_first():
    values = [0, 0.5, 0, 0, 0.4, 0, 0, -0.3, 0, 0]
    assert ev.shap_ranking(UNITS, values, predicted_real=True)[:2] == [1, 4]
    assert ev.shap_ranking(UNITS, values, predicted_real=False)[0] == 7
    assert ev.rankable(['a ', '  ', 'b']) == [0, 2]


def test_true_ranking_beats_random():
    true_rank = [1, 4] + [i for i in range(len(UNITS)) if i not in (1, 4)]
    good = ev.faithfulness(UNITS, true_rank, toy_predict)
    rand = ev.random_faithfulness(UNITS, toy_predict, seed=0, n_random=20)
    assert good['comprehensiveness'] > rand['comprehensiveness']
    assert good['sufficiency'] < rand['sufficiency']
    assert set(good['by_fraction']) == {str(f) for f in ev.FRACTIONS}


def test_shap_attributions_reconstruct_text_and_find_drivers():
    text = ''.join(UNITS)
    units, values = ev.shap_attributions(text, toy_predict, max_evals=200)
    assert ''.join(units) == text
    by_word = {u.strip().lower(): v for u, v in zip(units, values)}
    assert by_word['good'] > 0 > by_word['bad']
    assert abs(by_word['news']) < 1e-6


def test_image_faithfulness_prefers_cam_order():
    image = np.zeros((8, 8, 3))
    image[:4, :4] = 1.0                                             # bright top-left quadrant

    def predict(images):                                            # P(real) = brightness of top-left
        return np.array([0.5 + 0.5 * img[:4, :4].mean() for img in images])
    cam = np.array([[1.0, 0.0], [0.0, 0.0]])
    good = ev.image_faithfulness(image, cam, predict)
    worst = ev.image_faithfulness(image, cam, predict, ranking=[3, 2, 1, 0])
    assert good['comprehensiveness'] > worst['comprehensiveness']
    assert good['sufficiency'] <= worst['sufficiency']


def test_sentence_evidence_uses_nli_argmax():
    row = {'sentences': [['A.', {'entailment': 0.7, 'neutral': 0.2, 'contradiction': 0.1}],
                         ['B.', {'entailment': 0.1, 'neutral': 0.2, 'contradiction': 0.7}],
                         ['C.', {'entailment': 0.1, 'neutral': 0.8, 'contradiction': 0.1}]]}
    assert [e['stance']['label'] for e in ev.sentence_evidence(row)] == ['supports', 'refutes', 'unrelated']


def test_error_detection_finds_informative_eea():
    rng = random.Random(0)
    items = []
    for _ in range(200):
        correct = rng.random() < 0.7
        items.append({'correct': correct, 'confidence': rng.uniform(0.5, 1.0),
                      'eea_signed': rng.uniform(0.4, 1.0) if correct else rng.uniform(0.0, 0.6),
                      'eea_unsigned': rng.uniform(0, 1)})
    items.append({'correct': True, 'confidence': 0.9, 'eea_signed': None, 'eea_unsigned': None})
    result = ev.error_detection(items, n_bootstrap=100, seed=1)
    assert result['n'] == 200 and result['n_without_eea'] == 1
    assert result['auroc_low_eea']['value'] > 0.8
    assert 0.35 < result['auroc_low_confidence']['value'] < 0.65   # confidence is noise here
    assert result['mean_eea_wrong'] < result['mean_eea_correct']


def test_audit_item_and_summarise_end_to_end():
    row = {'id': 'liar2-1', 'liar2_label': 5, 'claim': 'The good news is good and not bad at all',
           'sentences': [['The news is good for everyone involved.',
                          {'entailment': 0.8, 'neutral': 0.1, 'contradiction': 0.1}]]}
    record = ev.audit_item(row, toy_predict, seeds=(1, 2), max_evals=200)
    assert record['correct'] and record['y_real'] == 1
    assert record['eea_signed'] is not None and 'good' in record['eea_matched']
    assert record['shap_comprehensiveness'] > record['random1_comprehensiveness']
    summary = ev.summarise([record, dict(record, id='liar2-2')], seeds=(1, 2), n_bootstrap=20)
    assert summary['n'] == 2 and summary['faithfulness']['comprehensiveness']['shap'] > 0
    assert summary['error_detection']['n'] == 2                     # one class only -> no AUROC


def test_clip_view_matches_clip_crop_size():
    from PIL import Image
    for size in ((640, 480), (100, 300), (224, 224)):
        assert ev.clip_view(Image.new('RGB', size)).size == (224, 224)


def test_summarise_images():
    records = [{'correct': True, 'gradcam_comprehensiveness': 0.3, 'random_comprehensiveness': 0.1,
                'gradcam_sufficiency': 0.0, 'random_sufficiency': 0.2}] * 10
    out = ev.summarise_images(records, n_bootstrap=20)
    assert out['n'] == 10 and out['comprehensiveness']['gradcam_minus_random']['value'] == pytest.approx(0.2)
    assert out['sufficiency']['gradcam_minus_random']['value'] == pytest.approx(-0.2)
