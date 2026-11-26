"""Explanation-Evidence Agreement (C3): do the model's reasons match the evidence?

    EEA = |TopK_SHAP  ∩  words of evidence with the same stance| / k

Direction: a SHAP value > 0 pushes toward "real" and should be backed by *supporting*
evidence; a value < 0 pushes toward "fake" and should be backed by *refuting* evidence.
The unsigned variant (ablation) accepts a match in supporting or refuting evidence.
'Unrelated' evidence never counts.

Matching rule (agreed before experiments): lowercase, keep letters/digits only, drop English
stopwords and words under 2 characters, then light stemming (see stem()). Exact match on stems.
TopK is taken after that filtering, so 'the' or '.' never fill the top-k.

Used by the API through app/services/assessment.py. The faithfulness (deletion) tests are in research/eval_explain.py.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.services.stance import REFUTES, SUPPORTS, evidence_text

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")
_MIN_STEM = 3


def stem(word: str) -> str:
    """Light suffix stripping. Rules in order, first match wins; a rule applies only if at least
    3 characters remain:  -ing, -ed, -ly; -es after s/x/z/ch/sh; -s (not -ss); then a final -e.
    taxes -> tax, rates/rated/rating/rate -> rat, voters -> voter."""
    for suffix in ('ing', 'ed', 'ly'):
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            word = word[:-len(suffix)]
            break
    else:
        if word.endswith('es') and word[:-2].endswith(('s', 'x', 'z', 'ch', 'sh')) and len(word) - 2 >= _MIN_STEM:
            word = word[:-2]
        elif word.endswith('s') and not word.endswith('ss') and len(word) - 1 >= _MIN_STEM:
            word = word[:-1]
    if word.endswith('e') and len(word) - 1 >= _MIN_STEM:
        word = word[:-1]
    return word


def normalize(token: str) -> str | None:
    """A SHAP token or evidence word -> stem, or None for stopwords, punctuation and fragments."""
    word = _NON_ALNUM.sub('', token.lower().replace('’', "'").split("'")[0])
    if len(word) < 2 or word in ENGLISH_STOP_WORDS:
        return None
    return stem(word)


def vocabulary(text: str) -> set[str]:
    return {w for w in (normalize(t) for t in _WORD.findall(text)) if w}


def top_words(tokens: list[dict], k: int) -> list[tuple[str, float]]:
    """Distinct normalized words by |value|, keeping each word's strongest occurrence."""
    best: dict[str, float] = {}
    for item in tokens:
        word, value = normalize(str(item.get('token', ''))), float(item.get('value', 0.0))
        if word and value != 0.0 and abs(value) > abs(best.get(word, 0.0)):
            best[word] = value
    return sorted(best.items(), key=lambda p: abs(p[1]), reverse=True)[:k]


def eea(tokens: list[dict], evidence: list[dict], k: int = 10, signed: bool = True) -> dict | None:
    """EEA for one prediction.

    tokens   : [{'token': str, 'value': float}] as in ModelOutput.explanation_tokens
    evidence : items with a 'stance' dict (see stance.classify_stance) and title/snippet text
    Returns None when there is no supporting or refuting evidence, or no usable SHAP word.
    k is the intended top-k; if fewer usable words exist, the denominator is the number found.
    """
    vocab = {SUPPORTS: set(), REFUTES: set()}
    for item in evidence:
        label = (item.get('stance') or {}).get('label')
        if label in vocab:
            vocab[label] |= vocabulary(evidence_text(item))
    if not vocab[SUPPORTS] and not vocab[REFUTES]:
        return None
    words = top_words(tokens, k)
    if not words:
        return None
    either = vocab[SUPPORTS] | vocab[REFUTES]
    matched = [w for w, v in words if w in (either if not signed else vocab[SUPPORTS if v > 0 else REFUTES])]
    return {'eea': round(len(matched) / len(words), 4), 'k': len(words), 'matched': matched,
            'signed': signed}
