"""Evidence stance (C1): does each evidence item support, refute or not address the claim?

Retrieval ranks evidence by similarity only, so a closely matching debunk reads as support.
This service labels each item with an NLI model (premise = evidence, hypothesis = claim):
entailment -> supports, contradiction -> refutes, neutral -> unrelated.
A fact-check rating, when present and unambiguous, overrides NLI.

By default evidence is scored sentence by sentence, skipping sentences that only restate
the claim (debunks usually quote the claim first), then combined (see aggregate()).
The default combining rule is 'decisive', chosen in the C1 experiment (results/c1_stance/README.md).

Evidence score: s_ev = sum_i r_i * (P_sup_i - P_ref_i) / sum_i r_i, in [-1, 1],
where r_i is the retrieval relevance. None means no usable evidence (not the same as 0).

STANCE_MODEL must name an NLI model with entailment/neutral/contradiction labels,
e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli. Requires requirements-ml.txt.
Used by the API through app/services/assessment.py (off until configured).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.config import get_settings

# (premise, hypothesis) pairs -> [{'entailment': p, 'neutral': p, 'contradiction': p}, ...]
NLIBackend = Callable[[Sequence[tuple[str, str]]], list[dict[str, float]]]

SUPPORTS, REFUTES, UNRELATED = 'supports', 'refutes', 'unrelated'

# Checked in this order: ambiguous first ("half true", "mixture"), then refuting ("not true",
# "mostly false" contain "true"/"false"), then supporting. Ambiguous ratings fall back to NLI.
_AMBIGUOUS_RATINGS = ('half', 'mix', 'unproven', 'unverified', 'missing context', 'needs context',
                      'partly', 'partially', 'satire', 'outdated')
_REFUTING_RATINGS = ('false', 'fake', 'pants on fire', 'incorrect', 'inaccurate', 'misleading', 'not true', 'untrue', 'wrong',
                     'fabricat', 'baseless', 'debunk', 'hoax', 'scam', 'no evidence', 'distort', 'exaggerat')
_SUPPORTING_RATINGS = ('true', 'correct', 'accurate', 'verified', 'confirmed')


def rating_stance(rating: str) -> str | None:
    """Map a ClaimReview textualRating to a stance, or None when it is missing or ambiguous."""
    text = (rating or '').strip().lower()
    if not text or any(word in text for word in _AMBIGUOUS_RATINGS):
        return None
    if any(word in text for word in _REFUTING_RATINGS):
        return REFUTES
    if any(word in text for word in _SUPPORTING_RATINGS):
        return SUPPORTS
    return None


def nli_label_indices(id2label: dict) -> dict[str, int]:
    """Find entailment/neutral/contradiction indices from a model config; label order differs by model."""
    found = {}
    for index, name in id2label.items():
        name = str(name).lower()
        for key in ('entailment', 'neutral', 'contradiction'):
            if name.startswith(key[:6]):            # 'entail', 'neutra', 'contra'
                found[key] = int(index)
    if len(found) != 3:
        raise RuntimeError(f'Stance model needs entailment/neutral/contradiction labels, got id2label={id2label}')
    return found


@lru_cache
def _nli_bundle(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from app.services.model_adapter import _device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(_device()).eval()
    return tokenizer, model, nli_label_indices(model.config.id2label)


def transformers_nli(model_name: str, batch_size: int = 16) -> NLIBackend:
    """NLI backend from a Hugging Face model name or local directory."""
    def run(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        import torch
        tokenizer, model, indices = _nli_bundle(model_name)
        results = []
        for start in range(0, len(pairs), batch_size):
            premises, hypotheses = zip(*pairs[start:start + batch_size])
            # Truncate only the evidence; the claim must stay whole.
            enc = tokenizer(list(premises), list(hypotheses), truncation='only_first', max_length=512,
                            padding=True, return_tensors='pt').to(model.device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu()
            results += [{key: float(row[i]) for key, i in indices.items()} for row in probs]
        return results
    return run


def _default_backend() -> NLIBackend:
    name = get_settings().stance_model
    if not name:
        raise RuntimeError('STANCE_MODEL is not set; evidence stance is disabled')
    return transformers_nli(name)


def evidence_text(item: dict) -> str:
    return ' '.join(part for part in (item.get('title', ''), item.get('snippet', '')) if part).strip()


_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=["\'“‘(]?[A-Z0-9])')
_WORD = re.compile(r"[a-z0-9']+")
_ATTRIBUTION = {'said', 'says', 'say', 'saying', 'claimed', 'claims', 'claiming', 'posted', 'wrote', 'tweeted',
                'stated', 'states', 'according', 'asserted', 'suggested', 'argued', 'alleged', 'shared'}
_CONTRAST = {'but', 'however', 'not', 'no', 'never', 'false', 'wrong', 'incorrect', 'actually', 'although',
             'though', 'misleading', 'only', 'nothing', "isn't", "wasn't", "doesn't", "didn't", 'contrary'}
MAX_SENTENCES = 12


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS}


def is_restatement(sentence: str, claim: str, min_overlap: float = 0.6) -> bool:
    """True for sentences that only report the claim ("X said that <claim>").

    NLI reads such a sentence as entailing the claim, so a debunk article that quotes
    the claim before refuting it would otherwise look like support. Sentences with a
    contrast or negation word are kept, because they may carry the refutation.
    """
    words = set(_WORD.findall(sentence.lower()))
    claim_words = _content_words(claim)
    if not claim_words or not words & _ATTRIBUTION or words & _CONTRAST:
        return False
    return len(claim_words & words) / len(claim_words) >= min_overlap


def evidence_sentences(text: str, claim: str) -> list[str]:
    """Sentences worth scoring: restatements removed, very short fragments dropped.
    Falls back to the whole text when nothing is left."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if len(s.split()) >= 4]
    kept = [s for s in sentences if not is_restatement(s, claim)][:MAX_SENTENCES]
    return kept or [text]


AGGREGATIONS = ('decisive', 'max', 'mean')


def premise_scores(pairs: Sequence[tuple[str, str]], nli: NLIBackend,
                   granularity: str = 'sentence') -> list[list[tuple[str, dict[str, float]]]]:
    """Per-premise NLI scores for (claim, evidence_text) pairs, in one batched backend call.
    Returns, for each pair, a list of (premise, probs). 'passage' gives one premise per pair."""
    if granularity not in ('passage', 'sentence'):
        raise ValueError("granularity must be 'passage' or 'sentence'")
    groups = [[text] if granularity == 'passage' else evidence_sentences(text, claim) for claim, text in pairs]
    flat = [(premise, claim) for (claim, _), premises in zip(pairs, groups) for premise in premises]
    scores = iter(nli(flat))
    return [[(premise, next(scores)) for premise in premises] for premises in groups]


def aggregate(group: Sequence[dict[str, float]], method: str = 'decisive') -> dict[str, float]:
    """Combine per-sentence NLI probabilities into one.

    decisive : the sentence with the highest max(entailment, contradiction) decides alone.
    max      : entailment from the most-entailing sentence, contradiction from the most-
               contradicting one, neutral from the least-neutral one; renormalised. A single
               "but..." sentence no longer silences a strongly supporting one.
    mean     : average over sentences.
    """
    if not group:
        raise ValueError('aggregate needs at least one score')
    if method == 'decisive':
        return dict(max(group, key=lambda p: max(p['entailment'], p['contradiction'])))
    if method == 'mean':
        return {key: sum(p[key] for p in group) / len(group) for key in ('entailment', 'neutral', 'contradiction')}
    if method == 'max':
        raw = {'entailment': max(p['entailment'] for p in group),
               'contradiction': max(p['contradiction'] for p in group),
               'neutral': min(p['neutral'] for p in group)}
        total = sum(raw.values()) or 1.0
        return {key: value / total for key, value in raw.items()}
    raise ValueError(f'aggregation must be one of {AGGREGATIONS}')


def stance_probs(pairs: Sequence[tuple[str, str]], nli: NLIBackend, granularity: str = 'sentence',
                 aggregation: str = 'decisive') -> list[dict[str, float]]:
    """NLI probabilities for (claim, evidence_text) pairs, in one batched backend call.

    granularity='passage'  : the whole evidence text is one premise.
    granularity='sentence' : each kept sentence is scored, then combined with `aggregation`.
    """
    return [aggregate([p for _, p in group], aggregation) for group in premise_scores(pairs, nli, granularity)]


def classify_stance(claim: str, evidence: list[dict], nli: NLIBackend | None = None,
                    granularity: str = 'sentence', aggregation: str = 'decisive') -> list[dict]:
    """Return copies of the evidence items, each with a 'stance' dict:
    {label, p_support, p_refute, p_unrelated, source: 'nli' | 'fact_check_rating' | 'empty'}.

    The rating override assumes the fact-checked claim is the user's claim. Google Fact Check
    can return a review of a related claim, so check override cases in the experiment logs.
    """
    items = [dict(item) for item in evidence]
    pending = []
    for item in items:
        override = rating_stance(item.get('rating', ''))
        if override is not None:
            item['stance'] = {'label': override, 'p_support': float(override == SUPPORTS),
                              'p_refute': float(override == REFUTES), 'p_unrelated': 0.0,
                              'source': 'fact_check_rating'}
        elif evidence_text(item):
            pending.append(item)
        else:
            item['stance'] = {'label': UNRELATED, 'p_support': 0.0, 'p_refute': 0.0, 'p_unrelated': 1.0,
                              'source': 'empty'}
    if pending:
        scores = stance_probs([(claim, evidence_text(item)) for item in pending], nli or _default_backend(),
                              granularity, aggregation)
        for item, p in zip(pending, scores, strict=True):
            probs = {SUPPORTS: p['entailment'], REFUTES: p['contradiction'], UNRELATED: p['neutral']}
            item['stance'] = {'label': max(probs, key=probs.get), 'p_support': round(probs[SUPPORTS], 4),
                              'p_refute': round(probs[REFUTES], 4), 'p_unrelated': round(probs[UNRELATED], 4),
                              'source': 'nli'}
    return items


def evidence_score(items: list[dict]) -> float | None:
    """Relevance-weighted support minus refute, in [-1, 1]. Items without 'relevance' weigh 1.0."""
    total = weighted = 0.0
    for item in items:
        stance = item.get('stance')
        if not stance or stance.get('source') == 'empty':
            continue
        weight = max(0.0, float(item.get('relevance', 1.0)))
        total += weight
        weighted += weight * (stance['p_support'] - stance['p_refute'])
    return round(weighted / total, 4) if total > 0 else None
