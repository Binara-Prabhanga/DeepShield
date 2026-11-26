"""Use a fact-checker's own verdict when one exists for the same claim.

The text and stance models only estimate how a claim reads. When Google Fact Check returns a review of
(nearly) the same claim, the publisher's rating is better evidence than any model score, so it decides
the result (shown to the user as a note naming the publisher and rating).

A review counts as the same claim when its reviewed-claim text shares at least SAME_CLAIM_JACCARD of its
content words with the submitted claim (research/eval_live.py --strict uses the plain 0.5 measure to *exclude* such
items from experiments; short claims need 0.75 here). Mixed ratings ("half true", "needs context") never override.
"""
from __future__ import annotations

from app.services.explain_audit import vocabulary

SAME_CLAIM_JACCARD = 0.5
SHORT_CLAIM_WORDS = 4         # a claim this short shares words with many unrelated reviews ...
SHORT_CLAIM_JACCARD = 0.75    # ... so it must match the reviewed claim much more closely
FALSE_SCORE, TRUE_SCORE = 8, 92   # not 0/100: a rating is strong evidence, not certainty

_MIXED = ('half', 'mixture', 'mixed', 'partly', 'partially', 'some truth', 'unproven', 'unverified', 'needs context',
          'outdated', 'missing context', 'in dispute', 'disputed', 'satire', 'legend')
_FALSE = ('false', 'pants on fire', 'incorrect', 'fake', 'misleading', 'untrue', 'not true', 'hoax', 'fabricated',
          'no evidence', 'baseless', 'wrong', 'inaccurate', 'scam', 'unsupported')
_TRUE = ('true', 'correct', 'accurate', 'legit', 'verified')


def rating_verdict(rating: str) -> str | None:
    """'false' | 'true' | 'mixed' | None for a textual rating such as 'Mostly False' or 'Pants on Fire'."""
    text = (rating or '').strip().lower()
    if not text:
        return None
    if any(word in text for word in _MIXED):
        return 'mixed'
    if any(word in text for word in _FALSE):   # before _TRUE: "not true" and "mostly false" contain true/false words
        return 'false'
    if any(word in text for word in _TRUE):
        return 'true'
    return None


def same_claim(claim: str, item: dict) -> bool:
    """True when the review's reviewed-claim text matches the claim. Short claims need a closer match: "Mount Fuji is in
    Japan" (3 content words) shares most of its words with "Japan's Mount Fuji has erupted for the first time", a
    different claim whose False rating must not be applied to it."""
    a, b = vocabulary(claim), vocabulary(item.get('snippet', ''))
    if not (a and b):
        return False
    needed = SHORT_CLAIM_JACCARD if len(a) <= SHORT_CLAIM_WORDS else SAME_CLAIM_JACCARD
    return len(a & b) / len(a | b) >= needed


def fact_check_verdict(claim: str, evidence: list[dict]) -> dict | None:
    """{'verdict': 'true'|'false', 'score', 'publisher', 'rating', 'url'} from same-claim reviews, else None.
    Reviews that disagree (one true, one false) or are all mixed/unknown give None."""
    matches = [(item, rating_verdict(item.get('rating', ''))) for item in evidence
               if item.get('source_type') == 'fact_check' and same_claim(claim, item)]
    verdicts = {v for _, v in matches if v in ('true', 'false')}
    if len(verdicts) != 1:
        return None
    verdict = verdicts.pop()
    item = next(i for i, v in matches if v == verdict)
    return {'verdict': verdict, 'score': TRUE_SCORE if verdict == 'true' else FALSE_SCORE,
            'publisher': item.get('publisher', ''), 'rating': item.get('rating', ''), 'url': item.get('url', '')}
