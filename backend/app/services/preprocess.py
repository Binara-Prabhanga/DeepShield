"""Lightweight preprocessing before inference and evidence retrieval.

Do not remove stop words or rewrite text for BERT/CLIP; real model training must
apply exactly the same normalization as inference.
"""
import html
import re


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r'<[^>]+>', ' ', value)  # Remove simple pasted markup
    value = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', ' ', value)
    value = re.sub(r'[ \t]+', ' ', value)
    value = re.sub(r'\n\s*\n\s*\n+', '\n\n', value)
    return value.strip()


LONG_TEXT = 400   # characters; shorter input is treated as the claim itself (headline or one post)
CLAIM_CUES = re.compile(r'\b(said|says|claims?|claimed|according to|reported|announced|confirmed|will|doubled|tripled|'
                        r'increased?|decreased?|rose|fell|killed|cost|million|billion|percent|per cent)\b|%', re.I)


def _first_sentence(value: str) -> str:
    first_line = next((line.strip() for line in value.splitlines() if len(line.strip()) > 15), value.strip())
    return re.split(r'(?<=[.!?])\s+', first_line, maxsplit=1)[0][:300]


def check_worthiness(sentence: str, position: int) -> float:
    """Heuristic: figures, attribution/assertion words and named entities score high; early sentences get a bonus."""
    figures = 2.0 if re.search(r'\d', sentence) else 0.0
    cues = min(len(CLAIM_CUES.findall(sentence)), 3)
    names = min(len(re.findall(r'(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+', sentence)), 4) * 0.5
    return figures + cues + names + 1.0 / (1 + 0.2 * position)


def extract_claim(value: str) -> str:
    """The claim to fact-check. Short input is the claim itself (first headline or sentence). For a long article
    this picks its most check-worthy sentence (check_worthiness); it is a heuristic, not a trained extractor."""
    if len(value) <= LONG_TEXT:
        return _first_sentence(value)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', value) if s.strip()]
    candidates = [(check_worthiness(s, i), -i, s) for i, s in enumerate(sentences)
                  if 30 <= len(s) <= 300 and not s.endswith('?')]
    return max(candidates)[2] if candidates else _first_sentence(value)
