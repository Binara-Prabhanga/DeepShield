"""Is this claim like the claims the text model was trained on? (out-of-domain check)

The text model learned US political statements (LIAR2). On other text, such as a plain encyclopaedic fact or an
invented story, its probability is not meaningful, yet it still looks confident. This measures how close the claim
is to its nearest LIAR2 training statements (mean cosine similarity of sentence embeddings, k = 25). Below a threshold fitted on
LIAR2 validation claims (research/fit_domain.py) the app says it cannot assess the claim instead of showing a
confident number. It separates Reddit titles from LIAR2 at AUROC 0.97 and Sri Lankan fact-check claims at 0.96
(results/domain), but it does not catch every non-political claim: a text that looks political passes, and a claim
near the threshold can go either way.

Set DOMAIN_MODEL_PATH to the directory written by `python -m research.fit_domain`; empty = check disabled.
If the embedding model cannot be loaded the check is skipped, never failing the analysis.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings


@lru_cache(maxsize=1)
def _bundle(path: str) -> dict:
    bundle = joblib.load(Path(path) / 'domain.joblib')
    bundle['matrix'] = np.asarray(bundle['matrix'], dtype=np.float32)
    return bundle


def similarity(text: str, bundle: dict) -> float:
    """Mean cosine similarity to the k nearest training statements (embeddings are unit length). Averaging k > 1
    neighbours stops one lucky near-duplicate from making an unfamiliar claim look familiar."""
    from app.services.evidence import embed_texts
    vector = embed_texts([text], bundle['model'])[0]
    sims = bundle['matrix'] @ vector
    k = min(int(bundle.get('k', 1)), len(sims))
    return float(np.partition(sims, -k)[-k:].mean())


def out_of_domain(text: str) -> tuple[bool, float | None]:
    """(True when the claim is farther from the training data than the fitted threshold, its similarity).
    (False, None) when the check is not configured or the embedding model is unavailable."""
    path = get_settings().domain_model_path
    if not path:
        return False, None
    try:
        bundle = _bundle(path)
        value = similarity(text, bundle)
    except Exception:
        return False, None
    return value < bundle['threshold'], round(value, 4)
