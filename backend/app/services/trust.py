"""Calibrated, decomposed Trust Score with abstention (C2).

    Trust = sigmoid((w_text * logit(p_text) + w_fused * logit(p_fused) + w_evidence * s_ev + b) / T)

p_text and p_fused are the text and image-text models' probabilities that the item is real;
s_ev is the evidence score from stance.evidence_score(), in [-1, 1]. The weights are learned by
logistic regression and T by temperature scaling, both on held-out data (never on test).

Missing signals: each signal also has a 0/1 'missing' feature. A missing signal contributes 0
through its value and lets the model learn how much the absence itself shifts the score. If a
signal was never missing in the fitting data, its 'missing' weight is 0, so absence is neutral.

Abstention: split conformal prediction. On a separate calibration set we take the
ceil((n+1)(1-alpha))-th smallest score 1 - P(true class) as qhat. The prediction set is every
class with 1 - P(class) <= qhat. It contains the true class with probability >= 1 - alpha
(marginally, if calibration and new data are exchangeable). Any set other than exactly one
class returns 'Needs review'.

Probabilities here are P(real), matching ModelOutput.fused_probability_real.
Used by the API through app/services/assessment.py (off until configured).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import get_settings

SIGNALS = ('text', 'fused', 'evidence')
FEATURES = tuple(f'{signal}_{kind}' for signal in SIGNALS for kind in ('value', 'missing'))
REAL, MISLEADING = 'real', 'misleading'
DECISIONS = {(REAL,): 'Likely authentic', (MISLEADING,): 'Potentially misleading'}
NEEDS_REVIEW = 'Needs review'
_EPS = 1e-4
CONFIG_FILE = 'trust_config.json'


def logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def features(signals: dict) -> list[float]:
    """{'text': p_real | None, 'fused': p_real | None, 'evidence': s_ev | None} -> FEATURES order.
    Probabilities enter as logits; s_ev enters as is."""
    row = []
    for name in SIGNALS:
        value = signals.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            row += [0.0, 1.0]
        else:
            row += [float(value) if name == 'evidence' else logit(value), 0.0]
    return row


@dataclass
class TrustModel:
    weights: dict[str, float]
    bias: float
    temperature: float = 1.0
    alpha: float | None = None
    qhat: float | None = None            # None = no abstention
    meta: dict = field(default_factory=dict)

    def margin(self, signals: dict) -> float:
        """Calibrated logit of P(real)."""
        z = sum(self.weights.get(f, 0.0) * x for f, x in zip(FEATURES, features(signals)))
        return (z + self.bias) / self.temperature

    def probability(self, signals: dict) -> float:
        return sigmoid(self.margin(signals))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrustModel:
        unknown = set(data.get('weights', {})) - set(FEATURES)
        if unknown:
            raise ValueError(f'Unknown trust features: {sorted(unknown)}')
        return cls(**{k: data[k] for k in ('weights', 'bias', 'temperature', 'alpha', 'qhat', 'meta') if k in data})

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / CONFIG_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return path


def load_trust_model(directory: str | Path | None = None) -> TrustModel:
    directory = directory or get_settings().trust_model_path
    if not directory:
        raise RuntimeError('TRUST_MODEL_PATH is not set; the calibrated trust score is disabled')
    return TrustModel.from_dict(json.loads((Path(directory) / CONFIG_FILE).read_text(encoding='utf-8')))


# --------------------------------------------------------------------------- fitting

def fit_weights(signal_rows: Sequence[dict], labels_real: Sequence[int], seed: int = 42,
                c: float = 1.0) -> TrustModel:
    """Logistic regression on FEATURES (L2, strength 1/c). labels_real: 1 = real, 0 = misleading.

    Features that are constant in the fitting data (e.g. a signal that is always missing) get
    weight 0; otherwise a constant 'missing' flag would share the bias and shift scores whenever
    that signal is present at inference time."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    x = np.array([features(row) for row in signal_rows])
    varying = [i for i in range(len(FEATURES)) if np.ptp(x[:, i]) > 0]
    weights = dict.fromkeys(FEATURES, 0.0)
    if not varying:
        p = min(max(float(np.mean(labels_real)), _EPS), 1 - _EPS)
        return TrustModel(weights=weights, bias=round(logit(p), 6))
    clf = LogisticRegression(C=c, random_state=seed, max_iter=1000).fit(x[:, varying], list(labels_real))
    weights.update({FEATURES[i]: round(float(w), 6) for i, w in zip(varying, clf.coef_[0])})
    return TrustModel(weights=weights, bias=round(float(clf.intercept_[0]), 6))


def fit_temperature(margins: Sequence[float], labels_real: Sequence[int]) -> float:
    """Scalar T minimising the log-loss of sigmoid(margin / T) on held-out margins."""
    from scipy.optimize import minimize_scalar

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        return -sum(math.log(min(max(sigmoid(m / t) if y else 1 - sigmoid(m / t), 1e-12), 1.0))
                    for m, y in zip(margins, labels_real))
    return round(math.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x), 6)


def fit_trust(signal_rows: Sequence[dict], labels_real: Sequence[int], seed: int = 42, folds: int = 5) -> TrustModel:
    """Weights on all rows; T on out-of-fold margins from the same rows, so T is not fitted to
    margins the weights have already seen."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    labels = np.asarray(labels_real)
    margins = np.zeros(len(labels))
    for train, held in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels):
        fold = fit_weights([signal_rows[i] for i in train], labels[train], seed)
        margins[held] = [fold.margin(signal_rows[i]) for i in held]
    model = fit_weights(signal_rows, labels, seed)
    model.temperature = fit_temperature(margins.tolist(), labels.tolist())
    return model


def conformal_qhat(probs_real: Sequence[float], labels_real: Sequence[int], alpha: float) -> float:
    """Split-conformal threshold on the score 1 - P(true class). 1.0 when n is too small for alpha."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    scores = sorted(1 - p if y else p for p, y in zip(probs_real, labels_real))
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return 1.0 if k > len(scores) else scores[k - 1]


def prediction_set(p_real: float, qhat: float | None) -> tuple[str, ...]:
    """Classes kept by the conformal rule (both orders fixed: real first). qhat None -> argmax only."""
    if qhat is None:
        return (REAL,) if p_real >= 0.5 else (MISLEADING,)
    return tuple(c for c, p in ((REAL, p_real), (MISLEADING, 1 - p_real)) if 1 - p <= qhat)


# --------------------------------------------------------------------------- scoring

def trust_score(signals: dict, model: TrustModel | None = None) -> dict:
    """Score one item: probability, 0-100 score, decision, conformal set and per-signal breakdown.

    breakdown[i]['contribution'] is that signal's term in the calibrated logit (value and
    missing-flag terms together); 'share' is its fraction of the summed absolute contributions.
    """
    model = model or load_trust_model()
    x = dict(zip(FEATURES, features(signals)))
    parts = []
    for name in SIGNALS:
        term = (model.weights.get(f'{name}_value', 0.0) * x[f'{name}_value']
                + model.weights.get(f'{name}_missing', 0.0) * x[f'{name}_missing']) / model.temperature
        parts.append({'signal': name, 'available': x[f'{name}_missing'] == 0.0,
                      'value': signals.get(name) if x[f'{name}_missing'] == 0.0 else None,
                      'contribution': round(term, 4)})
    total = sum(abs(p['contribution']) for p in parts)
    for part in parts:
        part['share'] = round(abs(part['contribution']) / total, 4) if total else 0.0
    p_real = model.probability(signals)
    kept = prediction_set(p_real, model.qhat)
    return {'probability_real': round(p_real, 4), 'score': round(p_real * 100),
            'decision': DECISIONS.get(kept, NEEDS_REVIEW), 'prediction_set': list(kept),
            'abstained': kept not in DECISIONS, 'alpha': model.alpha,
            'baseline': round(model.bias / model.temperature, 4), 'breakdown': parts}
