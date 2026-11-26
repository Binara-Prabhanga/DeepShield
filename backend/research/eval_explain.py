"""C3 experiment E5 (RQ4): are explanations faithful, and does low EEA warn of errors?

Faithfulness (ERASER, DeYoung et al. 2020), for the predicted class probability f:
  comprehensiveness(k) = f(x) - f(x without its top-k% units)   higher = more faithful
  sufficiency(k)       = f(x) - f(only the top-k% units)        lower  = more faithful
averaged over k in FRACTIONS (AOPC). SHAP rankings are compared with random rankings
(N_RANDOM per item, for each seed in SEEDS; mean and sd over seeds).
Text units are SHAP's word tokens; image units are Grad-CAM grid cells filled with the
image's mean colour (image_faithfulness(); runs once the CLIP fusion head is trained).

EEA: app.services.explain_audit.eea() against the justification's sentences, each labelled
supports/refutes/unrelated by the C1 NLI scores cached in premise_scores.jsonl (no NLI rerun).
H4: AUROC of (1 - EEA) for detecting wrong predictions, against (1 - confidence) and both
combined (5-fold cross-validated logistic regression).

Text model:
  --text-model tfidf  PLACEHOLDER TF-IDF + logistic regression on LIAR2 train (default for now).
                      SHAP on a near-linear model is close to faithful by design, so this run
                      mainly checks EEA and the method end to end. Not a DeepShield result.
  --text-model bert   the trained model at TEXT_MODEL_PATH, via model_adapter._text_probs.

Run:    cd backend && python -m research.eval_explain [--limit 100] [--text-model bert]
        python -m research.eval_explain --images 200     (Grad-CAM; needs MULTIMODAL_MODEL_PATH)
Writes results/c3_explain/<timestamp>/{config.json, metrics.json, items.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from app.services import explain_audit
from app.services.stance import REFUTES, SUPPORTS, UNRELATED

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'c3_explain'
DEFAULT_SCORES = (Path(__file__).resolve().parents[2] / 'results' / 'c1_stance' / 'trials'
                  / '500claims_base_20261122-200459' / 'premise_scores.jsonl')
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.5)
SEEDS = (42, 43, 44)
N_RANDOM = 5
TOP_K = 10


# --------------------------------------------------------------------------- attributions

def shap_attributions(text: str, predict, max_evals: int = 300) -> tuple[list[str], list[float]]:
    """SHAP values toward P(real) per word token (same explainer settings as the app).
    Tokens keep their trailing separators, so ''.join(units) reproduces the text."""
    import shap
    explainer = shap.Explainer(lambda xs: np.asarray(predict([str(x) for x in xs])), shap.maskers.Text(r'\W+'))
    sv = explainer([text], max_evals=max_evals, silent=True)[0]
    return [str(u) for u in sv.data], [float(v) for v in sv.values]


def rankable(units: list[str]) -> list[int]:
    return [i for i, u in enumerate(units) if u.strip()]


def shap_ranking(units: list[str], values: list[float], predicted_real: bool) -> list[int]:
    """Unit indices, most supportive of the predicted class first."""
    sign = 1.0 if predicted_real else -1.0
    return sorted(rankable(units), key=lambda i: -sign * values[i])


# --------------------------------------------------------------------------- faithfulness

def _class_prob(p_real, real: bool):
    p_real = np.asarray(p_real, float)
    return p_real if real else 1 - p_real


def faithfulness(units: list[str], ranking: list[int], predict, fractions=FRACTIONS) -> dict:
    """Comprehensiveness and sufficiency (AOPC over fractions) of one ranking of text units."""
    p0 = float(predict([''.join(units)])[0])
    real = p0 >= 0.5
    n = len(ranking)
    removed, kept = [], []
    for fraction in fractions:
        top = set(ranking[:max(1, math.ceil(fraction * n))])
        removed.append(''.join(u for i, u in enumerate(units) if i not in top))
        kept.append(''.join(u for i, u in enumerate(units) if i in top))
    f0 = float(_class_prob(p0, real))
    f = _class_prob(predict(removed + kept), real)
    comp, suff = f0 - f[:len(fractions)], f0 - f[len(fractions):]
    return {'comprehensiveness': float(comp.mean()), 'sufficiency': float(suff.mean()),
            'by_fraction': {str(fr): {'comprehensiveness': float(c), 'sufficiency': float(s)}
                            for fr, c, s in zip(fractions, comp, suff)}}


def random_faithfulness(units, predict, seed: int, n_random: int = N_RANDOM, fractions=FRACTIONS) -> dict:
    rng = np.random.default_rng(seed)
    base = rankable(units)
    runs = [faithfulness(units, list(rng.permutation(base)), predict, fractions) for _ in range(n_random)]
    return {key: float(np.mean([r[key] for r in runs])) for key in ('comprehensiveness', 'sufficiency')}


def image_faithfulness(image, cam, predict_images, fractions=FRACTIONS, ranking=None) -> dict:
    """Same measures for an image (H x W x C array) split into the cam's s x s grid. Deleted
    cells take the image's mean colour. predict_images(list of arrays) -> P(real) per image.
    ranking: cell indices (row-major) to use instead of the cam order, e.g. a random one."""
    image = np.asarray(image, float)
    cam = np.asarray(cam, float)
    side_r, side_c = cam.shape
    h, w = image.shape[:2]
    cells = [(r * h // side_r, (r + 1) * h // side_r, c * w // side_c, (c + 1) * w // side_c)
             for r in range(side_r) for c in range(side_c)]
    order = list(ranking) if ranking is not None else list(np.argsort(-cam.ravel(), kind='stable'))
    fill = image.reshape(-1, image.shape[-1]).mean(axis=0) if image.ndim == 3 else image.mean()

    def masked(indices, keep):
        out = image.copy() if not keep else np.broadcast_to(fill, image.shape).copy()
        for i in indices:
            r0, r1, c0, c1 = cells[i]
            out[r0:r1, c0:c1] = image[r0:r1, c0:c1] if keep else fill
        return out
    p0 = float(predict_images([image])[0])
    real = p0 >= 0.5
    tops = [order[:max(1, math.ceil(fr * len(cells)))] for fr in fractions]
    f = _class_prob(predict_images([masked(t, False) for t in tops] + [masked(t, True) for t in tops]), real)
    f0 = float(_class_prob(p0, real))
    comp, suff = f0 - f[:len(fractions)], f0 - f[len(fractions):]
    return {'comprehensiveness': float(comp.mean()), 'sufficiency': float(suff.mean())}


# --------------------------------------------------------------------------- evidence + error detection

def sentence_evidence(row: dict) -> list[dict]:
    """premise_scores.jsonl row -> evidence items, one per sentence, stance = NLI argmax."""
    names = {'entailment': SUPPORTS, 'contradiction': REFUTES, 'neutral': UNRELATED}
    return [{'snippet': sentence, 'stance': {'label': names[max(probs, key=probs.get)]}}
            for sentence, probs in row['sentences']]


def auroc(scores, errors) -> float | None:
    return float(roc_auc_score(errors, scores)) if len(set(errors)) == 2 else None


def bootstrap_auroc(scores, errors, n: int = 1000, seed: int = 42) -> dict | None:
    scores, errors = np.asarray(scores, float), np.asarray(errors)
    point = auroc(scores, errors)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(errors), len(errors))
        if len(set(errors[i])) == 2:
            values.append(roc_auc_score(errors[i], scores[i]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {'value': point, 'low': float(low), 'high': float(high)}


def error_detection(items: list[dict], n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """H4 on items that have an EEA value: higher score = more likely a wrong prediction."""
    usable = [r for r in items if r['eea_signed'] is not None]
    errors = [int(not r['correct']) for r in usable]
    out = {'n': len(usable), 'n_errors': sum(errors), 'n_without_eea': len(items) - len(usable)}
    if len(set(errors)) < 2:
        return out
    eea_score = [1 - r['eea_signed'] for r in usable]
    conf_score = [1 - r['confidence'] for r in usable]
    x = np.column_stack([eea_score, conf_score])
    folds = min(5, min(sum(errors), len(errors) - sum(errors)))
    combined = (cross_val_predict(LogisticRegression(random_state=seed), x, errors, cv=folds,
                                  method='predict_proba')[:, 1] if folds >= 2 else None)
    out.update({
        'auroc_low_eea': bootstrap_auroc(eea_score, errors, n_bootstrap, seed),
        'auroc_low_eea_unsigned': (bootstrap_auroc([1 - r['eea_unsigned'] for r in usable], errors, n_bootstrap, seed)
                                   if all(r.get('eea_unsigned') is not None for r in usable) else None),
        'auroc_low_confidence': bootstrap_auroc(conf_score, errors, n_bootstrap, seed),
        'auroc_combined_cv': bootstrap_auroc(combined, errors, n_bootstrap, seed) if combined is not None else None,
        'mean_eea_correct': float(np.mean([r['eea_signed'] for r in usable if r['correct']])),
        'mean_eea_wrong': float(np.mean([r['eea_signed'] for r in usable if not r['correct']])),
    })
    return out


# --------------------------------------------------------------------------- run

def audit_item(row: dict, predict, seeds=SEEDS, max_evals: int = 300, k: int = TOP_K) -> dict:
    units, values = shap_attributions(row['claim'], predict, max_evals)
    p_real = float(predict([row['claim']])[0])
    y_real = int(row['liar2_label'] >= 3)
    shap_f = faithfulness(units, shap_ranking(units, values, p_real >= 0.5), predict)
    tokens = [{'token': u, 'value': v} for u, v in zip(units, values)]
    evidence = sentence_evidence(row)
    signed, unsigned = explain_audit.eea(tokens, evidence, k), explain_audit.eea(tokens, evidence, k, signed=False)
    record = {'id': row['id'], 'liar2_label': row['liar2_label'], 'y_real': y_real, 'p_real': p_real,
              'correct': (p_real >= 0.5) == bool(y_real), 'confidence': max(p_real, 1 - p_real),
              'n_units': len(rankable(units)),
              'shap_comprehensiveness': shap_f['comprehensiveness'], 'shap_sufficiency': shap_f['sufficiency'],
              'eea_signed': signed['eea'] if signed else None, 'eea_unsigned': unsigned['eea'] if unsigned else None,
              'eea_matched': ' '.join(signed['matched']) if signed else ''}
    for seed in seeds:
        rand = random_faithfulness(units, predict, seed)
        record[f'random{seed}_comprehensiveness'] = rand['comprehensiveness']
        record[f'random{seed}_sufficiency'] = rand['sufficiency']
    return record


def summarise(items: list[dict], seeds=SEEDS, n_bootstrap: int = 1000) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in items]))
    faith = {}
    for measure in ('comprehensiveness', 'sufficiency'):
        per_seed = [mean(f'random{s}_{measure}') for s in seeds]
        diff = np.array([r[f'shap_{measure}'] - r[f'random{seeds[0]}_{measure}'] for r in items])
        rng = np.random.default_rng(seeds[0])
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_bootstrap)]
        faith[measure] = {'shap': mean(f'shap_{measure}'),
                          'random_mean': float(np.mean(per_seed)),
                          'random_sd': float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0,
                          'shap_minus_random': {'value': float(diff.mean()),
                                                'low': float(np.percentile(boots, 2.5)),
                                                'high': float(np.percentile(boots, 97.5))}}
    with_eea = [r for r in items if r['eea_signed'] is not None]
    return {'n': len(items), 'accuracy': float(np.mean([r['correct'] for r in items])),
            'faithfulness': faith,
            'eea': {'n_with_value': len(with_eea),
                    'mean_signed': float(np.mean([r['eea_signed'] for r in with_eea])) if with_eea else None,
                    'mean_unsigned': float(np.mean([r['eea_unsigned'] for r in with_eea])) if with_eea else None,
                    'share_zero': float(np.mean([r['eea_signed'] == 0 for r in with_eea])) if with_eea else None},
            'error_detection': error_detection(items, n_bootstrap, seeds[0])}


def load_rows(path: Path, split: str, limit: int | None) -> list[dict]:
    from research.data import load_liar2
    claims = {e.id: e.text for e in load_liar2(split)}
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['split'] == split and row['id'] in claims and row['sentences']:
                rows.append(dict(row, claim=claims[row['id']]))
    return rows[:limit] if limit else rows


def clip_view(image):
    """Resize the shortest side to 224 and centre-crop 224x224, as CLIPProcessor does, so the
    Grad-CAM grid and the deleted cells refer to the same pixels the model sees."""
    from PIL import Image
    w, h = image.size
    scale = 224 / min(w, h)
    image = image.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return image.crop((left, top, left + 224, top + 224))


def image_audit(n: int, seed: int = SEEDS[0], n_random: int = N_RANDOM) -> list[dict]:
    """Grad-CAM deletion test on n Fakeddit test items (placeholder images skipped), using the
    trained fusion model at MULTIMODAL_MODEL_PATH. The text is held fixed; only cells change."""
    import random
    import torch
    from PIL import Image
    from app.services.model_adapter import _multimodal_bundle, _multimodal_scores
    from research.data import load_fakeddit
    clip, processor, head, real_idx, temperature = _multimodal_bundle()
    rows = [e for e in load_fakeddit('test') if not e.meta['image_placeholder']]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    records = []
    for count, e in enumerate(rows, 1):
        image = clip_view(Image.open(e.image_path).convert('RGB'))
        p_real, _, cam = _multimodal_scores(e.text, image)
        if cam is None:
            continue
        enc = processor(text=[e.text], return_tensors='pt', truncation=True, max_length=77, padding=True)
        with torch.no_grad():
            t = clip.get_text_features(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            t = t / t.norm(dim=-1, keepdim=True)

        def predict_images(arrays, t=t):
            pixels = processor(images=[Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)) for a in arrays],
                               return_tensors='pt')['pixel_values']
            with torch.no_grad():
                i = clip.get_image_features(pixel_values=pixels)
                i = i / i.norm(dim=-1, keepdim=True)
                logits = head(t.expand(len(arrays), -1), i) / temperature
            return torch.softmax(logits, dim=-1)[:, real_idx].numpy()
        array = np.asarray(image, dtype=float)
        cam_f = image_faithfulness(array, cam, predict_images)
        rng = np.random.default_rng(seed)
        rand = [image_faithfulness(array, cam, predict_images, ranking=list(rng.permutation(cam.size)))
                for _ in range(n_random)]
        records.append({'id': e.id, 'y_real': 1 - e.binary_label, 'p_real': p_real,
                        'correct': (p_real >= 0.5) == bool(1 - e.binary_label),
                        'gradcam_comprehensiveness': cam_f['comprehensiveness'],
                        'gradcam_sufficiency': cam_f['sufficiency'],
                        'random_comprehensiveness': float(np.mean([r['comprehensiveness'] for r in rand])),
                        'random_sufficiency': float(np.mean([r['sufficiency'] for r in rand]))})
        if count % 25 == 0:
            print(f'{count}/{len(rows)} images audited', flush=True)
    return records


def summarise_images(records: list[dict], n_bootstrap: int = 1000, seed: int = SEEDS[0]) -> dict:
    out = {'n': len(records), 'accuracy': float(np.mean([r['correct'] for r in records]))}
    rng = np.random.default_rng(seed)
    for measure in ('comprehensiveness', 'sufficiency'):
        diff = np.array([r[f'gradcam_{measure}'] - r[f'random_{measure}'] for r in records])
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_bootstrap)]
        out[measure] = {'gradcam': float(np.mean([r[f'gradcam_{measure}'] for r in records])),
                        'random': float(np.mean([r[f'random_{measure}'] for r in records])),
                        'gradcam_minus_random': {'value': float(diff.mean()), 'low': float(np.percentile(boots, 2.5)),
                                                 'high': float(np.percentile(boots, 97.5))}}
    return out


def text_model(name: str):
    if name == 'bert':
        from app.services.model_adapter import _text_probs
        return _text_probs
    from research.data import load_liar2
    from research.eval_trust import tfidf_text_model
    return tfidf_text_model(load_liar2('train'), SEEDS[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scores', type=Path, default=DEFAULT_SCORES)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--text-model', choices=('tfidf', 'bert'), default='tfidf')
    parser.add_argument('--max-evals', type=int, default=300)
    parser.add_argument('--k', type=int, default=TOP_K)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--images', type=int, help='run the Grad-CAM deletion test on this many Fakeddit '
                        'test images instead (needs MULTIMODAL_MODEL_PATH)')
    parser.add_argument('--out', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.images:
        records = image_audit(args.images)
        metrics = summarise_images(records, args.bootstrap)
        out = args.out / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-images')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps({'dataset': 'fakeddit', 'split': 'test', 'n': args.images,
                                                     'model': 'MULTIMODAL_MODEL_PATH', 'fractions': list(FRACTIONS),
                                                     'n_random': N_RANDOM, 'seed': SEEDS[0]}, indent=2), encoding='utf-8')
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        with open(out / 'items.csv', 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f'Wrote {out}')
        print(json.dumps(metrics, indent=1))
        return

    rows = load_rows(args.scores, args.split, args.limit)
    predict = text_model(args.text_model)
    items = []
    for i, row in enumerate(rows, 1):
        items.append(audit_item(row, predict, max_evals=args.max_evals, k=args.k))
        if i % 50 == 0:
            print(f'{i}/{len(rows)} audited', flush=True)
    metrics = summarise(items, n_bootstrap=args.bootstrap)

    out = args.out / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    config = {'dataset': 'liar2', 'split': args.split, 'scores': str(args.scores), 'limit': args.limit,
              'text_model': 'tfidf_placeholder' if args.text_model == 'tfidf' else 'bert (TEXT_MODEL_PATH)',
              'max_evals': args.max_evals, 'k': args.k, 'fractions': list(FRACTIONS), 'seeds': list(SEEDS),
              'n_random': N_RANDOM, 'bootstrap': args.bootstrap, 'python': platform.python_version()}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with open(out / 'items.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    print(f'Wrote {out}')
    print(json.dumps({k: metrics[k] for k in ('n', 'accuracy', 'faithfulness', 'eea')}, indent=1))
    print(json.dumps(metrics['error_detection'], indent=1))


if __name__ == '__main__':
    main()
