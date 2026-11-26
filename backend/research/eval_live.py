"""Live-evidence test for C1-C3: the app's own retrieval instead of the fact-checker's article.

The C1-C3 experiments used each LIAR2 claim's PolitiFact justification as evidence, which is an
upper bound. Here the same 500 LIAR2 test claims as C3 go through the app pipeline exactly as a
user's claim would:
  gather_evidence (Wikipedia; Tavily / Google Fact Check only when their keys are set)
  -> run_models (trained DistilBERT + SHAP) -> assess (C1 stance, C2 Trust Score, C3 EEA).
Items from the claim's own fact-checker (LEAK_DOMAINS) are dropped so the label cannot leak.
--strict also drops fact checks of the same claim by any publisher (content-word Jaccard >= 0.5
between the reviewed claim and the LIAR2 claim): their rating is close to the label itself.

Phase 1 (network) saves the retrieved evidence to evidence.jsonl and can be resumed. Phase 2
reads only that snapshot, so the analysis is reproducible after the web changes:
  python -m research.eval_live                       # fetch + analyse into a new run folder
  python -m research.eval_live --snapshot DIR        # resume / re-analyse an existing run
  python -m research.eval_live --snapshot DIR --val-scores 1000   # live s_ev for val claims (C2 refit)
  python -m research.eval_live --snapshot NEW --evidence-from DIR --strict   # re-analyse DIR's evidence, strict

Compared with the fact-check-evidence runs on the same claims:
  C1  coverage, and AUROC of s_ev for real vs misleading (live vs gold)
  C3  H4: does low signed EEA flag DistilBERT's errors? (live vs gold)
  C2  Trust Score with live s_ev vs DistilBERT alone: Brier, ECE, macro-F1, abstention at alpha
Writes results/live_evidence/<timestamp>/{config.json, evidence.jsonl, items.csv, metrics.json}.
"""
from __future__ import annotations

import os as _os

# Experiments measure the models and evidence, not the app's later overrides: keep those off whatever backend/.env says.
for _name, _value in {'JUDGE_ENABLED': 'false', 'DOMAIN_MODEL_PATH': '', 'WIKIPEDIA_INTRO': 'false'}.items():
    _os.environ[_name] = _value

import argparse
import asyncio
import csv
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
RESULTS_DIR = ROOT / 'results' / 'live_evidence'
C3_ITEMS = ROOT / 'results' / 'c3_explain' / '20261123-125219' / 'items.csv'
C1_PREDICTIONS = ROOT / 'results' / 'c1_stance' / '20261122-193507' / 'predictions.csv'
DEFAULT_ENV = {'MODEL_MODE': 'real',
               'TEXT_MODEL_PATH': str(BACKEND / 'artifacts' / 'text-distilbert-base-uncased-liar2' / 'model'),
               'TRUST_MODEL_PATH': str(ROOT / 'results' / 'c2_trust' / '20261123-111702'),
               'STANCE_MODEL': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'}
LEAK_DOMAINS = ('politifact.com',)
SAME_CLAIM_JACCARD = 0.5
STRICT = False          # set by --strict
ITEM_FIELDS = ('id', 'liar2_label', 'y_real', 'p_real', 'n_evidence', 'n_leak_dropped', 'n_decisive',
               's_ev', 'gold_s_ev', 'eea_signed', 'gold_eea_signed', 'trust_p_real', 'trust_abstained')


def is_leak(item: dict) -> bool:
    host = urlparse(item.get('url', '')).netloc.lower()
    return any(host == d or host.endswith('.' + d) for d in LEAK_DOMAINS)


def same_claim(item: dict, claim: str) -> bool:
    """A fact check whose reviewed claim (its snippet) shares >= SAME_CLAIM_JACCARD of content words with claim."""
    from app.services.explain_audit import vocabulary
    if item.get('source_type') != 'fact_check':
        return False
    a, b = vocabulary(claim), vocabulary(item.get('snippet', ''))
    return bool(a and b) and len(a & b) / len(a | b) >= SAME_CLAIM_JACCARD


def keep(item: dict, claim: str) -> bool:
    return not is_leak(item) and not (STRICT and same_claim(item, claim))


def read_csv(path: Path) -> dict[str, dict]:
    with open(path, newline='', encoding='utf-8') as handle:
        return {row['id']: row for row in csv.DictReader(handle)}


def as_float(value) -> float | None:
    return None if value in (None, '') else float(value)


def fetch(claims: list[tuple[str, str]], path: Path, delay: float = 1.0) -> None:
    """Append {id, providers, evidence} per claim to evidence.jsonl, skipping ids already there."""
    from app.services.evidence import gather_evidence
    done = set()
    if path.is_file():
        with open(path, encoding='utf-8') as handle:
            done = {json.loads(line)['id'] for line in handle if line.strip()}
    with open(path, 'a', encoding='utf-8') as handle:
        for k, (cid, claim) in enumerate(claims):
            if cid in done:
                continue
            evidence, providers = asyncio.run(gather_evidence(claim))
            handle.write(json.dumps({'id': cid, 'providers': providers, 'evidence': evidence}, ensure_ascii=False) + '\n')
            handle.flush()
            if k % 50 == 0:
                print(f'  fetched {k + 1}/{len(claims)}', flush=True)
            time.sleep(delay)                                       # be polite to the public APIs


def analyse_item(claim: str, snapshot: dict) -> dict:
    """The app pipeline (minus storage) on one claim with its saved evidence."""
    from app.services.assessment import assess
    from app.services.model_adapter import run_models
    evidence = [dict(e) for e in snapshot['evidence'] if keep(e, claim)]
    model = run_models(claim, False)
    evidence, out = assess(claim, evidence, model)
    decisive = [e for e in evidence if e.get('stance', {}).get('label') in ('supports', 'refutes')]
    audit, trust = out['explanation_audit'], out['trust']
    return {'p_real': model.text_probability_real, 'n_evidence': len(evidence),
            'n_leak_dropped': len(snapshot['evidence']) - len(evidence), 'n_decisive': len(decisive),
            's_ev': out['evidence_score'], 'eea_signed': audit['eea'] if audit else None,
            'trust_p_real': trust['probability_real'] if trust else None,
            'trust_abstained': trust['abstained'] if trust else None, 'notes': out['notes'] + model.notes}


def summarise(items: list[dict], n_bootstrap: int = 1000, seed: int = 42) -> dict:
    from research.eval_explain import bootstrap_auroc
    from research.eval_trust import brier, ece, macro_f1
    n = len(items)
    y = np.array([r['y_real'] for r in items])
    p_text = np.array([r['p_real'] for r in items])
    errors = [int((r['p_real'] >= 0.5) != r['y_real']) for r in items]

    def stance_auroc(key, decisive_only=False):
        usable = [r for r in items if r[key] is not None and (not decisive_only or r['n_decisive'] > 0)]
        return {'n': len(usable), 'auroc': bootstrap_auroc([r[key] for r in usable], [r['y_real'] for r in usable],
                                                           n_bootstrap, seed)}

    def h4(key):
        """Same test as C3: low EEA vs low confidence vs both (5-fold CV) as warnings of DistilBERT errors."""
        from research.eval_explain import error_detection
        return error_detection([{'correct': not e, 'confidence': max(r['p_real'], 1 - r['p_real']), 'eea_signed': r[key]}
                                for r, e in zip(items, errors)], n_bootstrap, seed)

    # s_ev can be None despite a decisive item when all relevances are 0 (zero total weight)
    both = [r for r in items if r['n_decisive'] > 0 and r['s_ev'] is not None and r['gold_s_ev'] not in (None, 0.0)]
    p_trust = np.array([r['trust_p_real'] if r['trust_p_real'] is not None else r['p_real'] for r in items])
    return {
        'n': n,
        'coverage': {'any_evidence': sum(r['n_evidence'] > 0 for r in items) / n,
                     'any_decisive_stance': sum(r['n_decisive'] > 0 for r in items) / n,
                     'mean_items': float(np.mean([r['n_evidence'] for r in items])),
                     'leak_items_dropped': int(sum(r['n_leak_dropped'] for r in items))},
        'c1_stance': {'live': stance_auroc('s_ev'), 'live_decisive_only': stance_auroc('s_ev', decisive_only=True),
                      'gold_same_claims': stance_auroc('gold_s_ev'),
                      'agreement_where_both_decisive': {
                          'n': len(both),
                          'same_sign': float(np.mean([np.sign(r['s_ev']) == np.sign(r['gold_s_ev']) for r in both]))
                          if both else None}},
        'c3_h4': {'live': h4('eea_signed'), 'gold_same_claims': h4('gold_eea_signed')},
        'c2_trust': {'text_only': {'brier': brier(p_text, y), 'ece': ece(p_text, y), 'macro_f1': macro_f1(p_text, y)},
                     'trust_live': {'brier': brier(p_trust, y), 'ece': ece(p_trust, y), 'macro_f1': macro_f1(p_trust, y),
                                    'abstain_rate': float(np.mean([bool(r['trust_abstained']) for r in items])),
                                    'error_on_answered': float(np.mean([
                                        (r['trust_p_real'] >= 0.5) != r['y_real'] for r in items
                                        if r['trust_p_real'] is not None and not r['trust_abstained']] or [np.nan]))}},
    }


def stratified_sample(rows, n: int, seed: int = 42):
    """n examples with the same real/misleading share as rows (seeded)."""
    from sklearn.model_selection import train_test_split
    if n >= len(rows):
        return list(rows)
    sample, _ = train_test_split(rows, train_size=n, random_state=seed, stratify=[e.binary_label for e in rows])
    return sorted(sample, key=lambda e: e.id)


def live_s_ev(claim: str, snapshot: dict) -> float | None:
    """C1 evidence score from a claim's saved live evidence (leak domains dropped); None without evidence."""
    from app.services.stance import classify_stance, evidence_score
    evidence = [dict(e) for e in snapshot['evidence'] if keep(e, claim)]
    return evidence_score(classify_stance(claim, evidence)) if evidence else None


def write_val_scores(run_dir: Path, n_val: int, seed: int = 42) -> Path:
    """Live evidence + C1 stance for a stratified sample of LIAR2 val claims, so the Trust model (C2) can
    be fitted on live s_ev. Writes live_s_ev.csv (id, split, sentence_s_ev) for those val claims plus the
    run's test claims (s_ev from items.csv), in the format eval_trust --c1-predictions reads."""
    from research.data import load_liar2
    from research.eval_trust import C1_COLUMN
    val = stratified_sample([e for e in load_liar2('val') if e.binary_label is not None], n_val, seed)
    print(f'Fetching live evidence for {len(val)} val claims', flush=True)
    fetch([(e.id, e.text) for e in val], run_dir / 'val_evidence.jsonl')
    with open(run_dir / 'val_evidence.jsonl', encoding='utf-8') as handle:
        snapshots = {row['id']: row for row in map(json.loads, handle)}
    rows = []
    for k, e in enumerate(val):
        rows.append({'id': e.id, 'split': 'val', C1_COLUMN: live_s_ev(e.text, snapshots[e.id])})
        if k % 100 == 0:
            print(f'  stance {k + 1}/{len(val)}', flush=True)
    rows += [{'id': i, 'split': 'test', C1_COLUMN: as_float(r['s_ev'])} for i, r in read_csv(run_dir / 'items.csv').items()]
    path = run_dir / 'live_s_ev.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'split', C1_COLUMN])
        writer.writeheader()
        writer.writerows([{**r, C1_COLUMN: '' if r[C1_COLUMN] is None else r[C1_COLUMN]} for r in rows])
    return path


def resummarise(run_dir: Path, n_bootstrap: int = 1000) -> dict:
    """Recompute metrics.json from a finished run's items.csv (no network, no models)."""
    items = []
    for row in read_csv(run_dir / 'items.csv').values():
        record = {k: as_float(row[k]) for k in ITEM_FIELDS if k not in ('id', 'trust_abstained')}
        record.update(id=row['id'], y_real=int(row['y_real']), n_evidence=int(row['n_evidence']),
                      n_leak_dropped=int(row['n_leak_dropped']), n_decisive=int(row['n_decisive']),
                      trust_abstained=row['trust_abstained'] == 'True')
        items.append(record)
    metrics = summarise(items, n_bootstrap)
    if (run_dir / 'metrics.json').is_file():
        metrics['provider_status'] = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8')).get('provider_status')
    else:
        with open(run_dir / 'evidence.jsonl', encoding='utf-8') as handle:
            snapshots = [json.loads(line) for line in handle if line.strip()]
        metrics['provider_status'] = {name: sum(s['providers'].get(name) == 'connected' for s in snapshots)
                                      for name in ('wikipedia', 'tavily', 'google_factcheck')}
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', type=Path, help='existing run folder: resume fetching / re-analyse')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--from-items', action='store_true', help='with --snapshot: recompute metrics.json from items.csv')
    parser.add_argument('--val-scores', type=int, metavar='N',
                        help='with --snapshot: live s_ev for N stratified LIAR2 val claims + the test claims -> '
                             'live_s_ev.csv, for refitting the Trust model (eval_trust --c1-predictions ... --only-scored)')
    parser.add_argument('--strict', action='store_true', help='also drop fact checks of the same claim (any publisher)')
    parser.add_argument('--evidence-from', type=Path, metavar='DIR',
                        help='with --snapshot: copy DIR/evidence.jsonl and val_evidence.jsonl first (no re-fetching)')
    args = parser.parse_args()
    global STRICT
    STRICT = args.strict
    if args.evidence_from:
        import shutil
        args.snapshot.mkdir(parents=True, exist_ok=True)
        for name in ('evidence.jsonl', 'val_evidence.jsonl'):
            if (args.evidence_from / name).is_file() and not (args.snapshot / name).is_file():
                shutil.copy(args.evidence_from / name, args.snapshot / name)
    if args.from_items:
        resummarise(args.snapshot, args.bootstrap)
        return
    if args.val_scores:
        for key, value in DEFAULT_ENV.items():
            os.environ.setdefault(key, value)
        from app.config import get_settings
        get_settings.cache_clear()
        print(f'Wrote {write_val_scores(args.snapshot, args.val_scores)}')
        return
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    from app.config import get_settings
    from research.data import load_liar2
    get_settings.cache_clear()

    c3, c1 = read_csv(C3_ITEMS), read_csv(C1_PREDICTIONS)
    test = {e.id: e for e in load_liar2('test')}
    ids = [i for i in c3 if i in test][:args.limit] if args.limit else [i for i in c3 if i in test]
    out = args.snapshot or RESULTS_DIR / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    providers = {'wikipedia': True, 'tavily': bool(settings.tavily_api_key),
                 'google_factcheck': bool(settings.google_factcheck_api_key)}
    (out / 'config.json').write_text(json.dumps({
        'claims': f'LIAR2 test ids from {C3_ITEMS.relative_to(ROOT)}', 'n': len(ids), 'providers_enabled': providers,
        'leak_domains': list(LEAK_DOMAINS), 'strict_same_claim_jaccard': SAME_CLAIM_JACCARD if STRICT else None,
        'evidence_from': str(args.evidence_from) if args.evidence_from else None, 'text_model': settings.text_model_path, 'stance_model': settings.stance_model,
        'trust_model': settings.trust_model_path, 'bootstrap': args.bootstrap, 'python': platform.python_version()},
        indent=2), encoding='utf-8')

    print(f'Phase 1: retrieving evidence for {len(ids)} claims', flush=True)
    fetch([(i, test[i].text) for i in ids], out / 'evidence.jsonl')
    with open(out / 'evidence.jsonl', encoding='utf-8') as handle:
        snapshots = {row['id']: row for row in map(json.loads, handle)}

    print('Phase 2: running the app pipeline on the snapshot', flush=True)
    items = []
    for k, cid in enumerate(ids):
        e = test[cid]
        record = {'id': cid, 'liar2_label': e.label, 'y_real': 1 - e.binary_label,
                  **analyse_item(e.text, snapshots[cid]),
                  'gold_s_ev': as_float(c1.get(cid, {}).get('sentence_s_ev')),
                  'gold_eea_signed': as_float(c3[cid]['eea_signed'])}
        items.append(record)
        if k % 50 == 0:
            print(f'  analysed {k + 1}/{len(ids)}', flush=True)
    with open(out / 'items.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(items)
    metrics = summarise(items, args.bootstrap)
    metrics['provider_status'] = {name: sum(s['providers'].get(name) == 'connected' for s in snapshots.values())
                                  for name in ('wikipedia', 'tavily', 'google_factcheck')}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=1))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
