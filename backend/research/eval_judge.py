"""Does a Gemini evidence judge beat the signals we already have? (experiment; the app does not use it yet)

  python -m research.eval_judge --snapshot ../results/live_evidence/20261124-factcheck-strict --out ../results/judge

For each LIAR2 test claim of the snapshot, Gemini (app/services/judge.py) reads the claim plus the live evidence
(PolitiFact and same-claim fact checks removed, exactly as eval_live --strict) and answers supported / refuted /
not_enough_evidence. Compared on the same claims with: the DistilBERT text model (p_real from the snapshot's items.csv)
and the NLI evidence score s_ev. Reports coverage, accuracy where it decides, AUROC with a bootstrap CI (seed 42),
and a 5-fold cross-validated logistic model of text vs text + judge. Answers are cached in <out>/<run>/judge.jsonl
(resumable with --resume <run dir>). Gemini keys come from backend/.env and are never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import numpy as np

SCORE = {'supported': 1.0, 'not_enough_evidence': 0.0, 'refuted': -1.0}


def judge_score(verdict: str | None) -> float:
    return SCORE.get(verdict or '', 0.0)


def auroc_ci(y: np.ndarray, score: np.ndarray, seed: int = 42, n_boot: int = 1000) -> dict:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    value = float(roc_auc_score(y, score))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], score[idx]))
    low, high = np.percentile(boots, [2.5, 97.5])
    return {'value': value, 'low': float(low), 'high': float(high)}


def summarise(rows: list[dict]) -> dict:
    """rows: id, y_real (1 real), p_real (or None), s_ev (or None), verdict (or None when Gemini failed)."""
    answered = [r for r in rows if r['verdict'] is not None]
    y = np.array([r['y_real'] for r in answered])
    j = np.array([judge_score(r['verdict']) for r in answered])
    decided = np.array([r['verdict'] != 'not_enough_evidence' for r in answered], dtype=bool)
    out = {'n_claims': len(rows), 'n_answered': len(answered),
           'coverage_decided': float(decided.mean()) if len(answered) else 0.0,
           'verdicts': {v: sum(r['verdict'] == v for r in answered) for v in SCORE}}
    sup, ref = j > 0, j < 0
    out['supported'] = {'n': int(sup.sum()), 'right': int(y[sup].sum())}          # claim really is real
    out['refuted'] = {'n': int(ref.sum()), 'right': int((1 - y[ref]).sum())}      # claim really is not real
    kinds = np.array([r.get('kind') or '' for r in answered])
    out['supported_by_kind'] = {k: {'n': int((sup & (kinds == k)).sum()), 'right': int(y[sup & (kinds == k)].sum())}
                                for k in sorted(set(kinds[sup]))}
    out['kinds'] = {k: int((kinds == k).sum()) for k in sorted(set(kinds))}
    if decided.any():
        out['decided'] = {'n': int(decided.sum()),
                          'judge_accuracy': float(((j[decided] > 0).astype(int) == y[decided]).mean())}
    has_text = bool(answered) and all(r['p_real'] is not None for r in answered)
    if has_text and decided.any():
        p = np.array([r['p_real'] for r in answered])
        out['decided']['text_accuracy_same_claims'] = float(((p[decided] >= 0.5).astype(int) == y[decided]).mean())
    if has_text and len(set(y)) == 2:
        p = np.array([r['p_real'] for r in answered])
        out['auroc_all_answered'] = {'judge': auroc_ci(y, j), 'text': auroc_ci(y, p)}
        s_ev = np.array([r['s_ev'] if r['s_ev'] is not None else 0.0 for r in answered])
        out['auroc_all_answered']['s_ev'] = auroc_ci(y, s_ev)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        for name, X in (('text', logit[:, None]), ('text+judge', np.c_[logit, j]), ('text+judge+s_ev', np.c_[logit, j, s_ev])):
            pred = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv, method='predict_proba')[:, 1]
            out.setdefault('cv_auroc', {})[name] = float(roc_auc_score(y, pred))
    return out


def load_rows(snapshot: Path, limit: int | None, source: str = 'test', seed: int = 42):
    """LIAR2 claims with their live evidence (PolitiFact and same-claim fact checks removed, as eval_live --strict).
    source 'test' = the 500 live test claims (items.csv: p_real, s_ev); 'val' = validation claims (no text-model score)."""
    import random
    from research import eval_live
    from research.data import load_liar2
    eval_live.STRICT = True
    examples = {e.id: e for e in load_liar2('test' if source == 'test' else 'val')}
    items = {}
    if source == 'test':
        items = {r['id']: r for r in csv.DictReader(open(snapshot / 'items.csv', encoding='utf-8'))}
    path = snapshot / ('evidence.jsonl' if source == 'test' else 'val_evidence.jsonl')
    out = []
    for line in open(path, encoding='utf-8'):
        record = json.loads(line)
        cid = record['id']
        if cid not in examples or (source == 'test' and cid not in items):
            continue
        claim = examples[cid].text
        kept = [e for e in record['evidence'] if eval_live.keep(e, claim)]
        if source == 'test':
            row = items[cid]
            out.append({'id': cid, 'claim': claim, 'evidence': kept, 'y_real': int(row['y_real']),
                        'p_real': float(row['p_real']),
                        's_ev': float(row['s_ev']) if row['s_ev'] not in ('', 'None') else None})
        elif examples[cid].binary_label is not None:
            out.append({'id': cid, 'claim': claim, 'evidence': kept, 'y_real': 1 - int(examples[cid].binary_label),
                        'p_real': None, 's_ev': None})
    if source == 'val':
        random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def load_facts(split: str, cache: Path, limit: int | None):
    """research/judge_facts.csv (plain true/false facts) with live Wikipedia + Google Fact Check evidence, cached."""
    from app.services.evidence import gather_evidence
    path = Path(__file__).with_name('judge_facts.csv')
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['split'] == split]
    cached = {}
    if cache.is_file():
        cached = {r['id']: r for r in map(json.loads, open(cache, encoding='utf-8'))}
    out = []
    with open(cache, 'a', encoding='utf-8') as handle:
        for r in (rows[:limit] if limit else rows):
            if r['id'] not in cached:
                evidence, _ = asyncio.run(gather_evidence(r['claim']))
                cached[r['id']] = {'id': r['id'], 'evidence': evidence}
                handle.write(json.dumps(cached[r['id']], ensure_ascii=False) + chr(10))
                handle.flush()
            out.append({'id': r['id'], 'claim': r['claim'], 'evidence': cached[r['id']]['evidence'],
                        'y_real': int(r['real']), 'p_real': None, 's_ev': None})
    return out


def add_intros(rows: list[dict], cache: Path) -> None:
    """Replace Wikipedia search fragments by each page's opening sentences (one MediaWiki request per claim, cached)."""
    import httpx
    cached = json.loads(cache.read_text(encoding='utf-8')) if cache.is_file() else {}
    with httpx.Client(timeout=15, headers={'User-Agent': 'DeepShield/1.0 (research; local)'}) as client:
        for row in rows:
            titles = [e['title'] for e in row['evidence'] if e.get('publisher') == 'Wikipedia'][:8]
            missing = [t for t in titles if t not in cached]
            if missing:
                try:
                    response = client.get('https://en.wikipedia.org/w/api.php', params={
                        'action': 'query', 'prop': 'extracts', 'exintro': 1, 'explaintext': 1, 'exsentences': 4,
                        'exlimit': 'max', 'titles': '|'.join(missing), 'format': 'json', 'utf8': 1})
                    response.raise_for_status()
                    for page in response.json().get('query', {}).get('pages', {}).values():
                        cached[page.get('title', '')] = page.get('extract', '')
                    for t in missing:
                        cached.setdefault(t, '')
                    time.sleep(0.2)
                except (httpx.HTTPError, ValueError):
                    pass
            for e in row['evidence']:
                if e.get('publisher') == 'Wikipedia' and cached.get(e['title']):
                    e['snippet'] = cached[e['title']][:700]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', type=Path, default=Path('../results/live_evidence/20261124-factcheck-strict'))
    parser.add_argument('--source', choices=['test', 'val', 'facts'], default='test')
    parser.add_argument('--split', default='dev', help='facts only: dev or test half of research/judge_facts.csv')
    parser.add_argument('--variant', default='strict', help='judge prompt: strict or knowledge')
    parser.add_argument('--intros', action='store_true', help='use Wikipedia opening paragraphs as evidence text')
    parser.add_argument('--out', type=Path, default=Path('../results/judge'))
    parser.add_argument('--resume', type=Path, default=None, help='an earlier run directory to continue')
    parser.add_argument('--model', default='gemini-flash-lite-latest')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    from app.services.judge import judge

    run = args.resume or args.out / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=True)
    if args.source == 'facts':
        rows = load_facts(args.split, args.out / f'facts_evidence_{args.split}.jsonl', args.limit)
    else:
        rows = load_rows(args.snapshot, args.limit, args.source)
    if args.intros:
        add_intros(rows, args.out / 'wikipedia_intros.json')
    cache_path = run / 'judge.jsonl'
    cache = {}
    if cache_path.is_file():
        cache = {r['id']: r for r in map(json.loads, open(cache_path, encoding='utf-8')) if r.get('verdict')}
    failures = 0
    with open(cache_path, 'a', encoding='utf-8') as handle:
        for k, row in enumerate(rows):
            if row['id'] in cache:
                continue
            answer = asyncio.run(judge(row['claim'], row['evidence'], retries=6, model=args.model, variant=args.variant)) if row['evidence'] else \
                {'verdict': 'not_enough_evidence', 'reason': 'no evidence retrieved'}
            if answer is None:
                failures += 1
                if failures >= 25:
                    print('Gemini keeps failing (quota or overload); stopping. Re-run with --resume', run)
                    break
                continue
            failures = 0
            cache[row['id']] = {'id': row['id'], **answer}
            handle.write(json.dumps(cache[row['id']], ensure_ascii=False) + '\n')
            handle.flush()
            if k % 25 == 0:
                print(f'  judged {k + 1}/{len(rows)}', flush=True)
            time.sleep(args.delay)
    for row in rows:
        row['verdict'] = cache.get(row['id'], {}).get('verdict')
        row['kind'] = cache.get(row['id'], {}).get('kind')
    metrics = summarise(rows)
    (run / 'config.json').write_text(json.dumps({'snapshot': str(args.snapshot), 'model': args.model, 'source': args.source, 'split': args.split,
                                                 'variant': args.variant, 'intros': args.intros, 'limit': args.limit,
                                                 'strict_same_claim': True, 'seed': 42}, indent=2), encoding='utf-8')
    (run / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print('Wrote', run)


if __name__ == '__main__':
    main()
