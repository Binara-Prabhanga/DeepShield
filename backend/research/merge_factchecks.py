"""Build a 'Wikipedia + Google Fact Check' live-evidence snapshot without re-querying Wikipedia.

Why: in the 2026-11-24 fact-check run, Wikipedia answered for only 169/500 claims (rate limiting),
so that run's evidence was not comparable with the Wikipedia-only run (20261123-133939). This takes
the Wikipedia items from the earlier run and the fact-check items from the new fetch, then ranks
them with the app's own rank_evidence (fact checks first, top 8), exactly as gather_evidence would.

  python -m research.merge_factchecks --wiki-run ../results/live_evidence/20261123-133939 \
      --factchecks ../results/live_evidence/20261124-factcheck-raw/evidence.jsonl \
      --out ../results/live_evidence/20261124-factcheck-all

Test claims: fact-check items come from --factchecks. Val claims (the wiki run's val_evidence.jsonl):
fact checks are fetched here from Google only (resumable: <out>/val_factchecks.jsonl).
Then run eval_live --snapshot <out> (and --val-scores 1000); it re-uses the merged files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, encoding='utf-8') as handle:
        return {row['id']: row for row in map(json.loads, handle) if row}


def merge_one(claim: str, wiki: dict, facts: dict | None) -> dict:
    """{id, providers, evidence}: wiki's non-fact-check items + facts' fact-check items, ranked as in the app."""
    from app.services.evidence import rank_evidence
    items = [dict(e) for e in wiki['evidence'] if e.get('source_type') != 'fact_check']
    items += [dict(e) for e in (facts or {}).get('evidence', []) if e.get('source_type') == 'fact_check']
    providers = dict(wiki.get('providers', {}))
    providers['google_factcheck'] = (facts or {}).get('providers', {}).get('google_factcheck', 'missing')
    return {'id': wiki['id'], 'providers': providers, 'evidence': rank_evidence(claim, items)}


def fetch_fact_checks(claims: list[tuple[str, str]], path: Path, delay: float = 0.3) -> None:
    """Google Fact Check only, appended to path as {id, providers, evidence}; skips ids already there."""
    from app.services.evidence import google_fact_checks
    done = set(read_jsonl(path))
    with open(path, 'a', encoding='utf-8') as handle:
        for k, (cid, claim) in enumerate(claims):
            if cid in done:
                continue
            items, status = asyncio.run(google_fact_checks(claim))
            handle.write(json.dumps({'id': cid, 'providers': {'google_factcheck': status}, 'evidence': items},
                                    ensure_ascii=False) + '\n')
            handle.flush()
            if k % 100 == 0:
                print(f'  fact checks {k + 1}/{len(claims)}', flush=True)
            time.sleep(delay)


def write_merged(path: Path, texts: dict[str, str], wiki: dict[str, dict], facts: dict[str, dict]) -> int:
    with open(path, 'w', encoding='utf-8') as handle:
        for cid, row in wiki.items():
            handle.write(json.dumps(merge_one(texts[cid], row, facts.get(cid)), ensure_ascii=False) + '\n')
    return len(wiki)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wiki-run', type=Path, required=True)
    parser.add_argument('--factchecks', type=Path, required=True, help='evidence.jsonl of the fact-check fetch (test claims)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from research.data import load_liar2
    args.out.mkdir(parents=True, exist_ok=True)
    texts = {e.id: e.text for split in ('test', 'val') for e in load_liar2(split)}

    n = write_merged(args.out / 'evidence.jsonl', texts, read_jsonl(args.wiki_run / 'evidence.jsonl'),
                     read_jsonl(args.factchecks))
    print(f'test: merged {n} claims', flush=True)

    val_wiki = read_jsonl(args.wiki_run / 'val_evidence.jsonl')
    if val_wiki:
        print(f'val: fetching fact checks for {len(val_wiki)} claims', flush=True)
        fetch_fact_checks([(cid, texts[cid]) for cid in val_wiki], args.out / 'val_factchecks.jsonl')
        n = write_merged(args.out / 'val_evidence.jsonl', texts, val_wiki, read_jsonl(args.out / 'val_factchecks.jsonl'))
        print(f'val: merged {n} claims', flush=True)
    (args.out / 'merge_config.json').write_text(json.dumps({
        'wikipedia_from': str(args.wiki_run), 'test_factchecks_from': str(args.factchecks),
        'val_factchecks': 'fetched by research/merge_factchecks.py', 'ranking': 'app.services.evidence.rank_evidence'},
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
