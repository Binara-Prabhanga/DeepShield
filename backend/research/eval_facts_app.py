"""End-to-end check of the whole app pipeline on plain true/false facts (research/judge_facts.csv).

  python -m research.eval_facts_app --split test --out ../results/facts_app

For each claim: the real text model, the app's evidence (cached by eval_judge --source facts), assess() with the
Gemini judge and the out-of-domain check, then main.decide(). Reports what the user sees: "Likely authentic" (real),
"Potentially misleading" (not real) or "Needs review", against the truth. The test half was not used to choose the
prompt or the rule. Needs JUDGE_ENABLED=true, JUDGE_VARIANT, DOMAIN_MODEL_PATH, TEXT_MODEL_PATH and a Gemini key in
backend/.env (never printed).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', type=Path, default=Path('../results/facts_app'))
    parser.add_argument('--evidence', type=Path, default=None, help='facts_evidence_<split>.jsonl from eval_judge --source facts')
    parser.add_argument('--delay', type=float, default=3.0, help='seconds between claims (Gemini free-tier per-minute limits)')
    args = parser.parse_args()
    from app.main import decide
    from app.services.assessment import assess
    from app.services.model_adapter import run_models

    evidence_path = args.evidence or Path(f'../results/judge/facts_evidence_{args.split}.jsonl')
    evidence = {r['id']: r['evidence'] for r in map(json.loads, open(evidence_path, encoding='utf-8'))}
    rows = [r for r in csv.DictReader(open(Path(__file__).with_name('judge_facts.csv'), encoding='utf-8')) if r['split'] == args.split]
    out, table = [], {}
    for r in rows:
        model = run_models(r['claim'], False)
        _, assessment = assess(r['claim'], evidence[r['id']], model)
        score, label = decide(round(model.fused_probability_real * 100), assessment)
        truth = 'real' if r['real'] == '1' else 'not real'
        table[(truth, label)] = table.get((truth, label), 0) + 1
        time.sleep(args.delay)
        out.append({'id': r['id'], 'claim': r['claim'], 'truth': truth, 'label': label, 'score': score,
                    'judge': (assessment['judge'] or {}).get('verdict'), 'out_of_domain': assessment['out_of_domain'],
                    'text_p_real': round(model.text_probability_real, 3)})
    right = sum(v for (t, l), v in table.items() if (t, l) in (('real', 'Likely authentic'), ('not real', 'Potentially misleading')))
    wrong = sum(v for (t, l), v in table.items() if (t, l) in (('real', 'Potentially misleading'), ('not real', 'Likely authentic')))
    summary = {'split': args.split, 'n': len(out), 'right': right, 'wrong': wrong, 'needs_review': len(out) - right - wrong,
               'table': {f'{t} -> {l}': v for (t, l), v in sorted(table.items())}}
    run = args.out / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(run / 'predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(json.dumps(summary, indent=2))
    print('Wrote', run)


if __name__ == '__main__':
    main()
