"""Add predictions for a new evaluation target to an existing text export.

Only the weights seed's model is saved in <export>/model, so the new file is written for that seed
only (predictions/seed<weights_seed>/<dataset>_<split>.csv), using the saved temperature. Used for
targets created after training, such as the Sri Lankan set (srilanka/test).

Run:  cd backend && python -m research.predict_export artifacts/text-distilbert-base-uncased-liar2 [--target srilanka test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.eval_generalise import load_split
from research.train_text import predict_logits, write_predictions


def predict_target(export_dir: Path, dataset: str, split: str, max_length: int | None = None) -> Path:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    manifest = json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['kind'] != 'text':
        raise ValueError(f'{export_dir}: only text exports are supported')
    model_dir = export_dir / 'model'
    temperature = json.loads((model_dir / 'calibration.json').read_text(encoding='utf-8'))['temperature']
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    rows = load_split(dataset, split)
    max_length = max_length or manifest.get('hyperparameters', {}).get('max_length', 128)
    logits = predict_logits(model, tokenizer, [e.text for e in rows], max_length)
    path = export_dir / 'predictions' / f'seed{manifest["weights_seed"]}' / f'{dataset}_{split}.csv'
    write_predictions(path, rows, logits, temperature)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('export', type=Path, nargs='+')
    parser.add_argument('--target', nargs=2, metavar=('DATASET', 'SPLIT'), default=('srilanka', 'test'))
    args = parser.parse_args()
    for export_dir in args.export:
        print(f'Wrote {predict_target(export_dir, *args.target)}')


if __name__ == '__main__':
    main()
