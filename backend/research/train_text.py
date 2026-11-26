"""Fine-tune the text model and write an export that follows research/MODEL_EXPORT.md.

For each seed: train on the source's train split, fit a temperature on val logits, then write
predictions (misleading_probability after T, margin before T) for every protocol target.
Model weights are saved for the first seed only. A seed whose prediction folder is complete
is skipped, so an interrupted run can be resumed with the same command.

Defaults are sized for CPU: DistilBERT (a smaller BERT), 2 epochs, dynamic padding.

Run:  cd backend && python -m research.train_text --source liar2
      (sources: liar2, liar2_temporal, fakeddit, fakeddit_official)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import date
from pathlib import Path

import numpy as np

from research.check_export import data_loader_sha256
from research.eval_generalise import PROTOCOL, load_split, local_set_available

EXPORTS_DIR = Path(__file__).resolve().parents[1] / 'artifacts'
SEEDS = (42, 43, 44)
# Files each source exports: its val split (for T) + the protocol's in-dataset and other targets.
EXTRA_TARGETS = {'liar2': [('liar2', 'val'), ('liar_unseen', 'val')],
                 'liar2_temporal': [('liar2_temporal', 'val')],
                 'fakeddit': [('fakeddit', 'val'), ('liar2', 'val')],   # every text export needs liar2_val
                 'fakeddit_official': [('fakeddit_official', 'val'), ('liar2', 'val')]}


def targets_for(source: str) -> list[tuple[str, str]]:
    spec = PROTOCOL[source]
    out = [*EXTRA_TARGETS[source], spec['in'], *spec['targets']]
    if local_set_available():
        out.append(('srilanka', 'test'))
    return list(dict.fromkeys(out))


def set_seed(seed: int) -> None:
    import torch
    from transformers import set_seed as hf_set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)


def predict_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 64) -> np.ndarray:
    """(N, 2) logits; texts are batched by length for speed and returned in input order."""
    import torch
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), 2), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                            return_tensors='pt')
            out[idx] = model(**enc).logits.float().numpy()
    return out


def train_one(source: str, seed: int, args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label={0: 'fake', 1: 'real'}, label2id={'fake': 0, 'real': 1})
    train = load_split(source, 'train')
    if args.limit:
        train = random.Random(seed).sample(train, min(args.limit, len(train)))
    texts, labels = [e.text for e in train], [1 - e.binary_label for e in train]     # 1 = real
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    rng = random.Random(seed)
    model.train()
    step, started = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=args.max_length, padding=True,
                            return_tensors='pt')
            loss = model(**enc, labels=torch.tensor([labels[i] for i in idx])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                rate = (time.time() - started) / step
                print(f'  seed {seed} epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f} '
                      f'eta {rate * (steps - step) / 60:.0f} min', flush=True)
    return model, tokenizer


def write_predictions(path: Path, rows, logits: np.ndarray, temperature: float) -> None:
    margin = logits[:, 1] - logits[:, 0]
    p_real = 1 / (1 + np.exp(-margin / temperature))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'label', 'misleading_probability', 'margin'])
        for e, p, m in zip(rows, p_real, margin):
            writer.writerow([e.id, e.binary_label, round(float(1 - p), 6), round(float(m), 6)])


def run_seed(source: str, seed: int, args, out: Path) -> dict:
    from sklearn.metrics import f1_score
    from app.services.trust import fit_temperature
    folder = out / 'predictions' / f'seed{seed}'
    meta_path = folder / 'seed_meta.json'
    targets = targets_for(source)
    if meta_path.is_file() and all((folder / f'{d}_{s}.csv').is_file() for d, s in targets) and \
            (seed != args.seeds[0] or (out / 'model' / 'config.json').is_file()):
        print(f'seed {seed}: already done, skipping')
        return json.loads(meta_path.read_text(encoding='utf-8'))
    started = time.time()
    model, tokenizer = train_one(source, seed, args)
    val = load_split(source, 'val')
    val_logits = predict_logits(model, tokenizer, [e.text for e in val], args.max_length)
    y_val = [1 - e.binary_label for e in val]
    # bounded binary fit on margins (T in [0.05, 20]); same result as softmax scaling for 2 classes
    temperature = fit_temperature((val_logits[:, 1] - val_logits[:, 0]).tolist(), y_val)
    val_f1 = float(f1_score(y_val, (val_logits[:, 1] > val_logits[:, 0]).astype(int), average='macro'))
    for dataset, split in targets:
        rows = val if (dataset, split) == (source, 'val') else load_split(dataset, split)
        logits = val_logits if rows is val else predict_logits(model, tokenizer, [e.text for e in rows], args.max_length)
        write_predictions(folder / f'{dataset}_{split}.csv', rows, logits, temperature)
    if seed == args.seeds[0]:
        model.save_pretrained(out / 'model')
        tokenizer.save_pretrained(out / 'model')
        (out / 'model' / 'calibration.json').write_text(json.dumps({'temperature': temperature}), encoding='utf-8')
    meta = {'temperature': temperature, 'val_macro_f1': round(val_f1, 4), 'minutes': round((time.time() - started) / 60, 1)}
    meta_path.write_text(json.dumps(meta), encoding='utf-8')
    print(f'seed {seed}: T={temperature} val macro-F1={val_f1:.3f} ({meta["minutes"]} min)', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=tuple(PROTOCOL), required=True)
    parser.add_argument('--base-model', default='distilbert-base-uncased')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--limit', type=int, help='train on a sample (smoke tests only)')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or EXPORTS_DIR / f'text-{args.base_model.split("/")[-1]}-{args.source}'
    out.mkdir(parents=True, exist_ok=True)
    metas = {seed: run_seed(args.source, seed, args, out) for seed in args.seeds}
    manifest = {'kind': 'text', 'base_model': args.base_model, 'trained_on': args.source, 'seeds': args.seeds,
                'weights_seed': args.seeds[0], 'temperature_by_seed': {str(s): m['temperature'] for s, m in metas.items()},
                'data_loader_sha256': data_loader_sha256(),
                'hyperparameters': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
                                    'max_length': args.max_length, 'weight_decay': 0.01, 'warmup': 0.06,
                                    'limit': args.limit},
                'val_macro_f1_by_seed': {str(s): m['val_macro_f1'] for s, m in metas.items()},
                'created': str(date.today()),
                'notes': 'Trained on CPU by research/train_text.py. DistilBERT replaces the plan\'s BERT-base for speed.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
