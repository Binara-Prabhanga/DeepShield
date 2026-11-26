"""Train the CLIP fusion head on Fakeddit and write a multimodal export (research/MODEL_EXPORT.md).

CLIP (openai/clip-vit-base-patch32) stays frozen; its L2-normalised text and image embeddings
are computed once and cached. Only FusionHead from app/services/model_adapter.py is trained,
so the app loads the result unchanged (MULTIMODAL_MODEL_PATH = <export>/model).
Per seed: train with early stopping on val loss, fit a temperature on val, predict val and test.

--dataset fakeddit (default) is the unofficial mirror: for building and testing, not paper numbers.
--dataset fakeddit_official is the official v2.0 sample (research/fetch_fakeddit.py).

Run:  cd backend && python -m research.train_fusion [--dataset fakeddit_official]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date
from pathlib import Path

import numpy as np

from research.check_export import data_loader_sha256
from research.data import load_fakeddit, load_fakeddit_official

EXPORTS_DIR = Path(__file__).resolve().parents[1] / 'artifacts'
CLIP = 'openai/clip-vit-base-patch32'
SEEDS = (42, 43, 44)
LOADERS = {'fakeddit': load_fakeddit, 'fakeddit_official': load_fakeddit_official}
NOTES = {'fakeddit': 'Fakeddit mirror is unofficial: for building and testing only, not paper numbers.',
         'fakeddit_official': 'Official Fakeddit v2.0 multimodal sample (research/fetch_fakeddit.py); '
                              'rows whose image could not be downloaded are excluded.'}


def embed_split(split: str, cache_dir: Path, batch_size: int = 32, dataset: str = 'fakeddit') -> dict:
    """{'ids', 'labels_real', 'text', 'image', 'bad_images'} for a Fakeddit split, cached as .npz."""
    path = cache_dir / f'clip-vit-base-patch32_{dataset}_{split}.npz'
    rows = LOADERS[dataset](split)
    if path.is_file():
        cached = dict(np.load(path, allow_pickle=False))
        if list(cached['ids']) == [e.id for e in rows]:
            return cached
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    clip, processor = CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)
    text_emb, image_emb, bad = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for e in batch:
            try:
                images.append(Image.open(e.image_path).convert('RGB'))
            except OSError:
                bad.append(e.id)
                images.append(Image.new('RGB', (224, 224), (128, 128, 128)))
        inputs = processor(text=[e.text for e in batch], images=images, return_tensors='pt', truncation=True,
                           max_length=77, padding=True)
        with torch.no_grad():
            t = clip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            i = clip.get_image_features(pixel_values=inputs['pixel_values'])
        text_emb.append((t / t.norm(dim=-1, keepdim=True)).numpy())
        image_emb.append((i / i.norm(dim=-1, keepdim=True)).numpy())
        if (start // batch_size) % 20 == 0:
            print(f'  {split}: embedded {start + len(batch)}/{len(rows)}', flush=True)
    data = {'ids': np.array([e.id for e in rows]), 'labels_real': np.array([1 - e.binary_label for e in rows]),
            'text': np.concatenate(text_emb), 'image': np.concatenate(image_emb), 'bad_images': np.array(bad, dtype=str)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def train_head(train: dict, val: dict, seed: int, hidden: int, dropout: float, epochs: int = 40,
               batch_size: int = 64, lr: float = 1e-3, patience: int = 5):
    import torch
    from app.services.model_adapter import _fusion_head_cls
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    head = _fusion_head_cls()(train['text'].shape[1], hidden, dropout)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    tensors = {k: (torch.tensor(d['text']), torch.tensor(d['image']), torch.tensor(d['labels_real']).long())
               for k, d in (('train', train), ('val', val))}
    best, best_loss, bad_epochs = None, float('inf'), 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        t, i, y = tensors['train']
        for idx in torch.randperm(len(y), generator=generator).split(batch_size):
            opt.zero_grad()
            loss_fn(head(t[idx], i[idx]), y[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(head(*tensors['val'][:2]), tensors['val'][2]))
        if val_loss < best_loss - 1e-4:
            best, best_loss, bad_epochs = {k: v.clone() for k, v in head.state_dict().items()}, val_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    head.load_state_dict(best)
    return head.eval()


def margins(head, d: dict) -> np.ndarray:
    import torch
    with torch.no_grad():
        logits = head(torch.tensor(d['text']), torch.tensor(d['image'])).numpy()
    return logits[:, 1] - logits[:, 0]


def write_predictions(path: Path, d: dict, margin: np.ndarray, temperature: float) -> None:
    p_real = 1 / (1 + np.exp(-margin / temperature))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'label', 'misleading_probability', 'margin'])
        for rid, y_real, p, m in zip(d['ids'], d['labels_real'], p_real, margin):
            writer.writerow([rid, int(1 - y_real), round(float(1 - p), 6), round(float(m), 6)])


def main():
    import torch
    from sklearn.metrics import f1_score
    from app.services.trust import fit_temperature
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dataset', choices=tuple(LOADERS), default='fakeddit')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.out = args.out or EXPORTS_DIR / f'mm-clip-fusion-{args.dataset}'
    cache = EXPORTS_DIR / 'cache'
    splits = {s: embed_split(s, cache, dataset=args.dataset) for s in ('train', 'val', 'test')}
    temps, val_f1 = {}, {}
    for seed in args.seeds:
        head = train_head(splits['train'], splits['val'], seed, args.hidden, args.dropout)
        val_margin = margins(head, splits['val'])
        temps[str(seed)] = fit_temperature(val_margin.tolist(), splits['val']['labels_real'].tolist())
        val_f1[str(seed)] = round(float(f1_score(splits['val']['labels_real'], (val_margin > 0).astype(int),
                                                 average='macro')), 4)
        for split in ('val', 'test'):
            write_predictions(args.out / 'predictions' / f'seed{seed}' / f'{args.dataset}_{split}.csv', splits[split],
                              val_margin if split == 'val' else margins(head, splits[split]), temps[str(seed)])
        if seed == args.seeds[0]:
            (args.out / 'model').mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), args.out / 'model' / 'fusion_head.pt')
            (args.out / 'model' / 'fusion_config.json').write_text(json.dumps(
                {'clip_model': CLIP, 'hidden_dim': args.hidden, 'dropout': args.dropout, 'real_index': 1,
                 'temperature': temps[str(seed)]}, indent=2), encoding='utf-8')
        print(f'seed {seed}: T={temps[str(seed)]} val macro-F1={val_f1[str(seed)]}', flush=True)
    manifest = {'kind': 'multimodal', 'base_model': CLIP + ' (frozen) + FusionHead', 'trained_on': args.dataset,
                'seeds': args.seeds, 'weights_seed': args.seeds[0], 'temperature_by_seed': temps,
                'data_loader_sha256': data_loader_sha256(),
                'hyperparameters': {'hidden_dim': args.hidden, 'dropout': args.dropout, 'lr': 1e-3, 'batch_size': 64,
                                    'max_epochs': 40, 'patience': 5, 'weight_decay': 1e-4},
                'val_macro_f1_by_seed': val_f1, 'bad_images': {s: d['bad_images'].tolist() for s, d in splits.items()},
                'created': str(date.today()),
                'notes': NOTES[args.dataset]}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
