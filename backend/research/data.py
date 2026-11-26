"""Loaders for the local research datasets. Read datasets/README.md first.

Every loader returns a list of Example records with one shared label convention:
binary_label 1 = misleading, 0 = real (same as evaluate.py), None = no clean mapping.

Standard library only, apart from openpyxl for the Sinhala corpus.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

DATA_DIR_ENV = 'DEEPSHIELD_DATA_DIR'
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'datasets'

MISLEADING, REAL = 1, 0

FAKEDDIT_LABELS = {0: 'true', 1: 'satire', 2: 'misleading_content', 3: 'imposter_content',
                   4: 'false_connection', 5: 'manipulated_content'}
LIAR2_LABELS = {0: 'pants_fire', 1: 'false', 2: 'barely_true', 3: 'half_true', 4: 'mostly_true', 5: 'true'}
SINHALA_LABELS = {0: 'CREDIBLE', 1: 'PARTIAL', 2: 'FALSE', 3: 'UNCERTAIN'}
_SINHALA_IDS = {name: i for i, name in SINHALA_LABELS.items()}
_SINHALA_BINARY = {'CREDIBLE': REAL, 'PARTIAL': MISLEADING, 'FALSE': MISLEADING, 'UNCERTAIN': None}

_SPLITS = ('train', 'val', 'test')
_LIAR2_FILES = {'train': 'train', 'val': 'valid', 'test': 'test'}

csv.field_size_limit(2 ** 31 - 1)  # LIAR2 justifications are long quoted fields


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int                      # dataset's own label (Fakeddit 0-5, LIAR2 0-5, LIAR 0/1, Sinhala 0-3)
    label_name: str
    binary_label: int | None        # 1 misleading, 0 real, None = excluded from binary tasks
    split: str
    image_path: Path | None = None
    date: date | None = None
    meta: dict = field(default_factory=dict, compare=False)


def data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV) or _DEFAULT_DATA_DIR)


def _check_split(split: str) -> None:
    if split not in _SPLITS:
        raise ValueError(f'split must be one of {_SPLITS}, got {split!r}')


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- Fakeddit (image + text)

# Byte-identical placeholder images that carry no content (md5 of the file).
FAKEDDIT_PLACEHOLDER_MD5 = {'aa8b3fea9508d493b9794f485285b335'}   # imgur "image no longer available"
# Later splits win: train loses rows shared with val or test, val loses rows shared with test.
_FAKEDDIT_LATER = {'train': ('val', 'test'), 'val': ('test',), 'test': ()}


def _fakeddit_image_name(row: dict) -> str:
    return Path(row['image_path'].replace('\\', '/')).name


def _fakeddit_text_key(text: str) -> str:
    return ' '.join(text.lower().split())


def _is_placeholder(image: Path) -> bool:
    return image.is_file() and hashlib.md5(image.read_bytes()).hexdigest() in FAKEDDIT_PLACEHOLDER_MD5


def load_fakeddit(split: str, root: Path | None = None, check_images: bool = True,
                  drop_overlap: bool = True) -> list[Example]:
    """fakeddit_9k (unofficial mirror: build and test only, not for paper numbers).

    image_path in the CSV starts with "drive/MyDrive/btp2/"; only the file name is used
    (no two file names have identical bytes, so names are enough to find shared images).
    Image names are not unique (some rows share an image), so ids are split + row number.

    drop_overlap: the mirror's splits share rows (same image file or same text, ignoring case
    and spacing). By default train drops rows shared with val or test (99 rows) and val drops
    rows shared with test (19 rows); test is never changed. Ids keep their original row numbers.

    meta['image_placeholder'] is True when the image is a known no-content placeholder
    (FAKEDDIT_PLACEHOLDER_MD5); treat those rows as having no image.
    """
    _check_split(split)
    folder = (root or data_dir()) / 'fakeddit_9k'
    blocked_images, blocked_texts = set(), set()
    if drop_overlap:
        for later in _FAKEDDIT_LATER[split]:
            path = folder / f'{later}_balanced.csv'
            if path.is_file():
                rows = _read_csv(path)
                blocked_images |= {_fakeddit_image_name(r) for r in rows}
                blocked_texts |= {_fakeddit_text_key(r['text']) for r in rows}
    examples, missing = [], []
    for row_number, row in enumerate(_read_csv(folder / f'{split}_balanced.csv')):
        label = int(row['6_way_label'])
        if label not in FAKEDDIT_LABELS:
            raise ValueError(f'Unknown Fakeddit label {label} on row {row_number}')
        name = _fakeddit_image_name(row)
        if name in blocked_images or _fakeddit_text_key(row['text']) in blocked_texts:
            continue
        image = folder / 'image_cache' / name
        if check_images and not image.is_file():
            missing.append(image.name)
        examples.append(Example(
            id=f'fakeddit-{split}-{row_number}', text=row['text'].strip(), label=label,
            label_name=FAKEDDIT_LABELS[label], binary_label=REAL if label == 0 else MISLEADING,
            split=split, image_path=image,
            meta={'original_url': row.get('original_url', ''), 'image_placeholder': _is_placeholder(image)}))
    if missing:
        raise FileNotFoundError(f'{len(missing)} Fakeddit images missing in {folder / "image_cache"}, e.g. {missing[:3]}')
    return examples


def load_fakeddit_official(split: str, root: Path | None = None, drop_overlap: bool = True) -> list[Example]:
    """Official Fakeddit v2.0 multimodal sample (research/fetch_fakeddit.py: stratified per official split).

    Only rows whose image downloaded (image_status 'ok') are returned; the text is clean_title.
    drop_overlap: train drops rows sharing an image URL or title (ignoring case and spacing) with
    the val/test samples, and val drops rows shared with test; test is never changed.
    meta keeps subreddit and domain: Fakeddit labels were assigned per subreddit, so a model
    can score well by recognising the subreddit's style (see the C4 shortcut probe).
    """
    _check_split(split)
    folder = (root or data_dir()) / 'fakeddit_official'
    blocked_urls, blocked_texts = set(), set()
    if drop_overlap:
        for later in _FAKEDDIT_LATER[split]:
            path = folder / f'sample_{later}.csv'
            if path.is_file():
                rows = _read_csv(path)
                blocked_urls |= {r['image_url'] for r in rows}
                blocked_texts |= {_fakeddit_text_key(r['clean_title']) for r in rows}
    examples = []
    for row in _read_csv(folder / f'sample_{split}.csv'):
        if row['image_status'] != 'ok':
            continue
        if row['image_url'] in blocked_urls or _fakeddit_text_key(row['clean_title']) in blocked_texts:
            continue
        label = int(row['6_way_label'])
        if label not in FAKEDDIT_LABELS:
            raise ValueError(f'Unknown Fakeddit label {label} for id {row["id"]}')
        stamp = float(row['created_utc']) if row.get('created_utc') else None
        examples.append(Example(
            id=f'fakeddit_official-{row["id"]}', text=row['clean_title'].strip(), label=label,
            label_name=FAKEDDIT_LABELS[label], binary_label=REAL if label == 0 else MISLEADING, split=split,
            date=datetime.fromtimestamp(stamp, timezone.utc).date() if stamp else None,
            image_path=folder / 'images' / f'{row["id"]}.jpg',
            meta={'subreddit': row['subreddit'], 'domain': row['domain'], 'image_placeholder': False}))
    return examples


# --------------------------------------------------------------------------- LIAR2 (text claims)

_RATING_SENTENCE = re.compile(r"(?:That's why |So )?we rate (?:this|it|that|the|his|her|their|its)\b[^.]*\.", re.IGNORECASE)


def clean_justification(value: str) -> str:
    """Drop broken-encoding characters and the scrubbed "We rate this ." verdict sentence."""
    value = value.replace('�', ' ')
    value = _RATING_SENTENCE.sub(' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def _liar2_binary(label: int, half_true: str) -> int | None:
    if label == 3:
        return {'real': REAL, 'misleading': MISLEADING, 'drop': None}[half_true]
    return MISLEADING if label <= 2 else REAL


def _parse_liar2_date(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), '%B %d, %Y').date()
    except ValueError:
        return None


def load_liar2(split: str, root: Path | None = None, half_true: str = 'real') -> list[Example]:
    """LIAR2. Labels: 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true.

    half_true: 'real' (default), 'misleading' or 'drop' (binary_label None).
    The speaker credit-history count columns are left out on purpose: they are a known
    leakage source in LIAR, so experiments must opt in to them explicitly.
    """
    _check_split(split)
    if half_true not in ('real', 'misleading', 'drop'):
        raise ValueError("half_true must be 'real', 'misleading' or 'drop'")
    rows = _read_csv((root or data_dir()) / 'liar2' / f'{_LIAR2_FILES[split]}.csv')
    examples = []
    for row in rows:
        label = int(row['label'])
        if label not in LIAR2_LABELS:
            raise ValueError(f'Unknown LIAR2 label {label} for id {row["id"]}')
        examples.append(Example(
            id=f'liar2-{row["id"]}', text=row['statement'].strip(), label=label, label_name=LIAR2_LABELS[label],
            binary_label=_liar2_binary(label, half_true), split=split, date=_parse_liar2_date(row['date']),
            meta={'justification': clean_justification(row.get('justification', '')),
                  'speaker': row.get('speaker', ''), 'subject': row.get('subject', ''),
                  'context': row.get('context', ''), 'state_info': row.get('state_info', '')}))
    return examples


# --------------------------------------------------------------------------- LIAR binary (UKPLab)

def load_liar_binary(split: str, root: Path | None = None) -> list[Example]:
    """Original LIAR collapsed to two classes. labels: 0 true statement, 1 false statement."""
    _check_split(split)
    examples = []
    with open((root or data_dir()) / 'liar_binary_ukplab' / f'{split}.jsonl', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row['labels'])
            if label not in (0, 1):
                raise ValueError(f'Unknown LIAR label {label} on line {line_number + 1}')
            examples.append(Example(
                id=f'liar-{split}-{line_number}', text=row['text'].strip(), label=label,
                label_name=row.get('label_text', ''), binary_label=MISLEADING if label == 1 else REAL,
                split=split, meta={'context': row.get('context', '')}))
    return examples


# --------------------------------------------------------------------------- Sinhala (LIRNEasia)

def load_sinhala(root: Path | None = None, split: str = 'all', seed: int = 42) -> list[Example]:
    """LIRNEasia Sinhala corpus from Corpus.xlsx (Corpus.csv has broken encoding).

    The corpus has no official split. split='all' returns every row; 'train'/'val'/'test'
    returns that part of split_sinhala(seed=seed).
    """
    from openpyxl import load_workbook

    path = (root or data_dir()) / 'sinhala_lirneasia' / 'Corpus.xlsx'
    workbook = load_workbook(path, read_only=True)
    try:
        rows = workbook.worksheets[0].iter_rows(values_only=True)
        header = {name: i for i, name in enumerate(next(rows)) if name}
        examples = []
        for row in rows:
            if row is None or all(v is None for v in row):
                continue
            # Excel stores the FALSE label as a boolean, so normalise via str().
            name = str(row[header['type']]).strip().upper()
            if name not in _SINHALA_BINARY:
                raise ValueError(f'Unknown Sinhala label {name!r} for X1={row[header["X1"]]}')
            stamp = row[header['datestamp']]
            examples.append(Example(
                id=f'sinhala-{row[header["X1"]]}', text=str(row[header['content']] or '').strip(),
                label=_SINHALA_IDS[name], label_name=name,
                binary_label=_SINHALA_BINARY[name], split='all',
                date=stamp.date() if isinstance(stamp, datetime) else None,
                meta={'domain': row[header['domain']] or ''}))
    finally:
        workbook.close()
    if split == 'all':
        return examples
    _check_split(split)
    return split_sinhala(examples, seed=seed)[split]


def split_sinhala(examples: list[Example], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict[str, list[Example]]:
    """Stratified train/val/test split. Same seed and same rows give the same split:
    rows are sorted by id inside each label before a seeded shuffle."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError('ratios must sum to 1')
    by_label: dict[int, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    rng = random.Random(seed)
    parts: dict[str, list[Example]] = {name: [] for name in _SPLITS}
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda e: e.id)
        rng.shuffle(group)
        n_train = round(len(group) * ratios[0])
        n_val = round(len(group) * ratios[1])
        for name, chunk in zip(_SPLITS, (group[:n_train], group[n_train:n_train + n_val], group[n_train + n_val:])):
            parts[name].extend(replace(e, split=name) for e in chunk)
    return parts


# --------------------------------------------------------------------------- C4 evaluation sets

def claim_key(text: str) -> str:
    """Key for finding the same claim across LIAR and LIAR2. UKP's LIAR prefixes claims with
    '<Speaker> says ', so a leading '... says ' (within 60 characters) is dropped on both sides;
    then only lowercase letters and digits are kept."""
    head, sep, tail = text.partition(' says ')
    if sep and len(head) <= 60:
        text = tail
    return re.sub(r'[^a-z0-9]', '', text.lower())


def load_liar_unseen(split: str, root: Path | None = None) -> list[Example]:
    """LIAR rows whose claim appears in no LIAR2 split. About 75% of LIAR is inside LIAR2, so
    this is the only fair cross-dataset LIAR test for a model trained on LIAR2."""
    seen = {claim_key(e.text) for s in _SPLITS for e in load_liar2(s, root)}
    return [e for e in load_liar_binary(split, root) if claim_key(e.text) not in seen]


TEMPORAL_SPLITS = ('train', 'val', 'test', 'future')
TEMPORAL_PAST_END = date(2018, 1, 1)        # past = claims dated before this
TEMPORAL_FUTURE_START = date(2019, 1, 1)    # future = claims dated on/after this; 2018 is a gap year


def load_liar2_temporal(split: str, root: Path | None = None, half_true: str = 'real',
                        seed: int = 42) -> list[Example]:
    """LIAR2 re-split by time (C4). All LIAR2 rows are pooled.

    past (before 2018) is shuffled with a fixed seed into train 85% / val 7.5% / test 7.5%
    (test = in-period check). 'future' is every claim from 2019 on. Claims from 2018 are
    left out as a gap. val, test and future drop claims whose claim_key() is in train.
    Note: the misleading share rises from about 45% (past) to about 80% (future).
    """
    if split not in TEMPORAL_SPLITS:
        raise ValueError(f'split must be one of {TEMPORAL_SPLITS}, got {split!r}')
    pool = [e for s in _SPLITS for e in load_liar2(s, root, half_true) if e.date]
    past = sorted((e for e in pool if e.date < TEMPORAL_PAST_END), key=lambda e: e.id)
    random.Random(seed).shuffle(past)
    n_train, n_val = round(len(past) * 0.85), round(len(past) * 0.075)
    parts = {'train': past[:n_train], 'val': past[n_train:n_train + n_val], 'test': past[n_train + n_val:],
             'future': [e for e in pool if e.date >= TEMPORAL_FUTURE_START]}
    if split != 'train':
        train_keys = {claim_key(e.text) for e in parts['train']}
        parts[split] = [e for e in parts[split] if claim_key(e.text) not in train_keys]
    return [replace(e, split=split) for e in parts[split]]


LOCAL_COLUMNS = ('id', 'text', 'label', 'language', 'source', 'url', 'date')
_LOCAL_LABELS = {'real': REAL, 'misleading': MISLEADING, 'uncertain': None}


def load_local_labelled(path: Path | None = None) -> list[Example]:
    """Sri Lankan test set (C4), datasets/srilanka_local/labelled.csv (research/collect_srilanka.py).

    Columns: id, text, label (real | misleading | uncertain), language (en | si | ta),
    source (fact-checker or outlet), url, date (YYYY-MM-DD, may be empty).
    Test only: never train on it. 'uncertain' rows get binary_label None.
    """
    path = path or data_dir() / 'srilanka_local' / 'labelled.csv'
    rows = _read_csv(path)
    if rows and set(LOCAL_COLUMNS) - set(rows[0]):
        raise ValueError(f'{path} needs columns {LOCAL_COLUMNS}')
    examples = []
    for line, row in enumerate(rows, 2):
        name = row['label'].strip().lower()
        if name not in _LOCAL_LABELS:
            raise ValueError(f'{path} line {line}: label must be real, misleading or uncertain, got {name!r}')
        examples.append(Example(
            id=f'srilanka-{row["id"].strip()}', text=row['text'].strip(), label=list(_LOCAL_LABELS).index(name),
            label_name=name, binary_label=_LOCAL_LABELS[name], split='test',
            date=date.fromisoformat(row['date']) if row['date'].strip() else None,
            meta={'language': row['language'].strip(), 'source': row['source'].strip(), 'url': row['url'].strip()}))
    if len({e.id for e in examples}) != len(examples):
        raise ValueError(f'{path}: ids must be unique')
    return examples


# --------------------------------------------------------------------------- checks

def split_overlap(a: list[Example], b: list[Example]) -> dict[str, int]:
    """Count shared texts and images between two splits (leakage check for reports)."""
    images_a = {e.image_path.name for e in a if e.image_path}
    return {'text': len({e.text for e in a} & {e.text for e in b}),
            'image': len(images_a & {e.image_path.name for e in b if e.image_path})}
