"""Tests for research/data.py. Fixtures are tiny synthetic files written to tmp_path,
so these run without the real datasets. Smoke tests at the bottom use the real
datasets/ folder and are skipped when it is absent."""
import csv
import json
from datetime import date, datetime

import pytest

from research import data

FAKEDDIT_HEADER = ['text', 'image_path', '6_way_label', 'original_url']
LIAR2_HEADER = ['id', 'label', 'statement', 'date', 'subject', 'speaker', 'speaker_description', 'state_info',
                'true_counts', 'mostly_true_counts', 'half_true_counts', 'mostly_false_counts', 'false_counts',
                'pants_on_fire_counts', 'context', 'justification']


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def root(tmp_path):
    fk = tmp_path / 'fakeddit_9k'
    (fk / 'image_cache').mkdir(parents=True)
    for name in ('a.jpg', 'b.jpg'):
        (fk / 'image_cache' / name).write_bytes(b'x')
    write_csv(fk / 'train_balanced.csv', FAKEDDIT_HEADER, [
        ['real photo', 'drive/MyDrive/btp2/image_cache/a.jpg', '0', 'https://example.org/a'],
        ['satire post', 'drive/MyDrive/btp2/image_cache/b.jpg', '1', 'https://example.org/b'],
        ['same image again', 'drive/MyDrive/btp2/image_cache/b.jpg', '5', ''],
    ])
    write_csv(fk / 'val_balanced.csv', FAKEDDIT_HEADER, [
        ['no image file', 'drive/MyDrive/btp2/image_cache/missing.jpg', '0', ''],
    ])
    counts = ['0'] * 6
    write_csv(tmp_path / 'liar2' / 'train.csv', LIAR2_HEADER, [
        ['1', str(label), f'claim {label}', 'October 2, 2017', 'guns', 'someone', '', 'ohio', *counts, 'a tweet',
         'Some reasoning � here. We rate this .'] for label in range(6)
    ] + [['9', '1', 'bad date', 'not a date', '', '', '', '', *counts, '', '']])
    lb = tmp_path / 'liar_binary_ukplab'
    lb.mkdir()
    (lb / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'text': 'true one', 'label_text': 'true statement', 'labels': 0, 'context': 'a speech'},
        {'text': 'false one', 'label_text': 'false statement', 'labels': 1, 'context': 'a mailer'},
    ]) + '\n', encoding='utf-8')
    write_sinhala_xlsx(tmp_path / 'sinhala_lirneasia' / 'Corpus.xlsx')
    return tmp_path


def write_sinhala_xlsx(path):
    from openpyxl import Workbook
    path.parent.mkdir(parents=True)
    wb = Workbook()
    ws = wb.active
    ws.append([None, 'X1', 'domain', 'datestamp', 'type', 'content'])   # real file has an unnamed first column
    labels = ['CREDIBLE'] * 20 + ['UNCERTAIN'] * 40 + ['PARTIAL'] * 10 + [False] * 10   # FALSE stored as a boolean
    for i, label in enumerate(labels):
        ws.append([i, 1000 + i, 'example.lk', datetime(2020, 10, 1), label, f'සිංහල පාඨය {i}'])
    wb.save(path)


# --------------------------------------------------------------------------- Fakeddit

def test_fakeddit_uses_only_image_file_name(root):
    rows = data.load_fakeddit('train', root=root)
    assert [e.image_path for e in rows] == [root / 'fakeddit_9k' / 'image_cache' / n for n in ('a.jpg', 'b.jpg', 'b.jpg')]
    assert all(e.image_path.is_file() for e in rows)


def test_fakeddit_labels_and_unique_ids(root):
    rows = data.load_fakeddit('train', root=root)
    assert [(e.label, e.label_name, e.binary_label) for e in rows] == [
        (0, 'true', data.REAL), (1, 'satire', data.MISLEADING), (5, 'manipulated_content', data.MISLEADING)]
    assert len({e.id for e in rows}) == len(rows)   # shared images must not collide


def test_fakeddit_missing_image_raises_unless_disabled(root):
    with pytest.raises(FileNotFoundError, match='1 Fakeddit images missing'):
        data.load_fakeddit('val', root=root)
    assert len(data.load_fakeddit('val', root=root, check_images=False)) == 1


def test_bad_split_name(root):
    with pytest.raises(ValueError):
        data.load_fakeddit('validation', root=root)


def test_fakeddit_drops_cross_split_overlap_keeping_ids(tmp_path):
    fk = tmp_path / 'fakeddit_9k'
    (fk / 'image_cache').mkdir(parents=True)
    for name in ('a.jpg', 'b.jpg', 'c.jpg', 'd.jpg', 'e.jpg'):
        (fk / 'image_cache' / name).write_bytes(name.encode())
    img = 'drive/MyDrive/btp2/image_cache/'
    write_csv(fk / 'train_balanced.csv', FAKEDDIT_HEADER, [
        ['keep me', img + 'a.jpg', '0', ''],
        ['shares image with val', img + 'b.jpg', '1', ''],
        ['  SAME Text  as test', img + 'a.jpg', '2', ''],          # text match ignores case/spacing
        ['also kept', img + 'e.jpg', '0', ''],
    ])
    write_csv(fk / 'val_balanced.csv', FAKEDDIT_HEADER, [
        ['val one', img + 'b.jpg', '0', ''], ['shares image with test', img + 'd.jpg', '1', '']])
    write_csv(fk / 'test_balanced.csv', FAKEDDIT_HEADER, [
        ['same text as test', img + 'c.jpg', '0', ''], ['test two', img + 'd.jpg', '3', '']])
    train = data.load_fakeddit('train', root=tmp_path)
    assert [e.id for e in train] == ['fakeddit-train-0', 'fakeddit-train-3']
    assert [e.id for e in data.load_fakeddit('val', root=tmp_path)] == ['fakeddit-val-0']
    assert len(data.load_fakeddit('test', root=tmp_path)) == 2
    assert len(data.load_fakeddit('train', root=tmp_path, drop_overlap=False)) == 4


def test_fakeddit_official_keeps_downloaded_images_and_drops_overlap(tmp_path):
    header = ['id', 'clean_title', 'subreddit', 'domain', 'created_utc', '6_way_label', 'image_url', 'image_status']
    fo = tmp_path / 'fakeddit_official'
    write_csv(fo / 'sample_train.csv', header, [
        ['a1', 'Cat wins election', 'theonion', '', '1560491936.0', '1', 'http://x/1.jpg', 'ok'],
        ['a2', 'dog found', 'news', 'abc.com', '1560491936.0', '0', 'http://x/2.jpg', 'http_404'],   # no image
        ['a3', 'shared  TITLE', 'news', '', '1560491936.0', '0', 'http://x/3.jpg', 'ok'],           # title in test
        ['a4', 'other', 'news', '', '1560491936.0', '0', 'http://x/9.jpg', 'ok']])                  # image in val
    write_csv(fo / 'sample_val.csv', header, [['v1', 'val row', 'news', '', '', '0', 'http://x/9.jpg', 'ok']])
    write_csv(fo / 'sample_test.csv', header, [['t1', 'Shared title', 'news', '', '', '4', 'http://x/5.jpg', 'ok']])
    train = data.load_fakeddit_official('train', root=tmp_path)
    assert [e.id for e in train] == ['fakeddit_official-a1']
    e = train[0]
    assert e.binary_label == 1 and e.label_name == 'satire' and e.meta['subreddit'] == 'theonion'
    assert e.date == date(2019, 6, 14) and e.image_path == fo / 'images' / 'a1.jpg'
    assert len(data.load_fakeddit_official('train', root=tmp_path, drop_overlap=False)) == 3
    assert data.load_fakeddit_official('test', root=tmp_path)[0].binary_label == 1


def test_fakeddit_flags_placeholder_image(tmp_path, monkeypatch):
    fk = tmp_path / 'fakeddit_9k'
    (fk / 'image_cache').mkdir(parents=True)
    (fk / 'image_cache' / 'p.jpg').write_bytes(b'placeholder bytes')
    (fk / 'image_cache' / 'q.jpg').write_bytes(b'real photo')
    write_csv(fk / 'train_balanced.csv', FAKEDDIT_HEADER, [
        ['a', 'drive/MyDrive/btp2/image_cache/p.jpg', '0', ''], ['b', 'drive/MyDrive/btp2/image_cache/q.jpg', '0', '']])
    import hashlib
    monkeypatch.setattr(data, 'FAKEDDIT_PLACEHOLDER_MD5', {hashlib.md5(b'placeholder bytes').hexdigest()})
    assert [e.meta['image_placeholder'] for e in data.load_fakeddit('train', root=tmp_path)] == [True, False]


# --------------------------------------------------------------------------- LIAR2

def test_liar2_binary_mapping_default_half_true_is_real(root):
    rows = data.load_liar2('train', root=root)[:6]
    assert [e.binary_label for e in rows] == [1, 1, 1, 0, 0, 0]
    assert [e.label_name for e in rows] == ['pants_fire', 'false', 'barely_true', 'half_true', 'mostly_true', 'true']


@pytest.mark.parametrize('option, expected', [('drop', None), ('misleading', 1)])
def test_liar2_half_true_options(root, option, expected):
    half_true = data.load_liar2('train', root=root, half_true=option)[3]
    assert half_true.binary_label == expected


def test_liar2_metadata_date_and_clean_justification(root):
    rows = data.load_liar2('train', root=root)
    assert rows[0].date == date(2017, 10, 2)
    assert rows[-1].date is None
    assert rows[0].meta['justification'] == 'Some reasoning here.'
    assert 'true_counts' not in rows[0].meta   # credit history is a leakage source, kept out


def test_clean_justification_keeps_ordinary_sentences():
    assert data.clean_justification('We rated many claims before. So we rate it False.') == 'We rated many claims before.'
    assert data.clean_justification('Figures back him. We rate his statement . We rate her  .') == 'Figures back him.'


# --------------------------------------------------------------------------- LIAR binary

def test_liar_binary(root):
    rows = data.load_liar_binary('train', root=root)
    assert [(e.text, e.binary_label, e.meta['context']) for e in rows] == [
        ('true one', 0, 'a speech'), ('false one', 1, 'a mailer')]


# --------------------------------------------------------------------------- Sinhala

def test_sinhala_reads_boolean_false_label_and_maps_uncertain_to_none(root):
    rows = data.load_sinhala(root=root)
    assert len(rows) == 80
    by_name = {e.label_name: e for e in rows}
    assert set(by_name) == {'CREDIBLE', 'UNCERTAIN', 'PARTIAL', 'FALSE'}
    assert by_name['FALSE'].binary_label == data.MISLEADING
    assert by_name['UNCERTAIN'].binary_label is None
    assert by_name['CREDIBLE'].binary_label == data.REAL
    assert rows[0].id == 'sinhala-1000' and rows[0].text == 'සිංහල පාඨය 0' and rows[0].date == date(2020, 10, 1)


def test_sinhala_split_is_stratified_disjoint_and_repeatable(root):
    rows = data.load_sinhala(root=root)
    parts = data.split_sinhala(rows, seed=42)
    ids = {name: [e.id for e in part] for name, part in parts.items()}
    assert sum(len(v) for v in ids.values()) == 80
    assert not set(ids['train']) & set(ids['val']) and not set(ids['train']) & set(ids['test'])
    assert [e.label_name for e in parts['test']].count('UNCERTAIN') == 6   # 15% of 40
    assert all(e.split == name for name, part in parts.items() for e in part)
    assert ids == {n: [e.id for e in p] for n, p in data.split_sinhala(list(reversed(rows)), seed=42).items()}
    assert ids != {n: [e.id for e in p] for n, p in data.split_sinhala(rows, seed=7).items()}


def test_load_sinhala_split_argument(root):
    # per label: round(20*.15)=3, round(40*.15)=6, round(10*.15)=2 (x2 labels)
    assert len(data.load_sinhala(root=root, split='val')) == 13


def test_split_overlap(root):
    train = data.load_fakeddit('train', root=root)
    assert data.split_overlap(train[:2], train[1:]) == {'text': 1, 'image': 1}


# --------------------------------------------------------------------------- real datasets (smoke)

real = pytest.mark.skipif(not (data.data_dir() / 'README.md').is_file(), reason='datasets/ folder not present')


@real
@pytest.mark.parametrize('split, n', [('train', 4701), ('val', 2386), ('test', 1441)])   # after drop_overlap
def test_real_fakeddit_counts(split, n):
    assert len(data.load_fakeddit(split)) == n


@real
def test_real_fakeddit_splits_share_no_text_or_image():
    splits = {s: data.load_fakeddit(s) for s in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert data.split_overlap(splits[a], splits[b]) == {'text': 0, 'image': 0}
    assert sum(e.meta['image_placeholder'] for e in splits['test']) == 7


@real
@pytest.mark.parametrize('split, n', [('train', 18369), ('val', 2297), ('test', 2296)])
def test_real_liar2_counts(split, n):
    assert len(data.load_liar2(split)) == n


@real
@pytest.mark.parametrize('split, n', [('train', 10269), ('val', 1284), ('test', 1283)])
def test_real_liar_binary_counts(split, n):
    assert len(data.load_liar_binary(split)) == n


@real
def test_real_sinhala_counts():
    rows = data.load_sinhala()
    counts = {name: sum(e.label_name == name for e in rows) for name in data.SINHALA_LABELS.values()}
    assert counts == {'CREDIBLE': 1003, 'PARTIAL': 83, 'FALSE': 27, 'UNCERTAIN': 1887}


# --------------------------------------------------------------------------- C4 evaluation sets

def liar2_row(rid, label, statement, day):
    return [str(rid), str(label), statement, day, '', '', '', '', *(['0'] * 6), '', '']


@pytest.fixture
def c4_root(tmp_path):
    rows = {'train': [liar2_row(1, 0, 'Taxes doubled last year.', 'March 3, 2010'),
                      liar2_row(2, 5, 'Jobs rose in Ohio.', 'May 1, 2012')],
            'valid': [liar2_row(3, 1, 'Crime fell by half.', 'June 2, 2016')],
            'test': [liar2_row(4, 4, 'Wages went up.', 'July 4, 2018'),            # gap year: dropped
                     liar2_row(5, 0, 'Taxes doubled last year!', 'Jan 1, 2020'),   # bad date: dropped
                     liar2_row(6, 2, 'Schools closed.', 'February 9, 2021'),
                     liar2_row(7, 0, 'TAXES doubled last year', 'March 3, 2022')]}   # same claim as train id 1
    for name, data_rows in rows.items():
        write_csv(tmp_path / 'liar2' / f'{name}.csv', LIAR2_HEADER, data_rows)
    lb = tmp_path / 'liar_binary_ukplab'
    lb.mkdir()
    (lb / 'test.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'text': 'Jane Doe says Jobs rose in Ohio.', 'labels': 0},                  # in LIAR2 -> dropped
        {'text': 'John Roe says Bridges are falling down.', 'labels': 1},
    ]) + '\n', encoding='utf-8')
    return tmp_path


def test_claim_key_strips_speaker_prefix_and_punctuation():
    assert data.claim_key('Jane Doe says Jobs rose in Ohio.') == data.claim_key('Jobs rose, in Ohio') == 'jobsroseinohio'
    long_head = 'A' * 70 + ' says x'
    assert data.claim_key(long_head) == ('a' * 70) + 'saysx'


def test_load_liar_unseen_drops_claims_in_liar2(c4_root):
    assert [e.text for e in data.load_liar_unseen('test', root=c4_root)] == ['John Roe says Bridges are falling down.']


def test_liar2_temporal_split(c4_root):
    parts = {s: data.load_liar2_temporal(s, root=c4_root) for s in data.TEMPORAL_SPLITS}
    past_ids = {e.id for s in ('train', 'val', 'test') for e in parts[s]}
    assert {e.id for e in data.load_liar2_temporal('train', root=c4_root, seed=1)} <= past_ids
    # 3 past rows: round(3 * 0.85) = 3 -> all in train, val/test empty
    assert {e.id for e in parts['train']} == {'liar2-1', 'liar2-2', 'liar2-3'} and not parts['val']
    # 2018 gap drops liar2-4, the bad date drops liar2-5, liar2-7 repeats train claim liar2-1
    assert [(e.id, e.split) for e in parts['future']] == [('liar2-6', 'future')]
    with pytest.raises(ValueError):
        data.load_liar2_temporal('later', root=c4_root)


def test_load_local_labelled(tmp_path):
    path = tmp_path / 'labelled.csv'
    write_csv(path, list(data.LOCAL_COLUMNS), [
        ['a1', 'Claim one', 'misleading', 'en', 'Factcheck.lk', 'https://x', '2024-05-01'],
        ['a2', 'Claim two', 'Uncertain', 'si', 'Factcheck.lk', '', ''],
    ])
    rows = data.load_local_labelled(path)
    assert [(e.id, e.binary_label, e.split) for e in rows] == [('srilanka-a1', 1, 'test'), ('srilanka-a2', None, 'test')]
    assert rows[0].date == date(2024, 5, 1) and rows[1].meta['language'] == 'si'
    write_csv(path, list(data.LOCAL_COLUMNS), [['a1', 't', 'fake', 'en', '', '', '']])
    with pytest.raises(ValueError, match='real, misleading or uncertain'):
        data.load_local_labelled(path)


@real
def test_real_c4_sets():
    assert (len(data.load_liar_unseen('val')), len(data.load_liar_unseen('test'))) == (308, 339)
    parts = {s: data.load_liar2_temporal(s) for s in data.TEMPORAL_SPLITS}
    assert [len(parts[s]) for s in data.TEMPORAL_SPLITS] == [11870, 1043, 1046, 7623]
    train_keys = {data.claim_key(e.text) for e in parts['train']}
    assert not any(data.claim_key(e.text) in train_keys for s in ('val', 'test', 'future') for e in parts[s])
