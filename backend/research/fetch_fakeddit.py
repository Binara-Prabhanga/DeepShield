"""Sample the official Fakeddit multimodal splits and download the sampled images.

Source: github.com/entitize/Fakeddit, v2.0 multimodal_only_samples (Google Drive folder in the
README), saved as datasets/fakeddit_official/multimodal_{train,validate,test_public}.tsv.
The full image archive is very large, so a stratified sample (by 6_way_label, seed 42) is taken
from each official split and only those images are fetched from image_url. Images that are
gone (HTTP error, not an image, imgur's "removed" placeholder) are recorded, not replaced.

Writes datasets/fakeddit_official/sample_{train,val,test}.csv and images/<id>.jpg.
Resumable: images already on disk are kept.

Run:  cd backend && python -m research.fetch_fakeddit [--train 10000 --val 2000 --test 3000]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.data import FAKEDDIT_PLACEHOLDER_MD5, data_dir

OFFICIAL_FILES = {'train': 'multimodal_train.tsv', 'val': 'multimodal_validate.tsv', 'test': 'multimodal_test_public.tsv'}
SAMPLE_COLUMNS = ('id', 'clean_title', 'subreddit', 'domain', 'created_utc', '6_way_label', 'image_url', 'image_status')
USER_AGENT = 'research-dataset-sampler/1.0'
SEED = 42


def folder() -> Path:
    return data_dir() / 'fakeddit_official'


def stratified_sample(frame, n: int, seed: int = SEED):
    """n rows with the same 6-way label shares as the split (rounded; at least 1 per label)."""
    shares = frame['6_way_label'].value_counts(normalize=True)
    parts = [frame[frame['6_way_label'] == label].sample(min(max(1, round(n * share)), int((frame['6_way_label'] == label).sum())),
                                                         random_state=seed)
             for label, share in shares.sort_index().items()]
    import pandas as pd
    return pd.concat(parts).sort_values('id').reset_index(drop=True)


def classify_download(status_code: int, content_type: str, final_url: str, content: bytes) -> str:
    if status_code != 200:
        return f'http_{status_code}'
    if not content_type.startswith('image'):
        return 'not_image'
    if 'removed' in final_url or hashlib.md5(content).hexdigest() in FAKEDDIT_PLACEHOLDER_MD5 or len(content) < 1000:
        return 'placeholder'
    return 'ok'


def download(client, url: str, path: Path) -> str:
    if path.is_file():
        return 'ok'
    try:
        response = client.get(url)
        status = classify_download(response.status_code, response.headers.get('content-type', ''), str(response.url),
                                   response.content)
    except Exception as exc:                                        # network errors are recorded per image
        return f'error_{type(exc).__name__}'
    if status == 'ok':
        path.write_bytes(response.content)
    return status


def main():
    import httpx
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', type=int, default=10000)
    parser.add_argument('--val', type=int, default=2000)
    parser.add_argument('--test', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = folder()
    images = root / 'images'
    images.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=15, follow_redirects=True, headers={'User-Agent': USER_AGENT}) as client, \
            ThreadPoolExecutor(args.workers) as pool:
        for split, n in (('train', args.train), ('val', args.val), ('test', args.test)):
            frame = pd.read_csv(root / OFFICIAL_FILES[split], sep='\t', dtype={'id': str}, keep_default_na=False)
            frame = frame[(frame['hasImage'].astype(str) == 'True') & (frame['image_url'] != '')]
            frame['6_way_label'] = frame['6_way_label'].astype(int)
            sample = stratified_sample(frame, n)
            statuses = list(pool.map(lambda row: download(client, row[0], images / f'{row[1]}.jpg'),
                                     zip(sample['image_url'], sample['id'])))
            sample['image_status'] = statuses
            sample[list(SAMPLE_COLUMNS)].to_csv(root / f'sample_{split}.csv', index=False, quoting=csv.QUOTE_MINIMAL)
            counts = sample['image_status'].value_counts().to_dict()
            print(f'{split}: {len(sample)} sampled, image status {counts}', flush=True)


if __name__ == '__main__':
    main()
