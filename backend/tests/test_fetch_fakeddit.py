"""Tests for research/fetch_fakeddit.py (no network)."""
import pandas as pd

from research import fetch_fakeddit as ff


def test_stratified_sample_keeps_label_shares_and_is_seeded():
    frame = pd.DataFrame({'id': [f'r{i}' for i in range(1000)], '6_way_label': [0] * 600 + [1] * 300 + [5] * 100})
    sample = ff.stratified_sample(frame, 100)
    assert sample['6_way_label'].value_counts().to_dict() == {0: 60, 1: 30, 5: 10}
    assert sample['id'].tolist() == ff.stratified_sample(frame, 100)['id'].tolist()
    assert sample['id'].is_unique


def test_classify_download():
    jpeg = b'\xff\xd8' + b'x' * 5000
    assert ff.classify_download(200, 'image/jpeg', 'https://i.imgur.com/a.jpg', jpeg) == 'ok'
    assert ff.classify_download(404, 'text/html', 'https://i.redd.it/a.jpg', b'') == 'http_404'
    assert ff.classify_download(200, 'text/html', 'https://x/a', jpeg) == 'not_image'
    assert ff.classify_download(200, 'image/png', 'https://i.imgur.com/removed.png', jpeg) == 'placeholder'
    assert ff.classify_download(200, 'image/gif', 'https://x/a.gif', b'GIF89a') == 'placeholder'   # tiny file
