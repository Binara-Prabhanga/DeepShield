"""Check a Colab model export against research/MODEL_EXPORT.md before using it.

Run:  cd backend && python -m research.check_export <export_dir>
Prints errors and warnings; exits 1 if there are errors. Needs the datasets locally, because
ids and labels are compared with research/data.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

DATASETS = ('liar2', 'liar', 'liar_unseen', 'liar2_temporal', 'fakeddit', 'fakeddit_official', 'sinhala', 'srilanka')
EVAL_SPLITS = ('val', 'test', 'future')               # 'future' only for liar2_temporal
REQUIRED = {'text': ('liar2_val', 'liar2_test'), 'multimodal': ('fakeddit_val', 'fakeddit_test')}
REQUIRED_BY_SOURCE = {'liar2_temporal': ('liar2_temporal_val', 'liar2_temporal_test', 'liar2_temporal_future'),
                      'fakeddit_official': ('fakeddit_official_val', 'fakeddit_official_test')}
MANIFEST_KEYS = ('kind', 'base_model', 'trained_on', 'seeds', 'weights_seed', 'temperature_by_seed',
                 'data_loader_sha256', 'hyperparameters', 'created')
COLUMNS = ('id', 'label', 'misleading_probability', 'margin')
_FILE = re.compile(r'^([a-z0-9_]+?)_(val|test|future)\.csv$')
TOLERANCE = 0.01
MIN_COVERAGE = 0.99

Loader = Callable[[str, str], dict[str, int]]          # (dataset, split) -> {id: binary_label}


def required_files(manifest: dict) -> tuple[str, ...]:
    """Prediction files every seed must have: the source's own val/test when it has a rule, plus
    the kind's defaults (a text model trained on fakeddit_official still needs liar2 val/test)."""
    by_source = REQUIRED_BY_SOURCE.get(manifest['trained_on'], ())
    default = REQUIRED[manifest['kind']]
    if manifest['trained_on'] == 'fakeddit_official' and manifest['kind'] == 'multimodal':
        default = ()                                                 # multimodal: the mirror files are not needed
    if manifest['trained_on'] == 'liar2_temporal':
        default = ()
    return tuple(dict.fromkeys(by_source + default))


def default_loader(dataset: str, split: str) -> dict[str, int]:
    from research.eval_generalise import load_split
    return {e.id: e.binary_label for e in load_split(dataset, split)}


def data_loader_sha256() -> str:
    return hashlib.sha256((Path(__file__).parent / 'data.py').read_bytes()).hexdigest()


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def check_manifest(manifest: dict, errors: list, warnings: list) -> None:
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        errors.append(f'manifest.json: missing keys {missing}')
        return
    if manifest['kind'] not in REQUIRED:
        errors.append(f"manifest.json: kind must be 'text' or 'multimodal', got {manifest['kind']!r}")
    if manifest['trained_on'] not in DATASETS:
        errors.append(f"manifest.json: trained_on must be one of {DATASETS}")
    seeds = manifest['seeds']
    if not isinstance(seeds, list) or len(seeds) < 3:
        errors.append('manifest.json: seeds must list at least 3 seeds')
    elif manifest['weights_seed'] not in seeds:
        errors.append('manifest.json: weights_seed must be one of seeds')
    temps = manifest['temperature_by_seed']
    for seed in seeds if isinstance(seeds, list) else []:
        t = temps.get(str(seed)) if isinstance(temps, dict) else None
        if not isinstance(t, (int, float)) or not t > 0:
            errors.append(f'manifest.json: temperature_by_seed["{seed}"] must be a positive number')
    if manifest['data_loader_sha256'] != data_loader_sha256():
        warnings.append('manifest.json: data_loader_sha256 differs from this repo\'s research/data.py; '
                        'ids, splits or labels may not match')


def check_model_dir(model_dir: Path, kind: str, errors: list) -> None:
    if not model_dir.is_dir():
        errors.append('model/: folder missing')
        return
    if kind == 'text':
        config_path = model_dir / 'config.json'
        if not config_path.is_file():
            errors.append('model/config.json: missing (use save_pretrained())')
        else:
            from app.services.model_adapter import _REAL_LABEL_NAMES
            id2label = json.loads(config_path.read_text(encoding='utf-8')).get('id2label', {})
            if not any(str(v).lower() in _REAL_LABEL_NAMES for v in id2label.values()):
                errors.append(f'model/config.json: id2label {id2label} has no real-class name {sorted(_REAL_LABEL_NAMES)}')
        calibration = model_dir / 'calibration.json'
        if calibration.is_file():
            t = json.loads(calibration.read_text(encoding='utf-8')).get('temperature')
            if not isinstance(t, (int, float)) or not t > 0:
                errors.append('model/calibration.json: temperature must be a positive number')
    elif kind == 'multimodal':
        config_path = model_dir / 'fusion_config.json'
        if not config_path.is_file():
            errors.append('model/fusion_config.json: missing')
        else:
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            for key in ('clip_model', 'hidden_dim', 'real_index', 'temperature'):
                if key not in cfg:
                    errors.append(f'model/fusion_config.json: missing {key!r}')
        if not (model_dir / 'fusion_head.pt').is_file():
            errors.append('model/fusion_head.pt: missing')


def check_predictions(path: Path, gold: dict[str, int], temperature: float, errors: list, warnings: list) -> int:
    """Validate one predictions CSV against the loader's {id: label}. Returns the row count."""
    name = f'{path.parent.name}/{path.name}'
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS) <= set(reader.fieldnames or ()):
            errors.append(f'{name}: columns must be {list(COLUMNS)}, got {reader.fieldnames}')
            return 0
        rows = list(reader)
    seen, bad = set(), []
    for line, row in enumerate(rows, 2):
        problems = []
        rid = row['id']
        if rid in seen:
            problems.append('duplicate id')
        seen.add(rid)
        if rid not in gold:
            problems.append('id not in this split (wrong split, train row, or different data.py)')
        try:
            label, p, margin = int(row['label']), float(row['misleading_probability']), float(row['margin'])
        except ValueError:
            bad.append(f'line {line}: non-numeric value')
            continue
        if rid in gold and label != gold[rid]:
            problems.append(f'label {label} but data.py says {gold[rid]}')
        if not (math.isfinite(p) and 0 <= p <= 1):
            problems.append('misleading_probability outside [0, 1]')
        elif not math.isfinite(margin):
            problems.append('margin not finite')
        elif abs((1 - p) - _sigmoid(margin / temperature)) > TOLERANCE:
            problems.append(f'1 - misleading_probability != sigmoid(margin / {temperature})')
        if problems:
            bad.append(f"line {line} ({rid}): {'; '.join(problems)}")
    if bad:
        errors.append(f'{name}: {len(bad)} bad rows, e.g. ' + ' | '.join(bad[:3]))
    coverage = len(seen & gold.keys()) / len(gold) if gold else 0.0
    if coverage < MIN_COVERAGE:
        errors.append(f'{name}: covers {coverage:.1%} of the split; export every example with a label')
    return len(rows)


def check_export(export_dir: str | Path, loader: Loader = default_loader) -> tuple[list[str], list[str]]:
    root = Path(export_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        return ['manifest.json: missing'], warnings
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    check_manifest(manifest, errors, warnings)
    if errors:
        return errors, warnings
    check_model_dir(root / 'model', manifest['kind'], errors)

    gold_cache: dict[tuple[str, str], dict[str, int]] = {}
    for seed in manifest['seeds']:
        folder = root / 'predictions' / f'seed{seed}'
        files = sorted(folder.glob('*.csv')) if folder.is_dir() else []
        names = {f.stem for f in files}
        for required in required_files(manifest):
            if required not in names:
                errors.append(f'predictions/seed{seed}/{required}.csv: required file missing')
        for path in files:
            match = _FILE.match(path.name)
            if (not match or match.group(1) not in DATASETS
                    or (match.group(2) == 'future' and match.group(1) != 'liar2_temporal')):
                errors.append(f'predictions/seed{seed}/{path.name}: name must be <dataset>_<val|test>.csv '
                              f'(or liar2_temporal_future.csv) with dataset in {DATASETS}')
                continue
            key = (match.group(1), match.group(2))
            if key not in gold_cache:
                gold_cache[key] = loader(*key)
            check_predictions(path, gold_cache[key], float(manifest['temperature_by_seed'][str(seed)]),
                              errors, warnings)
    return errors, warnings


def read_export_probs(export_dir: str | Path, dataset: str, split: str, seed: int | None = None) -> dict[str, float]:
    """{id: P(real)} from an export's predictions (default: the weights seed)."""
    root = Path(export_dir)
    if seed is None:
        seed = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['weights_seed']
    with open(root / 'predictions' / f'seed{seed}' / f'{dataset}_{split}.csv', newline='', encoding='utf-8-sig') as handle:
        return {row['id']: 1 - float(row['misleading_probability']) for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('export_dir', type=Path)
    args = parser.parse_args()
    errors, warnings = check_export(args.export_dir)
    for message in warnings:
        print('WARNING', message)
    for message in errors:
        print('ERROR  ', message)
    print(f'{len(errors)} error(s), {len(warnings)} warning(s)')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
