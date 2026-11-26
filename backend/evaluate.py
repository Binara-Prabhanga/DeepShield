"""Evaluate held-out predictions exported by a trained classifier."""

import argparse
import csv
import json
import math

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate(path, probability_column, threshold):
    labels, probabilities = [], []
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'label', probability_column}.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV requires label and {probability_column} columns')
        for row in reader:
            label = int(row['label'])
            probability = float(row[probability_column])
            if label not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'Invalid binary label or probability on line {reader.line_num}')
            labels.append(label)
            probabilities.append(probability)

    if set(labels) != {0, 1}:
        raise ValueError('Evaluation requires held-out examples from both classes')
    predictions = [int(probability >= threshold) for probability in probabilities]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0,
    )
    return {
        'examples': len(labels),
        'threshold': threshold,
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc_score(labels, probabilities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='Held-out CSV, label 1 means misleading')
    parser.add_argument('--probability-column', default='misleading_probability')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    try:
        result = evaluate(args.predictions, args.probability_column, args.threshold)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
