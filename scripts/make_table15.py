"""
Rebuild Table 15 (per-category performance of the ablation variants) from the
result files, with every column taken from the metric it is labelled with.

No training: it only reads each run's test_metrics_multi_seed.csv.

USAGE (from the repository root; use your own run folders)
    python make_table15.py \
        --run "GRU=runs/agg_gru" \
        --run "BiGRU=runs/agg_bigru" \
        --run "Transformer=runs/agg_bitransformer" \
        --run "GRU+Transformer=runs/agg_gru_transformer" \
        --run "BiTA=runs/warden-cat"

Prints the LaTeX rows: for each class, precision, recall and one-vs-rest AUC,
mean +- standard deviation across seeds. Per-class "accuracy" is not printed:
in this code it is the fraction of a class's instances classified correctly,
which is exactly its recall (the result files hold identical values).
"""
import argparse
import os
import sys

import pandas as pd

CLASSES = ['Anomaly.Traffic', 'Availability.DDoS', 'Availability.DoS', 'Recon.Scanning']
METRICS = ['precision', 'recall', 'auc']


def cell(s):
    return f"${s.mean():.3f} \\pm {s.std(ddof=1):.3f}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, help='NAME=run_dir')
    p.add_argument('--file', default='test_metrics_multi_seed.csv')
    a = p.parse_args()

    for spec in a.run:
        name, path = spec.split('=', 1)
        f = os.path.join(os.path.expanduser(path), a.file)
        if not os.path.exists(f):
            sys.exit(f'not found: {f}')
        df = pd.read_csv(f)
        # sanity check: per-class accuracy must equal recall
        for k in range(len(CLASSES)):
            if f'test_c{k}_acc' in df and f'test_c{k}_recall' in df:
                d = (df[f'test_c{k}_acc'] - df[f'test_c{k}_recall']).abs().max()
                if d > 1e-9:
                    print(f'% WARNING {name}: class {k} accuracy != recall (max diff {d:.3g})')
        cells = [cell(df[f'test_c{k}_{m}']) for k in range(len(CLASSES)) for m in METRICS]
        print(f"{name} & {len(df)} & " + ' & '.join(cells) + r' \\')


if __name__ == '__main__':
    main()
