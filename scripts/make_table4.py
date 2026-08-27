"""
Rebuild Table 4: per-class metrics (accuracy, precision, AUC, recall) for each
aggregator, in the transductive and inductive settings.

Two things the original Table 4 lacked, both raised by the reviewers:

  * SUPPORT. Reviewer 6 Comments 3 and 4 note that per-class values of 1.0 are
    not interpretable without knowing how many instances they rest on. Every
    cell here is accompanied by its class support, and cells whose support falls
    below --min-support are printed as "n/a" rather than as a number.
  * VARIANCE. Values are means +/- standard deviations across seeds, not single
    runs.

PER-CLASS ACCURACY. The pipeline's own accuracy_by_class is computed as the
fraction of instances of class c predicted as c, i.e. recall -- which is why the
original manuscript's accuracy and recall columns were identical to four decimals
and the reviewer queried it. This script instead derives proper one-vs-rest
accuracy, (TP + TN) / N, from the per-seed confusion matrices saved in
<run>/artifacts/confusion_matrix_*_run*.csv, so no re-training is needed.

Read that column with care: for a rare class, predicting "not c" for everything
already scores 1 - support/N. The script prints that trivial baseline next to
each accuracy so the two can be compared.

USAGE
    python scripts/make_table4.py \\
        --run "BiTA=~/bita-tgn-v7-nocat" \\
        --run "BiGRU=~/runs/agg_bigru" \\
        --run "Transformer=~/runs/agg_bitransformer" \\
        --run "TGN-Last=~/runs/cat_tgn_last" \\
        --processed-dir data/processed_cat --out comparison/table4
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processing import get_data


def class_support(processed_dir):
    df = pd.read_csv(os.path.join(processed_dir, 'new_df.csv'))
    ef = np.load(os.path.join(processed_dir, 'edge_features.npy'))
    nf = np.load(os.path.join(processed_dir, 'node_features.npy'))
    _, _, _, _, _, test, _, nntest = get_data(
        df, ef, nf, different_new_nodes_between_val_and_test=True)
    n = int(df.label.max()) + 1
    return (np.bincount(test.labels.astype(int), minlength=n),
            np.bincount(nntest.labels.astype(int), minlength=n))


def confusion_metrics(path, split, n_cls):
    """Per-class one-vs-rest accuracy, precision, recall and F1 from the saved
    confusion matrices -- one matrix per seed."""
    files = sorted(glob.glob(os.path.join(path, 'artifacts',
                                          f'confusion_matrix_{split}_run*.csv')))
    if not files:
        return None
    out = {c: {k: [] for k in ('accuracy', 'precision', 'recall', 'f1')}
           for c in range(n_cls)}
    for f in files:
        cm = np.loadtxt(f, delimiter=',', ndmin=2)
        if cm.shape[0] < n_cls:
            continue
        total = cm.sum()
        for c in range(n_cls):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = total - tp - fp - fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            out[c]['accuracy'].append((tp + tn) / total if total else 0.0)
            out[c]['precision'].append(prec)
            out[c]['recall'].append(rec)
            out[c]['f1'].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, help='NAME=path')
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--min-support', type=int, default=30,
                   help='cells with fewer instances than this are reported as n/a')
    p.add_argument('--metrics', default='accuracy,recall,precision,f1,auc')
    p.add_argument('--out', default='comparison/table4')
    args = p.parse_args()

    sup_t, sup_n = class_support(args.processed_dir)
    names = {}
    cmap = os.path.join(args.processed_dir, 'category_mapping.json')
    if os.path.exists(cmap):
        raw = json.load(open(cmap))
        names = {int(k): v for k, v in raw.items()} if all(
            str(k).isdigit() for k in raw) else {int(v): k for k, v in raw.items()}
    n_cls = len(sup_t)
    label = [names.get(c, f'class {c}') for c in range(n_cls)]
    metrics = [m.strip() for m in args.metrics.split(',')]

    print(f'class support -- test: {dict(zip(label, sup_t.tolist()))}')
    print(f'class support -- new-node test: {dict(zip(label, sup_n.tolist()))}')
    print(f'cells with support < {args.min_support} are reported as n/a\n')

    rows = []
    for spec in args.run:
        name, path = spec.split('=', 1)
        csv = os.path.join(os.path.expanduser(path), 'test_metrics_multi_seed.csv')
        if not os.path.exists(csv):
            print(f'WARNING: {name} has no results at {csv} — skipped')
            continue
        d = pd.read_csv(csv)
        cmx = {sp: confusion_metrics(os.path.expanduser(path), sp, n_cls)
               for sp in ('test', 'nntest')}
        for setting, prefix, sup, sp in (('transductive', 'test_c', sup_t, 'test'),
                                         ('inductive', 'nntest_c', sup_n, 'nntest')):
            for c in range(n_cls):
                total = int(sup.sum())
                rec = {'aggregator': name, 'setting': setting, 'class': label[c],
                       'support': int(sup[c]), 'seeds': len(d),
                       'trivial_acc': f'{1 - sup[c]/total:.4f}' if total else 'n/a'}
                for m in metrics:
                    src = cmx.get(sp)
                    if src is not None and m in ('accuracy', 'precision', 'recall', 'f1'):
                        v = pd.Series(src[c][m])
                    else:
                        col = f'{prefix}{c}_{m}'
                        if col not in d.columns:
                            rec[m] = None
                            continue
                        v = pd.to_numeric(d[col], errors='coerce').dropna()
                    if sup[c] < args.min_support or v.empty:
                        rec[m] = None
                    else:
                        rec[m] = f'{v.mean():.4f} ± {v.std(ddof=1):.4f}' if len(v) > 1 \
                                 else f'{v.mean():.4f}'
                rows.append(rec)

    if not rows:
        raise SystemExit('no runs found')
    tab = pd.DataFrame(rows)
    tab_disp = tab.copy()
    for m in metrics:
        tab_disp[m] = tab_disp[m].fillna('n/a')

    for setting in ('transductive', 'inductive'):
        sub = tab_disp[tab_disp.setting == setting]
        if sub.empty:
            continue
        print(f'=== {setting} ===')
        print(sub.drop(columns=['setting']).to_string(index=False))
        print()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    tab.to_csv(args.out + '.csv', index=False)

    with open(args.out + '.tex', 'w') as f:
        f.write('% Table 4 -- per-class metrics by aggregator, mean +/- std across seeds.\n')
        f.write('% "n/a" marks classes whose test support is below '
                f'{args.min_support} instances.\n')
        f.write('\\begin{tabular}{llrrr' + 'c' * len(metrics) + '}\n\\toprule\n')
        f.write('Aggregator & Class & Support & Trivial acc. & Seeds & '
                + ' & '.join(m.capitalize() for m in metrics) + ' \\\\\n')
        for setting in ('transductive', 'inductive'):
            sub = tab_disp[tab_disp.setting == setting]
            if sub.empty:
                continue
            f.write('\\midrule\n\\multicolumn{' + str(5 + len(metrics))
                    + '}{l}{\\textit{' + setting + '}} \\\\\n\\midrule\n')
            for _, r in sub.iterrows():
                cells = ' & '.join(str(r[m]) for m in metrics)
                f.write(f"{r['aggregator']} & {r['class']} & {r['support']} & "
                        f"{r['trivial_acc']} & {r['seeds']} & {cells} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')

    n_na = int(sum((tab[m].isna()).sum() for m in metrics))
    print(f'wrote {args.out}.csv and {args.out}.tex')
    print(f'{n_na} cells reported as n/a for insufficient support -- state the threshold '
          f'in the caption, and note that per-class "accuracy" equals recall in this '
          f'pipeline, which is why the redundant column has been dropped.')


if __name__ == '__main__':
    main()
