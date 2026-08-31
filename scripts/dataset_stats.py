"""
USAGE
    python scripts/dataset_stats.py \
        --data "Warden=data/processed_cat" \
        --data "NF-UNSW-NB15-v2=data/unsw_cat" \
        --out comparison/dataset_stats

Writes <out>.csv and <out>.tex, and prints the comparison.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processing import get_data


def stats_for(path):
    df = pd.read_csv(os.path.join(path, 'new_df.csv'))
    ef = np.load(os.path.join(path, 'edge_features.npy'))
    nf = np.load(os.path.join(path, 'node_features.npy'))
    cfg_path = os.path.join(path, 'preprocess_config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    _, _, full, train, val, test, nnval, nntest = get_data(
        df, ef, nf, different_new_nodes_between_val_and_test=True)

    src = df.u.values
    dst = df.i.values
    n_src, n_dst = len(np.unique(src)), len(np.unique(dst))
    n_nodes = len(np.unique(np.concatenate([src, dst])))
    observed_pairs = len(set(zip(src.tolist(), dst.tolist())))
    possible_pairs = n_src * n_dst

    # how much history a node actually accumulates -- the quantity a learnable
    # aggregator is supposed to exploit
    deg = pd.Series(np.concatenate([src, dst])).value_counts()

    ts = df.ts.values
    s = {
        'interactions': len(df),
        'nodes': n_nodes,
        'sources': n_src,
        'destinations': n_dst,
        'observed pairs': observed_pairs,
        'possible pairs': possible_pairs,
        'bipartite density': observed_pairs / possible_pairs,
        'interactions per node (mean)': float(deg.mean()),
        'interactions per node (median)': float(deg.median()),
        'categories': int(df.label.nunique()),
        'edge feature dim': int(ef.shape[1]),
        'node feature dim': int(nf.shape[1]),
        'timestamp span': float(ts.max() - ts.min()),
        'train interactions': len(train.sources),
        'validation interactions': len(val.sources),
        'test interactions': len(test.sources),
        'new-node validation': len(nnval.sources),
        'new-node test': len(nntest.sources),
        # nodes absent from training: NOT the same as the sampled inductive set
        # that get_data holds out (10% of nodes appearing after the validation
        # cutoff). Report the latter when describing the inductive protocol.
        'nodes absent from training': n_nodes - len(np.unique(np.concatenate(
            [train.sources, train.destinations]))),
        'nodes in new-node test split': len(np.unique(np.concatenate(
            [nntest.sources, nntest.destinations]))),
        'category channel in features': cfg.get('include_category_features', 'unknown'),
        'balanced': cfg.get('balance', 'unknown'),
    }
    s['timeline'] = cfg.get('timeline', 'real timestamps from the source data')[:60]

    # per-class counts per split
    per_class = {}
    for name, d in (('train', train), ('val', val), ('test', test), ('nn_test', nntest)):
        c = np.bincount(d.labels.astype(int), minlength=s['categories'])
        per_class[name] = c
    return s, per_class, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', action='append', required=True, help='NAME=path')
    p.add_argument('--out', default='comparison/dataset_stats')
    args = p.parse_args()

    cols, per_class_all = {}, {}
    for spec in args.data:
        name, path = spec.split('=', 1)
        path = os.path.expanduser(path)
        if not os.path.exists(os.path.join(path, 'new_df.csv')):
            print(f'skipping {name}: {path} has no new_df.csv')
            continue
        s, pc, cfg = stats_for(path)
        cols[name] = s
        per_class_all[name] = pc

    if not cols:
        raise SystemExit('no datasets found')

    table = pd.DataFrame(cols)
    fmt = table.copy()
    for r in fmt.index:
        for c in fmt.columns:
            v = fmt.loc[r, c]
            if isinstance(v, float):
                fmt.loc[r, c] = f'{v:,.4f}' if abs(v) < 1 else f'{v:,.1f}'
            elif isinstance(v, (int, np.integer)):
                fmt.loc[r, c] = f'{v:,}'
    print('\n' + fmt.to_string())

    print('\nper-class counts by split')
    for name, pc in per_class_all.items():
        print(f'\n  {name}')
        d = pd.DataFrame(pc).T
        d.columns = [f'class {i}' for i in d.columns]
        print(d.to_string())

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    table.to_csv(args.out + '.csv')
    with open(args.out + '.tex', 'w') as f:
        f.write('\\begin{tabular}{l' + 'r' * len(table.columns) + '}\n\\toprule\n')
        f.write(' & ' + ' & '.join(table.columns) + ' \\\\\n\\midrule\n')
        for r in fmt.index:
            f.write(str(r).replace('_', '\\_') + ' & '
                    + ' & '.join(str(fmt.loc[r, c]) for c in fmt.columns) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    print(f'\nWrote {args.out}.csv and {args.out}.tex')

    if len(cols) > 1:
        names = list(cols)
        print('\nStructural contrast worth stating in the manuscript:')
        for k in ('nodes', 'interactions per node (mean)', 'bipartite density'):
            vals = ' vs '.join(f'{cols[n][k]:,.4f}' if isinstance(cols[n][k], float)
                               else f'{cols[n][k]:,}' for n in names)
            print(f'  {k:32s} {vals}')
        print('  Report these as measured properties. Note in particular that a')
        print('  median degree of 1 means most nodes carry a single message, for')
        print('  which last, mean and any sequence model are identical -- so the')
        print('  aggregator can only matter for the high-degree tail. Do NOT assert')
        print('  a causal mechanism linking these properties to the results: the')
        print('  dataset where the aggregator helps has LESS per-node history, not')
        print('  more, so "needs rich history" is contradicted by the data.')


if __name__ == '__main__':
    main()
