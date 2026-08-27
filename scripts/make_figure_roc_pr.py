"""
Regenerate Figures 7 and 8: ROC and precision-recall curves for BiTA against the
baselines, in the transductive and inductive settings.

Curves from the temporal runs are averaged over seeds by interpolating onto a
common grid, so the figure carries the same statistical weight as the tables
rather than showing one arbitrary run. A shaded band gives +/- one standard
deviation across seeds; single-run baselines (the classical models) are drawn
without a band, and the legend says so.

INPUTS
  temporal runs   <run>/artifacts/curves_{test,nntest}_run*.npz
                  written automatically by run_train.py
  classical runs  <run>/curves.npz written by baselines/classical.py

USAGE
    python scripts/make_figure_roc_pr.py \\
        --run "BiTA=~/bita-tgn-v7-nocat" \\
        --run "TGN-Last=~/runs/cat_tgn_last" \\
        --classical "Random Forest=~/runs/classical/randomforest" \\
        --classical "SVM=~/runs/classical/linearsvm" \\
        --out-dir comparison
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as auc_of


def mean_roc(fprs, tprs, grid):
    ys = [np.interp(grid, f, t) for f, t in zip(fprs, tprs)]
    ys = np.stack(ys)
    return ys.mean(0), ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(ys[0])


def mean_pr(recalls, precisions, grid):
    # precision_recall_curve returns recall descending; flip for interpolation
    ys = [np.interp(grid, r[::-1], p[::-1]) for r, p in zip(recalls, precisions)]
    ys = np.stack(ys)
    return ys.mean(0), ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(ys[0])


def load_temporal(path, split):
    files = sorted(glob.glob(os.path.join(path, 'artifacts', f'curves_{split}_run*.npz')))
    if not files:
        return None
    fpr, tpr, rec, pre = [], [], [], []
    for f in files:
        z = np.load(f)
        fpr.append(z['roc_fpr']); tpr.append(z['roc_tpr'])
        pre.append(z['pr_precision']); rec.append(z['pr_recall'])
    return dict(n=len(files), fpr=fpr, tpr=tpr, precision=pre, recall=rec)


def load_classical(path, inductive):
    f = os.path.join(path, 'curves.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    pre = 'nn_' if inductive else ''
    return dict(n=1, fpr=[z[pre + 'roc_fpr']], tpr=[z[pre + 'roc_tpr']],
                precision=[z[pre + 'pr_precision']], recall=[z[pre + 'pr_recall']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', default=[], help='NAME=path (temporal run)')
    p.add_argument('--classical', action='append', default=[], help='NAME=path (classical run)')
    p.add_argument('--out-dir', default='comparison')
    p.add_argument('--inductive', action='store_true',
                   help='draw the inductive (new-node) setting instead of transductive')
    p.add_argument('--both', action='store_true',
                   help='draw transductive and inductive on the same axes')
    args = p.parse_args()

    grid = np.linspace(0, 1, 200)
    settings = ['test', 'nntest'] if args.both else (['nntest'] if args.inductive else ['test'])
    series = []
    for spec in args.run:
        name, path = spec.split('=', 1)
        for split in settings:
            d = load_temporal(os.path.expanduser(path), split)
            if d is None:
                print(f'WARNING: no curve files for {name} ({split}) — skipped')
                continue
            lbl = name + (' (new nodes)' if split == 'nntest' else '')
            series.append((lbl, d, split == 'nntest'))
    for spec in args.classical:
        name, path = spec.split('=', 1)
        for split in settings:
            d = load_classical(os.path.expanduser(path), split == 'nntest')
            if d is None:
                print(f'WARNING: no curves.npz for {name} — skipped')
                continue
            lbl = name + (' (new nodes)' if split == 'nntest' else '')
            series.append((lbl, d, split == 'nntest'))
    if not series:
        raise SystemExit('nothing to plot')

    os.makedirs(args.out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for lbl, d, ind in series:
        m, sd = mean_roc(d['fpr'], d['tpr'], grid)
        a = auc_of(grid, m)
        ls = '--' if ind else '-'
        line, = ax1.plot(grid, m, ls, lw=1.8,
                         label=f'{lbl} (AUC = {a:.3f}' + (f', n={d["n"]})' if d['n'] > 1 else ')'))
        if d['n'] > 1:
            ax1.fill_between(grid, m - sd, m + sd, alpha=.15, color=line.get_color())

        mp, sp = mean_pr(d['recall'], d['precision'], grid)
        ap = auc_of(grid, mp)
        line2, = ax2.plot(grid, mp, ls, lw=1.8,
                          label=f'{lbl} (AP = {ap:.3f}' + (f', n={d["n"]})' if d['n'] > 1 else ')'))
        if d['n'] > 1:
            ax2.fill_between(grid, mp - sp, mp + sp, alpha=.15, color=line2.get_color())

    ax1.plot([0, 1], [0, 1], 'k:', lw=1, label='random (AUC = 0.500)')
    ax1.set_xlabel('False positive rate'); ax1.set_ylabel('True positive rate')
    ax1.set_title('(a) ROC'); ax1.legend(fontsize=8, loc='lower right'); ax1.grid(alpha=.3)
    ax2.axhline(0.5, color='k', ls=':', lw=1, label='random (AP = 0.500)')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('(b) Precision-Recall'); ax2.legend(fontsize=8, loc='lower left'); ax2.grid(alpha=.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    fig.tight_layout()
    tag = 'both' if args.both else ('inductive' if args.inductive else 'transductive')
    out = os.path.join(args.out_dir, f'figure_roc_pr_{tag}.png')
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.png', '.pdf'))
    print('wrote', out, 'and the PDF version')
    print('Shaded bands are +/- 1 std across seeds; curves without a band are single runs.')
    print('AUC/AP in the legend are computed from the averaged curve, so they can differ')
    print('in the last decimal from the per-seed means in the tables — state which you quote.')


if __name__ == '__main__':
    main()
