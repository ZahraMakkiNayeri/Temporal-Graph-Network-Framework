"""
Merge every model's test_metrics_multi_seed.csv into one comparison, and
regenerate the SOTA figure with the corrected metrics.

Every run directory must have been produced under the SAME protocol (same
processed-dir, same negative samplers, same link-ranking candidate count);
otherwise the comparison is not valid and the figure would repeat the problem
Reviewer 6 raised.

USAGE
    python scripts/collect_results.py \
        --run "BiTA=~/bita-tgn-v3" \
        --run "TGN-Last=~/runs/tgn_last" \
        --run "TGN-Mean=~/runs/tgn_mean" \
        --run "DyRep=~/runs/dyrep" \
        --run "EdgeBank=~/runs/edgebank" \
        --out-dir comparison

Writes comparison/comparison.csv (mean ± std per model), comparison.tex
(paste-ready LaTeX table) and comparison_sota.png (Figure 26 replacement).
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics worth putting in the headline comparison, in display order.
HEADLINE = [
    ('test_auc', 'AUC'),
    ('test_ap', 'AP'),
    ('test_link_f1', 'Link F1'),
    ('test_link_mrr', 'Link MRR'),
    ('test_link_hits1', 'Link Hits@1'),
    ('test_auc_historical', 'AUC (hist. neg.)'),
    ('test_ap_historical', 'AP (hist. neg.)'),
    ('test_category_accuracy', 'Category Acc.'),
    ('test_f1_macro', 'Category macro-F1'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True,
                   help='NAME=PATH (path to the run directory or the CSV itself)')
    p.add_argument('--out-dir', default='comparison')
    p.add_argument('--seeds', default=None,
                   help='restrict every run to these seeds, e.g. 0,1,2. Essential when '
                        'comparing runs with different seed counts: a 3-seed mean and a '
                        '25-seed mean are not comparable, and seed-selection effects can '
                        'dominate the differences you are trying to read.')
    p.add_argument('--figure-metrics', default='test_auc,test_ap,test_link_mrr,test_link_f1')
    args = p.parse_args()

    rows, missing = [], []
    for spec in args.run:
        assert '=' in spec, f'expected NAME=PATH, got {spec!r}'
        name, path = spec.split('=', 1)
        path = os.path.expanduser(path)
        csv = path if path.endswith('.csv') else os.path.join(path, 'test_metrics_multi_seed.csv')
        if not os.path.exists(csv):
            missing.append((name, csv))
            continue
        df = pd.read_csv(csv)
        if args.seeds:
            want = [int(x) for x in args.seeds.split(',')]
            key = 'seed' if 'seed' in df.columns else 'run'
            df = df[df[key].isin(want)]
            if df.empty:
                print(f'WARNING: {name} has none of seeds {want} — skipped')
                continue
        rec = {'model': name, 'n_seeds': len(df)}
        for col in df.columns:
            if col in ('run', 'seed', 'model'):
                continue
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals):
                rec[col] = vals.mean()
                rec[col + '__std'] = vals.std() if len(vals) > 1 else np.nan
        rows.append(rec)

    for name, csv in missing:
        print(f'WARNING: no results for {name} ({csv} not found) — skipped')
    if not rows:
        raise SystemExit('no result files found')

    table = pd.DataFrame(rows).set_index('model')
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, 'comparison.csv'))

    # ---- console + LaTeX table over the headline metrics ------------------
    present = [(c, lbl) for c, lbl in HEADLINE if c in table.columns]
    disp = pd.DataFrame(index=table.index)
    for col, label in present:
        std = table.get(col + '__std')
        disp[label] = [
            f'{m:.4f}' if (std is None or pd.isna(std.iloc[k])) else f'{m:.4f} ± {std.iloc[k]:.4f}'
            for k, m in enumerate(table[col])
        ]
    disp.insert(0, 'seeds', table['n_seeds'].values)
    print('\n' + disp.to_string())

    with open(os.path.join(args.out_dir, 'comparison.tex'), 'w') as f:
        f.write('% mean ± std over seeds; identical protocol across all rows\n')
        f.write('\\begin{tabular}{l' + 'c' * (len(present) + 1) + '}\n\\toprule\n')
        f.write('Method & Seeds & ' + ' & '.join(lbl for _, lbl in present) + ' \\\\\n\\midrule\n')
        for model in disp.index:
            cells = ' & '.join(str(disp.loc[model, lbl]) for _, lbl in present)
            f.write(f'{model} & {disp.loc[model, "seeds"]} & {cells} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    # ---- Figure 26 replacement -------------------------------------------
    fig_cols = [c for c in args.figure_metrics.split(',') if c in table.columns]
    labels = dict(HEADLINE)
    x = np.arange(len(table))
    width = 0.8 / max(len(fig_cols), 1)
    fig, ax = plt.subplots(figsize=(1.7 * len(table) + 3, 5))
    for k, col in enumerate(fig_cols):
        vals = table[col].values
        err = table[col + '__std'].values if col + '__std' in table else None
        bars = ax.bar(x + k * width, vals, width, label=labels.get(col, col),
                      yerr=err, capsize=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    ax.set_xticks(x + width * (len(fig_cols) - 1) / 2)
    ax.set_xticklabels(table.index, rotation=20, ha='right')
    ax.set_ylabel('Performance Score')
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'comparison_sota.png'), dpi=200)
    print(f'\nWrote {args.out_dir}/comparison.csv, comparison.tex, comparison_sota.png')
    print('Error bars are std across seeds — single-seed rows (e.g. EdgeBank) have none.')


if __name__ == '__main__':
    main()
