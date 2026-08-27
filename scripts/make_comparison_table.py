"""
Generate the consolidated comparison table for the manuscript.

One table, methods as rows and metrics as columns, each cell mean +/- standard
deviation across seeds, with a marker where the proposed method is significantly
better or worse than that baseline after correction. A companion table gives the
paired difference, effect size and corrected p-value for each comparison.

Two choices matter and are made deliberately here.

  MULTIPLE-COMPARISON FAMILY. Correction is applied across the set of BASELINES
  for each metric, not across the metrics of one comparison. AUC, AP, link F1
  and link accuracy are correlated measurements of a single comparison rather
  than independent hypotheses, so correcting across them destroys power without
  controlling anything meaningful. Correcting across baselines is what the
  reviewers' comments describe.

  SEED COUNT. Every comparison uses only seeds present in BOTH runs, and the
  count is printed per row. Comparisons at different seed counts are not
  interchangeable: see the seed-sensitivity analysis. The script warns when
  runs disagree, and refuses to mark significance below --min-seeds.

USAGE
    python scripts/make_comparison_table.py \\
        --proposed "BiTA=~/bita-tgn-v7-nocat" \\
        --baseline "TGN-Last=~/runs/cat_tgn_last" \\
        --baseline "TGN-Mean=~/runs/cat_tgn_mean" \\
        --baseline "TGN-Attention=~/runs/agg_bitransformer" \\
        --baseline "DyRep=~/runs/cat_dyrep" \\
        --baseline "BiGRU=~/runs/agg_bigru" \\
        --baseline "EdgeBank=~/runs/edgebank" \\
        --out comparison/table_main
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = [
    ('test_auc', 'AUC'),
    ('test_ap', 'AP'),
    ('test_auc_historical', r'AUC$_{\mathrm{hist}}$'),
    ('test_link_mrr', 'MRR'),
    ('test_f1_macro', 'Macro-F1'),
]


def holm(p):
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m, run = len(idx), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load(spec):
    name, path = spec.split('=', 1)
    path = os.path.expanduser(path)
    csv = path if path.endswith('.csv') else os.path.join(path, 'test_metrics_multi_seed.csv')
    if not os.path.exists(csv):
        print(f'WARNING: {name} has no results at {csv} — skipped')
        return name, None
    d = pd.read_csv(csv)
    return name, d.set_index('seed' if 'seed' in d.columns else 'run').sort_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proposed', required=True, help='NAME=path')
    p.add_argument('--baseline', action='append', required=True, help='NAME=path')
    p.add_argument('--metrics', default=None,
                   help='comma-separated metric columns; default is the primary set')
    p.add_argument('--labels', default=None, help='comma-separated display labels')
    p.add_argument('--test', choices=['wilcoxon', 'ttest'], default='wilcoxon')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-seeds', type=int, default=10,
                   help='below this, differences are reported but not marked significant')
    p.add_argument('--out', default='comparison/table_main')
    args = p.parse_args()

    if args.metrics:
        cols = [m.strip() for m in args.metrics.split(',')]
        labs = ([l.strip() for l in args.labels.split(',')] if args.labels else cols)
        metrics = list(zip(cols, labs))
    else:
        metrics = DEFAULT_METRICS

    pname, P = load(args.proposed)
    if P is None:
        raise SystemExit('proposed method has no results')
    baselines = [load(s) for s in args.baseline]
    baselines = [(n, d) for n, d in baselines if d is not None]

    # ---- per-metric comparison against every baseline ---------------------
    cells, detail = {}, []
    for col, lab in metrics:
        raw_p, recs = [], []
        for name, B in baselines:
            if col not in P.columns or col not in B.columns:
                raw_p.append(np.nan)
                recs.append({'baseline': name, 'n': 0})
                continue
            seeds = sorted(set(P.index) & set(B.index))
            a = pd.to_numeric(P.loc[seeds, col], errors='coerce').values
            b = pd.to_numeric(B.loc[seeds, col], errors='coerce').values
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            n = len(a)
            rec = {'baseline': name, 'n': n,
                   'mean_baseline': b.mean() if n else np.nan,
                   'std_baseline': b.std(ddof=1) if n > 1 else 0.0,
                   'diff': (a - b).mean() if n else np.nan}
            if n >= 2 and (a - b).std(ddof=1) > 0:
                d = a - b
                rec['dz'] = d.mean() / d.std(ddof=1)
                half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
                rec['ci_lo'], rec['ci_hi'] = d.mean() - half, d.mean() + half
                try:
                    rec['p'] = (stats.wilcoxon(a, b)[1] if args.test == 'wilcoxon'
                                else stats.ttest_rel(a, b)[1])
                except ValueError:
                    rec['p'] = np.nan
            else:
                rec['dz'] = rec['p'] = rec['ci_lo'] = rec['ci_hi'] = np.nan
            raw_p.append(rec['p'])
            recs.append(rec)

        adj = holm(raw_p)
        for k, rec in enumerate(recs):
            rec['metric'], rec['label'] = col, lab
            rec['p_holm'] = adj[k]
            sig = (not np.isnan(adj[k]) and adj[k] < args.alpha
                   and rec['n'] >= args.min_seeds)
            rec['significant'] = sig
            cells[(rec['baseline'], col)] = rec
            detail.append(rec)

        if col in P.columns:
            v = pd.to_numeric(P[col], errors='coerce').dropna()
            cells[(pname, col)] = {'mean_baseline': v.mean(),
                                   'std_baseline': v.std(ddof=1) if len(v) > 1 else 0.0,
                                   'n': len(v), 'significant': False, 'diff': np.nan}

    # ---- console ----------------------------------------------------------
    seed_counts = {n: len(set(P.index) & set(B.index)) for n, B in baselines}
    if len(set(seed_counts.values())) > 1:
        print('WARNING: comparisons use different seed counts:',
              ', '.join(f'{k} {v}' for k, v in seed_counts.items()))
        print('         Extend the shorter runs before publishing; effect-size')
        print('         estimates are not comparable across seed counts.\n')

    rows = [pname] + [n for n, _ in baselines]
    disp = pd.DataFrame(index=rows, columns=['seeds'] + [l for _, l in metrics])
    for r in rows:
        any_n = [cells[(r, c)]['n'] for c, _ in metrics if (r, c) in cells]
        disp.loc[r, 'seeds'] = max(any_n) if any_n else 0
        for c, l in metrics:
            e = cells.get((r, c))
            if e is None or np.isnan(e.get('mean_baseline', np.nan)):
                disp.loc[r, l] = '--'
                continue
            txt = f"{e['mean_baseline']:.4f}"
            if e.get('n', 0) > 1:
                txt += f" ± {e['std_baseline']:.4f}"
            # console uses the unicode sign; the LaTeX writer swaps in \pm below
            if e.get('significant'):
                txt += '*'
            disp.loc[r, l] = txt
    print(disp.to_string())
    print(f"\n* proposed method differs significantly from this baseline "
          f"({args.test}, Holm-corrected across baselines, alpha={args.alpha})")

    det = pd.DataFrame([d for d in detail if d.get('n', 0) > 0])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    det.to_csv(args.out + '_detail.csv', index=False)
    disp.to_csv(args.out + '.csv')

    # ---- LaTeX: main table ------------------------------------------------
    with open(args.out + '.tex', 'w') as f:
        f.write('% Main comparison table. Cells are mean +/- std across seeds.\n')
        f.write(f'% * marks a baseline from which {pname} differs significantly\n')
        f.write(f'% ({args.test}, Holm-corrected across baselines per metric).\n')
        f.write('\\begin{tabular}{lr' + 'c' * len(metrics) + '}\n\\toprule\n')
        f.write('Method & Seeds & ' + ' & '.join(l for _, l in metrics) + ' \\\\\n\\midrule\n')
        for r in rows:
            cells_txt = ' & '.join(
                str(disp.loc[r, l]).replace('±', '$\\pm$').replace('*', '$^{*}$')
                for _, l in metrics)
            bold = r == pname
            nm = f'\\textbf{{{r}}}' if bold else r
            f.write(f"{nm} & {disp.loc[r, 'seeds']} & {cells_txt} \\\\\n")
            if bold:
                f.write('\\midrule\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    # ---- LaTeX: significance detail ---------------------------------------
    with open(args.out + '_detail.tex', 'w') as f:
        f.write('% Paired comparisons against the proposed method.\n')
        f.write('\\begin{tabular}{llrrrrr}\n\\toprule\n')
        f.write('Metric & Baseline & $n$ & $\\Delta$ & 95\\% CI & $d_z$ & '
                '$p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n')
        for c, l in metrics:
            for name, _ in baselines:
                e = cells.get((name, c))
                if e is None or not e.get('n'):
                    continue
                star = '$^{*}$' if e.get('significant') else ''
                ci = ('--' if np.isnan(e.get('ci_lo', np.nan))
                      else f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]")
                dz = '--' if np.isnan(e.get('dz', np.nan)) else f"{e['dz']:+.2f}"
                ph = '--' if np.isnan(e.get('p_holm', np.nan)) else f"{e['p_holm']:.4f}"
                f.write(f"{l} & {name} & {e['n']} & {e['diff']:+.4f} & {ci} & "
                        f"{dz} & {ph}{star} \\\\\n")
            f.write('\\midrule\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    n_sig = sum(1 for d in detail if d.get('significant'))
    print(f'\n{n_sig} of {len(detail)} comparisons significant after correction')
    print(f'wrote {args.out}.tex (main), {args.out}_detail.tex (statistics), '
          f'and the matching .csv files')
    if n_sig == 0:
        print('\nNo comparison survives correction. State that plainly rather than '
              'quoting uncorrected p-values.')


if __name__ == '__main__':
    main()
