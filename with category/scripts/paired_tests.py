"""
Paired significance tests between two runs, across matched seeds.

Both runs use torch.manual_seed(i)/np.random.seed(i) for run i, so seed i of run
A and seed i of run B share an initialization regime. Differences are therefore
PAIRED, and paired tests are both correct and more powerful than the unpaired
alternative.

Reports, per metric:
  n              number of matched seeds
  mean A, mean B, mean difference and its 95% CI
  dz             Cohen's d for paired samples (mean diff / std of diffs)
  t, p_t         paired t-test (assumes roughly normal differences)
  W, p_w         Wilcoxon signed-rank (distribution-free; the safer choice at
                 small n, but see the warning below)
  p_holm         Holm-Bonferroni adjusted p (family = all metrics tested)

IMPORTANT — Wilcoxon needs at least 6 seeds. The exact two-sided p-value can
never fall below 2/2^n: with n=5 the floor is 0.0625, so the test CANNOT reach
significance at alpha=0.05 no matter how large the effect. n=6 gives a floor of
0.03125, n=10 gives 0.00195. If you intend to report Wilcoxon results, run at
least 6 seeds and preferably 10. The script prints this warning when it applies.

USAGE
    python scripts/paired_tests.py \
        --a "with-category=~/bita-tgn-v3" \
        --b "no-category=~/bita-tgn-v3-nocat" \
        --out comparison/paired_tests.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = [
    'test_auc', 'test_ap', 'test_link_f1', 'test_link_acc',
    'test_link_mrr', 'test_link_hits1', 'test_link_hits3', 'test_link_hits10',
    'test_category_accuracy', 'test_f1_macro',
    'test_auc_historical', 'test_ap_historical',
    'test_auc_inductive', 'test_ap_inductive',
    'nn_test_auc', 'nn_test_ap', 'nn_test_link_mrr',
]


def load(spec):
    name, path = spec.split('=', 1)
    path = os.path.expanduser(path)
    csv = path if path.endswith('.csv') else os.path.join(path, 'test_metrics_multi_seed.csv')
    if not os.path.exists(csv):
        raise SystemExit(f'not found: {csv}')
    df = pd.read_csv(csv)
    key = 'seed' if 'seed' in df.columns else 'run'
    return name, df.set_index(key).sort_index()


def holm(pvals):
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='NAME=PATH (baseline / reference run)')
    ap.add_argument('--b', required=True, help='NAME=PATH (comparison run)')
    ap.add_argument('--metrics', default=None,
                    help='comma-separated metric names (default: a standard set)')
    ap.add_argument('--all-metrics', action='store_true',
                    help='test every numeric column the two runs share')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    name_a, A = load(args.a)
    name_b, B = load(args.b)
    seeds = sorted(set(A.index) & set(B.index))
    n = len(seeds)
    if n < 2:
        raise SystemExit(f'need at least 2 matched seeds, found {n}')
    A, B = A.loc[seeds], B.loc[seeds]

    if args.all_metrics:
        metrics = [c for c in A.columns if c in B.columns
                   and pd.api.types.is_numeric_dtype(A[c]) and c not in ('run', 'seed')]
    elif args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
    else:
        metrics = [m for m in DEFAULT_METRICS if m in A.columns and m in B.columns]

    wilcoxon_floor = 2.0 / (2 ** n)
    print(f'{name_b} vs {name_a}: {n} matched seeds ({seeds})\n')
    if wilcoxon_floor > args.alpha:
        print(f'!! WARNING: with n={n}, the smallest possible two-sided Wilcoxon p is '
              f'{wilcoxon_floor:.4f} > alpha={args.alpha}.')
        print(f'   Wilcoxon CANNOT reach significance here regardless of effect size.')
        print(f'   Use at least 6 seeds (floor 0.0313), preferably 10 (floor 0.0020).\n')

    rows = []
    for m in metrics:
        a = pd.to_numeric(A[m], errors='coerce').values
        b = pd.to_numeric(B[m], errors='coerce').values
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        if len(a) < 2:
            continue
        d = b - a
        sd = d.std(ddof=1)

        if np.allclose(d, 0):
            t_stat, p_t, w_stat, p_w, dz = 0.0, 1.0, 0.0, 1.0, 0.0
            lo = hi = 0.0
        else:
            t_stat, p_t = stats.ttest_rel(b, a)
            try:
                w_stat, p_w = stats.wilcoxon(b, a)
            except ValueError:
                w_stat, p_w = np.nan, np.nan
            dz = d.mean() / sd if sd > 0 else np.inf
            half = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
            lo, hi = d.mean() - half, d.mean() + half

        rows.append({'metric': m, 'n': len(a),
                     f'mean_{name_a}': a.mean(), f'mean_{name_b}': b.mean(),
                     'mean_diff': d.mean(), 'ci95_lo': lo, 'ci95_hi': hi,
                     'cohens_dz': dz, 't': t_stat, 'p_ttest': p_t,
                     'W': w_stat, 'p_wilcoxon': p_w})

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit('no shared metrics found')
    res['p_ttest_holm'] = holm(res['p_ttest'].values)
    res['sig'] = np.where(res['p_ttest_holm'] < args.alpha, '*', '')

    show = res[['metric', 'n', f'mean_{name_a}', f'mean_{name_b}', 'mean_diff',
                'ci95_lo', 'ci95_hi', 'cohens_dz', 'p_ttest', 'p_wilcoxon',
                'p_ttest_holm', 'sig']]
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(show.round(4).to_string(index=False))

    n_sig = int((res['p_ttest_holm'] < args.alpha).sum())
    print(f'\n{n_sig} of {len(res)} metrics significant at alpha={args.alpha} '
          f'after Holm correction across {len(res)} tests.')
    print('CI excluding 0 and |dz| > 0.8 indicate a large, reliable effect; a wide CI '
          'straddling 0 means the data cannot distinguish the two runs.')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        res.to_csv(args.out, index=False)
        print('Wrote', args.out)


if __name__ == '__main__':
    main()
