"""
Seed-sensitivity analysis: how paired effect-size estimates and p-values depend
on the number of seeds.

This is a preliminary experiment for the manuscript, run entirely on results
that already exist -- no additional training. It establishes the seed count
required for the comparisons reported later, and it quantifies how misleading a
small-sample estimate can be in this setting.

Two curves are produced per comparison:

  SEQUENTIAL   the estimate you would have obtained had you stopped after the
               first n seeds. This is the trajectory an experimenter actually
               observes as runs accumulate.

  SUBSAMPLED   the mean and interquartile range of the estimate over many
               random subsets of size n drawn from the full set of seeds. This
               removes the arbitrariness of seed ordering and shows the spread
               of estimates a single n-seed study could have produced.

Also reported: the number of seeds required for 80% power at the effect size
observed with the full set, and the smallest attainable Wilcoxon p-value at each
n (the exact two-sided test cannot go below 2/2^n, so small samples cannot reach
significance regardless of effect size).

USAGE
    python scripts/seed_sensitivity.py \\
        --pair "BiTA vs BiGRU=~/bita-tgn-v7-nocat:~/runs/agg_bigru" \\
        --pair "BiTA vs Transformer=~/bita-tgn-v7-nocat:~/runs/agg_bitransformer" \\
        --metric test_auc_historical --out-dir comparison
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_pair(spec):
    name, paths = spec.split('=', 1)
    pa, pb = paths.split(':')
    out = []
    for p in (pa, pb):
        p = os.path.expanduser(p)
        csv = p if p.endswith('.csv') else os.path.join(p, 'test_metrics_multi_seed.csv')
        if not os.path.exists(csv):
            raise SystemExit(f'not found: {csv}')
        d = pd.read_csv(csv)
        out.append(d.set_index('seed' if 'seed' in d.columns else 'run').sort_index())
    return name, out[0], out[1]


def power_for(dz, n):
    crit = stats.t.ppf(0.975, n - 1)
    ncp = dz * np.sqrt(n)
    return 1 - stats.nct.cdf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)


def seeds_for_power(dz, target=0.80, cap=500):
    if dz == 0:
        return cap
    n = 4
    while n < cap and power_for(abs(dz), n) < target:
        n += 1
    return n


def stats_of(d):
    """dz and paired-t p for a vector of differences."""
    if len(d) < 2:
        return np.nan, np.nan
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    _, p = stats.ttest_rel(d, np.zeros_like(d)) if sd > 0 else (np.nan, np.nan)
    return dz, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', action='append', required=True,
                   help='NAME=pathA:pathB  (the comparison is A minus B)')
    p.add_argument('--metric', default='test_auc_historical')
    p.add_argument('--min-n', type=int, default=3)
    p.add_argument('--n-subsamples', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--panel-b-max-n', type=int, default=None,
                   help='truncate panel (b) at this n. Near n=N the number of '
                        'distinct subsets collapses (25 at n=24, 1 at n=25), so the '
                        'estimate degenerates. Defaults to N-5.')
    p.add_argument('--out-dir', default='comparison')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = []

    for spec in args.pair:
        name, A, B = load_pair(spec)
        seeds = sorted(set(A.index) & set(B.index))
        if args.metric not in A.columns or args.metric not in B.columns:
            print(f'{name}: metric {args.metric} missing — skipped')
            continue
        a = pd.to_numeric(A.loc[seeds, args.metric], errors='coerce').values
        b = pd.to_numeric(B.loc[seeds, args.metric], errors='coerce').values
        diff = a - b
        N = len(diff)
        dz_full, p_full = stats_of(diff)
        need = seeds_for_power(dz_full)
        print(f'\n{name}  ({args.metric}, {N} matched seeds)')
        print(f'  full sample: mean diff {diff.mean():+.4f}, dz {dz_full:+.3f}, '
              f'paired-t p {p_full:.4f}')
        print(f'  seeds for 80% power at dz {abs(dz_full):.2f}: {need}')

        ns = np.arange(args.min_n, N + 1)
        seq = [stats_of(diff[:n]) for n in ns]
        seq_dz = np.array([s[0] for s in seq])
        seq_p = np.array([s[1] for s in seq])

        sub_dz, sub_lo, sub_hi, sub_sig = [], [], [], []
        for n in ns:
            vals, sig = [], 0
            for _ in range(args.n_subsamples):
                idx = rng.choice(N, n, replace=False)
                dzi, pi = stats_of(diff[idx])
                if not np.isnan(dzi):
                    vals.append(dzi)
                if not np.isnan(pi) and pi < 0.05:
                    sig += 1
            vals = np.array(vals)
            sub_dz.append(vals.mean())
            sub_lo.append(np.percentile(vals, 25))
            sub_hi.append(np.percentile(vals, 75))
            sub_sig.append(sig / args.n_subsamples)

        line, = ax1.plot(ns, seq_dz, marker='o', ms=3, label=f'{name} (sequential)')
        ax1.plot(ns, sub_dz, '--', color=line.get_color(), alpha=.8,
                 label=f'{name} (subsampled mean)')
        ax1.fill_between(ns, sub_lo, sub_hi, color=line.get_color(), alpha=.13)
        cap = args.panel_b_max_n or (N - 5)
        keep = ns <= cap
        ax2.plot(ns[keep], np.array(sub_sig)[keep], marker='s', ms=3,
                 color=line.get_color(), label=name)

        for k, n in enumerate(ns):
            rows.append({'comparison': name, 'metric': args.metric, 'n': int(n),
                         'dz_sequential': seq_dz[k], 'p_sequential': seq_p[k],
                         'dz_subsampled_mean': sub_dz[k],
                         'dz_subsampled_q25': sub_lo[k], 'dz_subsampled_q75': sub_hi[k],
                         'prob_significant': sub_sig[k],
                         'wilcoxon_floor': 2.0 / (2 ** n)})
        # Report the subsampled mean at n=10 against the full sample. The
        # sequential value at very small n is noisy in both directions and is
        # not a sound basis for a claim about small-sample bias.
        for probe in (5, 10):
            if probe in list(ns):
                k = list(ns).index(probe)
                print(f'  n={probe:2d}: subsampled mean dz {sub_dz[k]:+.3f} '
                      f'(IQR {sub_lo[k]:+.3f} to {sub_hi[k]:+.3f}), '
                      f'P(p<0.05) = {sub_sig[k]:.0%}')
        print(f'  n={N}: dz {dz_full:+.3f}, p {p_full:.4f}')
        if 10 in list(ns) and dz_full:
            k = list(ns).index(10)
            print(f'  ratio of |dz| at n=10 to n={N}: '
                  f'{abs(sub_dz[k] / dz_full):.1f}x (subsampled mean)')

    ax1.axhline(0, color='k', lw=.8)
    ax1.axhline(0.8, color='grey', ls=':', lw=1)
    ax1.text(ax1.get_xlim()[1], 0.81, 'large effect', ha='right', fontsize=8, color='grey')
    ax1.set_xlabel('number of seeds'); ax1.set_ylabel("Cohen's $d_z$")
    #ax1.set_title('(a) effect-size estimate vs seed count')
    ax1.legend(fontsize=7); ax1.grid(alpha=.3); ax1.set_axisbelow(True)

    ax2.axhline(0.80, color='grey', ls=':', lw=1)
    ax2.text(ax2.get_xlim()[1], 0.81, '80% power', ha='right', fontsize=8, color='grey')
    ax2.set_xlabel('number of seeds')
    ax2.set_ylabel(r'P(p < 0.05) over random subsets')
    #ax2.set_title('(b) probability of declaring significance')
    ax2.set_ylim(0, 1); ax2.legend(fontsize=7); ax2.grid(alpha=.3); ax2.set_axisbelow(True)

    fig.tight_layout()
    base = os.path.join(args.out_dir, 'seed_sensitivity')
    fig.savefig(base + '.png', dpi=300); fig.savefig(base + '.pdf')
    pd.DataFrame(rows).to_csv(base + '.csv', index=False)

    print(f'\nwrote {base}.png, .pdf and .csv')
    print('\nWilcoxon exact two-sided floor (2/2^n): '
          + ', '.join(f'n={n}: {2/2**n:.4f}' for n in (5, 6, 8, 10, 15)))
    print('A small-sample study cannot reach alpha=0.05 with the rank test below n=6,')
    print('and panel (b) shows how often it would declare significance if it could.')


if __name__ == '__main__':
    main()
