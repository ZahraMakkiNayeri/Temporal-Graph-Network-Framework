"""
Difference-in-differences test: does the learnable aggregator exploit richer
message content better than heuristic aggregation does?

Motivation. On matched 9-dim features BiTA and TGN-Last are indistinguishable,
but with the 13-dim (historical-category) features BiTA gains far more than
TGN-Last. That is an INTERACTION between aggregator and message content, and it
is a stronger and more honest claim than a bare main effect: the aggregator is
not universally better, it converts additional message signal into performance
where a heuristic operator cannot.

For each seed i this computes
    delta_A(i) = A_rich(i) - A_plain(i)      (gain of aggregator A)
    delta_B(i) = B_rich(i) - B_plain(i)
and runs a paired test on delta_A - delta_B. Seeds are matched across all four
runs, so the test is paired throughout.

USAGE
    python scripts/interaction_test.py \
        --a-plain "BiTA-9dim=~/bita-tgn-v7" \
        --a-rich  "BiTA-13dim=~/bita-tgn-v7-nocat" \
        --b-plain "TGNLast-9dim=~/runs/tgn_last" \
        --b-rich  "TGNLast-13dim=~/runs/cat_tgn_last" \
        --metrics test_auc,test_ap,test_auc_historical
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

def load(spec):
    name, path = spec.split('=', 1)
    path = os.path.expanduser(path)
    csv = path if path.endswith('.csv') else os.path.join(path, 'test_metrics_multi_seed.csv')
    if not os.path.exists(csv):
        raise SystemExit(f'not found: {csv}')
    df = pd.read_csv(csv)
    key = 'seed' if 'seed' in df.columns else 'run'
    return name, df.set_index(key).sort_index()

p = argparse.ArgumentParser()
for f in ('a-plain', 'a-rich', 'b-plain', 'b-rich'):
    p.add_argument('--' + f, required=True, help='NAME=path')
p.add_argument('--metrics', default='test_auc,test_ap,test_auc_historical')
p.add_argument('--out', default=None)
args = p.parse_args()

(na, A0), (nb, A1) = load(args.a_plain), load(args.a_rich)
(nc, B0), (nd, B1) = load(args.b_plain), load(args.b_rich)
seeds = sorted(set(A0.index) & set(A1.index) & set(B0.index) & set(B1.index))
print(f"interaction test over {len(seeds)} matched seeds {seeds}")
print(f"  aggregator A: {na} -> {nb}")
print(f"  aggregator B: {nc} -> {nd}\n")

rows = []
for m in [x.strip() for x in args.metrics.split(',')]:
    if not all(m in d.columns for d in (A0, A1, B0, B1)):
        print(f"  {m}: missing in at least one run, skipped"); continue
    dA = A1.loc[seeds, m].values - A0.loc[seeds, m].values
    dB = B1.loc[seeds, m].values - B0.loc[seeds, m].values
    diff = dA - dB
    t, pt = stats.ttest_rel(dA, dB)
    try:
        w, pw = stats.wilcoxon(dA, dB)
    except ValueError:
        w, pw = np.nan, np.nan
    sd = diff.std(ddof=1)
    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
    rows.append({'metric': m, 'gain_A': dA.mean(), 'gain_B': dB.mean(),
                 'interaction': diff.mean(), 'ci95_lo': diff.mean() - half,
                 'ci95_hi': diff.mean() + half, 'dz': diff.mean() / sd if sd else np.inf,
                 'p_ttest': pt, 'p_wilcoxon': pw})
    print(f"  {m}")
    print(f"    gain of {na.split('-')[0]:>10s}: {dA.mean():+.4f}   "
          f"gain of {nc.split('-')[0]:>10s}: {dB.mean():+.4f}")
    print(f"    interaction {diff.mean():+.4f}  95% CI [{diff.mean()-half:+.4f}, "
          f"{diff.mean()+half:+.4f}]  dz {diff.mean()/sd if sd else float('inf'):.2f}"
          f"  t p={pt:.4f}  Wilcoxon p={pw:.4f}"
          f"{'  *' if pw < 0.05 else ''}\n")

if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('Wrote', args.out)
print("A positive interaction with a CI excluding zero means the learnable")
print("aggregator converts the extra message content into performance more")
print("effectively than the heuristic operator does.")
