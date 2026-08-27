"""
Combine several paired_tests.py outputs and apply multiple-comparison correction
across the family the reviewers actually mean: the set of BASELINES compared
against, for each metric.

Why this matters. paired_tests.py corrects across the metrics it tested, which
is too conservative: AUC, AP, link F1 and link accuracy are highly correlated
measurements of one comparison, not independent hypotheses, so Holm over 17 of
them destroys statistical power. Reviewer 1 #16, Reviewer 6 #11 and Reviewer 8
#10 ask for correction over the family of COMPARISONS. This script does that,
and additionally supports declaring a small set of primary metrics.

USAGE
    python scripts/significance_table.py \
        --in "TGN-Last=comparison/vs_tgn_last.csv" \
        --in "TGN-Mean=comparison/vs_tgn_mean.csv" \
        --in "TGN-Attention=comparison/vs_tgn_attn.csv" \
        --in "DyRep=comparison/vs_dyrep.csv" \
        --primary test_auc,test_ap,test_category_accuracy \
        --out comparison/significance.csv
"""
import argparse, os
import numpy as np, pandas as pd

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inputs', action='append', required=True,
                help='NAME=path/to/paired_tests_output.csv')
ap.add_argument('--test', choices=['p_ttest', 'p_wilcoxon'], default='p_wilcoxon',
                help='which test to correct (Wilcoxon is the safer default at n=10)')
ap.add_argument('--primary', default=None,
                help='comma-separated primary metrics; others reported uncorrected')
ap.add_argument('--alpha', type=float, default=0.05)
ap.add_argument('--out', default='comparison/significance.csv')
a = ap.parse_args()

frames = {}
for spec in a.inputs:
    name, path = spec.split('=', 1)
    frames[name] = pd.read_csv(os.path.expanduser(path)).set_index('metric')

metrics = sorted(set.intersection(*(set(f.index) for f in frames.values())))
if a.primary:
    primary = [m.strip() for m in a.primary.split(',')]
    metrics = [m for m in primary if m in metrics] + [m for m in metrics if m not in primary]
else:
    primary = metrics

rows = []
for m in metrics:
    raw = [frames[n].loc[m, a.test] for n in frames]
    adj = holm(raw) if m in primary else [np.nan] * len(raw)
    for k, n in enumerate(frames):
        f = frames[n].loc[m]
        rows.append({'metric': m, 'baseline': n, 'primary': m in primary,
                     'mean_diff': f['mean_diff'], 'ci95_lo': f['ci95_lo'],
                     'ci95_hi': f['ci95_hi'], 'dz': f['cohens_dz'],
                     'p_raw': raw[k], 'p_holm': adj[k],
                     'sig': '*' if (m in primary and adj[k] < a.alpha) else ''})

res = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
res.to_csv(a.out, index=False)

for m in metrics:
    sub = res[res.metric == m]
    tag = '' if sub.primary.iloc[0] else '   (secondary, uncorrected)'
    print(f"\n{m}{tag}")
    for _, r in sub.iterrows():
        pa = 'n/a' if np.isnan(r.p_holm) else f"{r.p_holm:.4f}"
        print(f"   vs {r.baseline:16s} {r.mean_diff:+.4f} "
              f"[{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]  dz {r.dz:5.2f}  "
              f"p_raw {r.p_raw:.4f}  p_holm {pa} {r.sig}")

n_sig = int((res.sig == '*').sum())
print(f"\n{n_sig} significant comparisons at alpha={a.alpha} after Holm correction "
      f"across {len(frames)} baselines, over {len(primary)} primary metric(s).")
print(f"Wrote {a.out}")
