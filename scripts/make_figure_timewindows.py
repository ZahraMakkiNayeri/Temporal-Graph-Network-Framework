"""
Figure 9: performance as a function of training-window length (Experiment 3).

Each point is a model trained on the first N days of the stream and tested on
everything after it, so the x-axis is training duration and the gap to the test
period grows as N shrinks. Values are means over seeds with standard-deviation
error bars, which the original figure lacked.

The manuscript concluded from this experiment that "the model maintains stable
performance across such temporal gaps, suggesting that frequent retraining is
unnecessary". Whether that survives is now a measurable question rather than an
assertion: with error bars, stability means the bars overlap.

USAGE
    python scripts/make_figure_timewindows.py --runs-root ~/runs --prefix tw \
        --out-dir comparison
"""
import argparse, glob, os, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--runs-root', default=os.path.expanduser('~/runs'))
p.add_argument('--prefix', default='tw', help='run directories named <prefix><days>')
p.add_argument('--metrics', default='test_ap,test_auc,test_category_accuracy,test_f1_macro')
p.add_argument('--labels', default='AP,AUC,Category accuracy,Category macro-F1')
p.add_argument('--out-dir', default='comparison')
a = p.parse_args()

metrics = [m.strip() for m in a.metrics.split(',')]
labels = [l.strip() for l in a.labels.split(',')]
rows = []
for d in sorted(glob.glob(os.path.join(os.path.expanduser(a.runs_root), a.prefix + '*'))):
    m = re.search(rf'{re.escape(a.prefix)}(\d+(?:\.\d+)?)$', os.path.basename(d))
    csv = os.path.join(d, 'test_metrics_multi_seed.csv')
    if not m or not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    rec = {'days': float(m.group(1)), 'seeds': len(df)}
    for k in metrics:
        if k in df.columns:
            v = pd.to_numeric(df[k], errors='coerce').dropna()
            rec[k] = v.mean()
            rec[k + '_std'] = v.std(ddof=1) if len(v) > 1 else 0.0
    rows.append(rec)

if not rows:
    raise SystemExit(f'no runs matching {a.prefix}<days> under {a.runs_root}')
t = pd.DataFrame(rows).sort_values('days').reset_index(drop=True)
print(t.round(4).to_string(index=False))

os.makedirs(a.out_dir, exist_ok=True)
t.to_csv(os.path.join(a.out_dir, 'time_windows.csv'), index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for k, lbl in zip(metrics, labels):
    if k not in t.columns:
        continue
    ax.errorbar(t['days'], t[k], yerr=t.get(k + '_std'), marker='o', capsize=3, label=lbl)
ax.set_xlabel('training window (days); tested on all later days')
ax.set_ylabel('metric value')
ax.set_xticks(t['days'])
ax.grid(alpha=.3); ax.set_axisbelow(True); ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(a.out_dir, 'figure9_time_windows.png')
fig.savefig(out, dpi=300); fig.savefig(out.replace('.png', '.pdf'))
print('wrote', out)

for k, lbl in zip(metrics, labels):
    if k not in t.columns or len(t) < 2:
        continue
    spread = t[k].max() - t[k].min()
    typical = t[k + '_std'].mean()
    verdict = 'within seed noise' if spread <= 2 * typical else 'exceeds seed noise'
    print(f'  {lbl:22s} range across windows {spread:.4f}, mean std {typical:.4f} -> {verdict}')
print('\nTest sets differ between points (each is "everything after the window"), so')
print('these are not paired comparisons -- describe the trend, do not test it.')
