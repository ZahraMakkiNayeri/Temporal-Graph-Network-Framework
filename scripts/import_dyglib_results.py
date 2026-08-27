"""
Collect DyGLib result files into the same CSV shape that collect_results.py and
paired_tests.py consume, so DyGLib baselines sit in one table with BiTA.

DyGLib writes JSON under saved_results/<model>/<dataset>/... . This script globs
them, pulls out AP / AUC per run, and writes one test_metrics_multi_seed.csv per
model. It is deliberately tolerant about key names because they vary a little by
DyGLib version — run it, check the printed key mapping, and adjust --ap-key /
--auc-key if your version differs.

USAGE
    python scripts/import_dyglib_results.py --dyglib-dir ~/DyGLib --name warden \
        --out-root ~/runs
"""
import argparse, glob, json, os, re
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--dyglib-dir', required=True)
p.add_argument('--name', default='warden')
p.add_argument('--out-root', default=os.path.expanduser('~/runs'))
p.add_argument('--ap-key', default='average_precision')
p.add_argument('--auc-key', default='roc_auc')
a = p.parse_args()

root = os.path.expanduser(a.dyglib_dir)
pattern = os.path.join(root, 'saved_results', '*', a.name, '**', '*.json')
files = glob.glob(pattern, recursive=True)
if not files:
    raise SystemExit(f'no result files under {pattern}\n'
                     'check that DyGLib finished and that --name matches')

def dig(obj, needle):
    """Find the first numeric value whose key contains `needle`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in k.lower() and isinstance(v, (int, float)):
                return float(v)
            got = dig(v, needle)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = dig(v, needle)
            if got is not None:
                return got
    return None

by_model = {}
for f in files:
    model = f.split(os.sep + 'saved_results' + os.sep)[1].split(os.sep)[0]
    blob = json.load(open(f))
    seed = 0
    m = re.search(r'seed[_-]?(\d+)|run[_-]?(\d+)', os.path.basename(f), re.I)
    if m:
        seed = int(m.group(1) or m.group(2))
    row = {'run': seed, 'seed': seed,
           'test_ap': dig(blob, a.ap_key) or dig(blob, 'average precision'),
           'test_auc': dig(blob, a.auc_key) or dig(blob, 'auc')}
    if row['test_auc'] is None and row['test_ap'] is None:
        continue
    by_model.setdefault(model, []).append(row)

for model, rows in sorted(by_model.items()):
    d = os.path.join(os.path.expanduser(a.out_root), f'dyglib_{model.lower()}')
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates('run').sort_values('run').set_index('run')
    df.to_csv(os.path.join(d, 'test_metrics_multi_seed.csv'))
    print(f'{model:14s} {len(df)} run(s)  AUC {df["test_auc"].mean():.4f}  '
          f'AP {df["test_ap"].mean():.4f}  -> {d}')

print('\nAdd these to collect_results.py with --run "JODIE=~/runs/dyglib_jodie" etc.')
print('NOTE: DyGLib reports link prediction only — category metrics stay blank')
print('for these rows, which is correct and should be stated in the table caption.')
