"""
Classical baselines under the identical protocol as the temporal models.

Addresses Reviewer 6 Comment 13 and Reviewer 1 Comment 8, which note that the
original SVM/RF baselines were reported with no description of feature
construction, class balancing or hyperparameter tuning, and with near-chance
inductive AUCs suggesting they were not competitively configured.

PROTOCOL (identical to run_train.py)
  - the same chronological 70/85-percentile splits from get_data
  - the same 1:1 destination-corrupted negative sampling, same seeds
  - the same three negative-sampling regimes at test time
  - hyperparameters selected on the VALIDATION split only, grid stated below
  - class imbalance handled by class_weight='balanced' (never by resampling)

FEATURES. Temporal models see per-node memory and sampled temporal neighbours;
a classical model cannot. To give these baselines a fair chance rather than a
strawman, each candidate pair (u, v, t) is represented by:
    node features of u and v
    |f(u) - f(v)| and f(u) * f(v)          (standard link-prediction pair encodings)
    degree of u and of v in the training window
    number of past u-v interactions, and time since the last one
    whether the pair was ever observed in the training window
All of these are computed from interactions strictly before t, so the baselines
respect the same causality constraint as BiTA.

Category prediction is evaluated on ground-truth edges only, as in the temporal
pipeline, using node features plus the edge features of that interaction.

USAGE
    python -m baselines.classical --processed-dir data/processed_cat \\
        --out-dir ~/runs/classical
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import get_data
from utils.utils import RandEdgeSampler
from utils.negative_samplers import (CollisionCheckedRandEdgeSampler,
                                     HistoricalEdgeSampler, InductiveEdgeSampler)


class PairFeaturiser:
    """Builds causal pair features from interactions strictly before each time."""

    def __init__(self, node_features):
        self.nf = node_features
        self.deg = defaultdict(int)
        self.pair_count = defaultdict(int)
        self.pair_last = {}

    def observe(self, src, dst, ts):
        for s, d, t in zip(src, dst, ts):
            self.deg[int(s)] += 1
            self.deg[int(d)] += 1
            self.pair_count[(int(s), int(d))] += 1
            self.pair_last[(int(s), int(d))] = float(t)

    def encode(self, src, dst, ts):
        fs = self.nf[np.asarray(src)]
        fd = self.nf[np.asarray(dst)]
        rows = []
        for k, (s, d, t) in enumerate(zip(src, dst, ts)):
            key = (int(s), int(d))
            cnt = self.pair_count.get(key, 0)
            last = self.pair_last.get(key)
            rows.append([self.deg.get(int(s), 0), self.deg.get(int(d), 0),
                         cnt, 1.0 if cnt > 0 else 0.0,
                         (float(t) - last) if last is not None else -1.0])
        extra = np.asarray(rows, dtype=np.float32)
        return np.concatenate([fs, fd, np.abs(fs - fd), fs * fd, extra], axis=1)


def link_dataset(feat, data, sampler, seed):
    """Positives plus one destination-corrupted negative each, as in training."""
    rng = np.random.RandomState(seed)
    if getattr(sampler, 'needs_batch_context', False):
        _, neg = sampler.sample(len(data.sources), sources=data.sources,
                                destinations=data.destinations)
    else:
        _, neg = sampler.sample(len(data.sources))
    X = np.vstack([feat.encode(data.sources, data.destinations, data.timestamps),
                   feat.encode(data.sources, neg, data.timestamps)])
    y = np.concatenate([np.ones(len(data.sources)), np.zeros(len(neg))])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def scores_of(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def link_metrics(y, s, thr):
    p = (s >= thr).astype(int)
    return {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s),
            'acc': accuracy_score(y, p), 'precision': precision_score(y, p, zero_division=0),
            'recall': recall_score(y, p, zero_division=0), 'f1': f1_score(y, p, zero_division=0)}


def tune_threshold(y, s):
    best, bf = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf:
            best, bf = t, f
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--out-dir', default='runs/classical')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-svm-train', type=int, default=40000,
                   help='LinearSVC is used; cap kept for runtime parity')
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    ef = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    nf = np.load(os.path.join(args.processed_dir, 'node_features.npy'))
    nfeat, efeat, full, train, val, test, _, nntest = get_data(
        df, ef, nf, different_new_nodes_between_val_and_test=True)

    # ---- causal featuriser: fit on train, then extended with val for testing ----
    feat = PairFeaturiser(nfeat)
    feat.observe(train.sources, train.destinations, train.timestamps)
    Xtr, ytr = link_dataset(feat, train, RandEdgeSampler(train.sources, train.destinations, seed=0), 0)
    Xva, yva = link_dataset(feat, val, RandEdgeSampler(full.sources, full.destinations, seed=1), 1)
    feat.observe(val.sources, val.destinations, val.timestamps)   # history grows with time
    print(f'link features: {Xtr.shape[1]} dims, train {Xtr.shape[0]:,}, val {Xva.shape[0]:,}')

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    # ---- hyperparameter selection on VALIDATION only ----
    grids = {
        'RandomForest': [RandomForestClassifier(n_estimators=n, max_depth=d, n_jobs=-1,
                                                class_weight='balanced', random_state=args.seed)
                         for n in (200, 400) for d in (None, 12)],
        'LinearSVM': [LinearSVC(C=c, class_weight='balanced', max_iter=5000,
                                random_state=args.seed) for c in (0.01, 0.1, 1.0)],
        'LogisticRegression': [LogisticRegression(C=c, class_weight='balanced', max_iter=2000)
                               for c in (0.01, 0.1, 1.0)],
    }

    chosen, rows = {}, []
    for name, cands in grids.items():
        best, best_ap, best_desc = None, -1, ''
        for m in cands:
            Xa, Xb = (Xtr, Xva) if name == 'RandomForest' else (Xtr_s, Xva_s)
            m.fit(Xa, ytr)
            ap = average_precision_score(yva, scores_of(m, Xb))
            desc = str(m).replace('\n', ' ')
            print(f'  {name:20s} val AP {ap:.4f}   {desc[:70]}')
            if ap > best_ap:
                best, best_ap, best_desc = m, ap, desc
        chosen[name] = best
        print(f'  -> selected {name}: {best_desc[:80]} (val AP {best_ap:.4f})\n')

    # ---- test under the three negative-sampling regimes ----
    hist_src = np.concatenate([train.sources, val.sources])
    hist_dst = np.concatenate([train.destinations, val.destinations])
    regimes = {
        'random': RandEdgeSampler(full.sources, full.destinations, seed=2),
        'random_cc': CollisionCheckedRandEdgeSampler(full.sources, full.destinations, seed=2),
        'historical': HistoricalEdgeSampler(hist_src, hist_dst, full.destinations, seed=2),
        'inductive': InductiveEdgeSampler(hist_src, hist_dst, test.sources,
                                          test.destinations, full.destinations, seed=2),
    }
    print('link prediction on the test split')
    for name, model in chosen.items():
        rec = {'model': name}
        Xv = Xva if name == 'RandomForest' else Xva_s
        thr = tune_threshold(yva, scores_of(model, Xv))
        rec['tuned_threshold'] = thr
        for rname, sampler in regimes.items():
            Xte, yte = link_dataset(feat, test, sampler, 2)
            Xte = Xte if name == 'RandomForest' else scaler.transform(Xte)
            m = link_metrics(yte, scores_of(model, Xte), thr)
            for k, v in m.items():
                rec[f'test_{k}_{rname}' if rname != 'random' else f'test_{k}'] = v
        Xnn, ynn = link_dataset(feat, nntest, regimes['random'], 3)
        Xnn = Xnn if name == 'RandomForest' else scaler.transform(Xnn)
        for k, v in link_metrics(ynn, scores_of(model, Xnn), thr).items():
            rec[f'nn_test_{k}'] = v
        # save ROC/PR curve data so Figures 7-8 can be regenerated
        from sklearn.metrics import precision_recall_curve, roc_curve
        Xte0, yte0 = link_dataset(feat, test, regimes['random'], 2)
        Xte0 = Xte0 if name == 'RandomForest' else scaler.transform(Xte0)
        s_te = scores_of(model, Xte0)
        s_nn = scores_of(model, Xnn)
        cdir = os.path.join(os.path.expanduser(args.out_dir), name.lower())
        os.makedirs(cdir, exist_ok=True)
        pr_t, pr_n = precision_recall_curve(yte0, s_te), precision_recall_curve(ynn, s_nn)
        ro_t, ro_n = roc_curve(yte0, s_te), roc_curve(ynn, s_nn)
        np.savez_compressed(os.path.join(cdir, 'curves.npz'),
                            pr_precision=pr_t[0], pr_recall=pr_t[1],
                            roc_fpr=ro_t[0], roc_tpr=ro_t[1],
                            nn_pr_precision=pr_n[0], nn_pr_recall=pr_n[1],
                            nn_roc_fpr=ro_n[0], nn_roc_tpr=ro_n[1])
        rows.append(rec)
        print(f"  {name:20s} AUC {rec['test_auc']:.4f}  AP {rec['test_ap']:.4f}  "
              f"F1 {rec['test_f1']:.4f}  | hist AUC {rec['test_auc_historical']:.4f}  "
              f"| new-node AUC {rec['nn_test_auc']:.4f}")

    # ---- category prediction on ground-truth edges only ----
    # The current edge's OWN category block must be excluded: it is a bijective
    # embedding of the label, so including it makes the task trivial (accuracy
    # 1.0) and measures leakage rather than baseline skill. The fair analogue of
    # what the temporal model receives is the distribution of categories
    # observed for the source node STRICTLY BEFORE t, which we add explicitly.
    cfg_path = os.path.join(args.processed_dir, 'preprocess_config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    edge_dim = efeat.shape[1]
    keep = edge_dim - 4 if cfg.get('include_category_features') else edge_dim
    if keep != edge_dim:
        print(f'\nexcluding the category block (edge feature columns {keep}:{edge_dim}) '
              f'from the category-prediction features')

    n_cat = int(df.label.max()) + 1

    def past_category_profile(d, history_src, history_lab):
        """Causal: category counts for each source node before its interaction."""
        counts = defaultdict(lambda: np.zeros(n_cat, dtype=np.float32))
        for s, l in zip(history_src, history_lab):
            counts[int(s)][int(l)] += 1
        prof = np.zeros((len(d.sources), n_cat), dtype=np.float32)
        for k, s in enumerate(d.sources):
            c = counts[int(s)]
            prof[k] = c / c.sum() if c.sum() > 0 else c
        return prof

    prof_tr = past_category_profile(train, train.sources, train.labels)
    hist_s = np.concatenate([train.sources, val.sources])
    hist_l = np.concatenate([train.labels, val.labels])
    prof_te = past_category_profile(test, hist_s, hist_l)

    print('category prediction (ground-truth edges only)')
    def cat_X(d, prof):
        return np.concatenate([nfeat[d.sources], nfeat[d.destinations],
                               efeat[d.edge_idxs][:, :keep], prof], axis=1)
    Ctr, Cte = cat_X(train, prof_tr), cat_X(test, prof_te)
    cscaler = StandardScaler().fit(Ctr)
    for name, mk in (('RandomForest', lambda: RandomForestClassifier(
                          n_estimators=400, class_weight='balanced', n_jobs=-1,
                          random_state=args.seed)),
                     ('LogisticRegression', lambda: LogisticRegression(
                          C=1.0, class_weight='balanced', max_iter=2000))):
        m = mk()
        A, B = (Ctr, Cte) if name == 'RandomForest' else (cscaler.transform(Ctr),
                                                          cscaler.transform(Cte))
        m.fit(A, train.labels.astype(int))
        pred = m.predict(B)
        acc = accuracy_score(test.labels.astype(int), pred)
        mf1 = f1_score(test.labels.astype(int), pred, average='macro', zero_division=0)
        for r in rows:
            if r['model'] == name:
                r['test_category_accuracy'] = acc
                r['test_f1_macro'] = mf1
        print(f'  {name:20s} accuracy {acc:.4f}  macro-F1 {mf1:.4f}')
    maj = np.bincount(test.labels.astype(int)).argmax()
    print(f"  majority-class baseline: {accuracy_score(test.labels.astype(int), np.full(len(test.labels), maj)):.4f}")

    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    out = pd.DataFrame(rows)
    out.insert(0, 'seed', args.seed)
    out.insert(0, 'run', range(len(out)))
    for _, r in out.iterrows():
        d = os.path.join(os.path.expanduser(args.out_dir), r['model'].lower())
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([r]).to_csv(os.path.join(d, 'test_metrics_multi_seed.csv'), index=False)
    out.to_csv(os.path.join(os.path.expanduser(args.out_dir), 'classical_all.csv'), index=False)
    print(f"\nWrote per-model CSVs under {args.out_dir} "
          f"(collect_results.py can read them directly)")


if __name__ == '__main__':
    main()
