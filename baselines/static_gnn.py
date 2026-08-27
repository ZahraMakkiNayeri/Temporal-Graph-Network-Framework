"""
Static graph neural network baselines: GCN and GraphSAGE.

These are the two baselines in Experiment 5 that have no counterpart in the
temporal pipeline. They are deliberately static: a single graph is built from
the TRAINING WINDOW ONLY, node embeddings are computed once, and the same
embeddings are used to score test interactions. No interaction at or after the
prediction time contributes, so the causality constraint matches the temporal
models. What they cannot do -- and this is the point of the comparison -- is
update a node's representation as new interactions arrive.

Implemented directly in PyTorch (no torch-geometric dependency):
  GCN        symmetric normalised propagation, Z = sigma(D^-1/2 (A+I) D^-1/2 Z W)
  GraphSAGE  mean-aggregation, Z = sigma([Z ‖ mean_neighbours(Z)] W)

Protocol is identical to run_train.py: same chronological splits, same 1:1
destination-corrupted negatives with the same seeds, the same four
negative-sampling regimes at test time, threshold tuned on validation, and
multi-seed reporting.

Also writes per-day metrics, which is what Figure 12 ("metrics over time")
needs.

USAGE
    python -m baselines.static_gnn --processed-dir data/processed_cat \\
        --model gcn --n-runs 5 --out-dir ~/runs/gcn
    python -m baselines.static_gnn --processed-dir data/processed_cat \\
        --model graphsage --n-runs 5 --out-dir ~/runs/graphsage
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import get_data
from utils.utils import RandEdgeSampler
from utils.negative_samplers import (CollisionCheckedRandEdgeSampler,
                                     HistoricalEdgeSampler, InductiveEdgeSampler)


def build_adjacency(sources, destinations, n_nodes, device, normalise):
    """Sparse adjacency over the training window, symmetric, with self-loops."""
    s = np.concatenate([sources, destinations, np.arange(n_nodes)])
    d = np.concatenate([destinations, sources, np.arange(n_nodes)])
    idx = torch.tensor(np.stack([s, d]), dtype=torch.long, device=device)
    val = torch.ones(idx.shape[1], device=device)
    A = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    if normalise == 'gcn':                       # D^-1/2 A D^-1/2
        dinv = deg.pow(-0.5)
        v = A.values() * dinv[A.indices()[0]] * dinv[A.indices()[1]]
    else:                                        # row-mean, for GraphSAGE
        v = A.values() / deg[A.indices()[0]]
    return torch.sparse_coo_tensor(A.indices(), v, (n_nodes, n_nodes)).coalesce()


class StaticGNN(nn.Module):
    def __init__(self, in_dim, hid, n_cat, edge_dim, kind):
        super().__init__()
        self.kind = kind
        mult = 2 if kind == 'graphsage' else 1
        self.l1 = nn.Linear(in_dim * mult, hid)
        self.l2 = nn.Linear(hid * mult, hid)
        self.link = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.cat = nn.Sequential(nn.Linear(2 * hid + edge_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, n_cat))

    def propagate(self, A, X, layer):
        agg = torch.sparse.mm(A, X)
        h = torch.cat([X, agg], dim=1) if self.kind == 'graphsage' else agg
        return F.relu(layer(h))

    def embed(self, A, X):
        return self.propagate(A, self.propagate(A, X, self.l1), self.l2)

    def score_link(self, z, u, v):
        return self.link(torch.cat([z[u], z[v]], dim=1)).squeeze(-1)

    def score_cat(self, z, u, v, ef):
        return self.cat(torch.cat([z[u], z[v], ef], dim=1))


def sample_neg(sampler, data):
    if getattr(sampler, 'needs_batch_context', False):
        _, neg = sampler.sample(len(data.sources), sources=data.sources,
                                destinations=data.destinations)
    else:
        _, neg = sampler.sample(len(data.sources))
    return neg


def link_scores(model, z, data, neg, device):
    u = torch.tensor(data.sources, dtype=torch.long, device=device)
    v = torch.tensor(data.destinations, dtype=torch.long, device=device)
    n = torch.tensor(neg, dtype=torch.long, device=device)
    with torch.no_grad():
        pos = torch.sigmoid(model.score_link(z, u, v)).cpu().numpy()
        negs = torch.sigmoid(model.score_link(z, u, n)).cpu().numpy()
    return pos, negs


def metrics_from(pos, neg, thr):
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    p = (s >= thr).astype(int)
    return {'auc': roc_auc_score(y, s), 'ap': average_precision_score(y, s),
            'link_acc': accuracy_score(y, p),
            'link_precision': precision_score(y, p, zero_division=0),
            'link_recall': recall_score(y, p, zero_division=0),
            'link_f1': f1_score(y, p, zero_division=0)}


def ranking(model, z, data, pool, Q, device, rng):
    u = torch.tensor(data.sources, dtype=torch.long, device=device)
    v = torch.tensor(data.destinations, dtype=torch.long, device=device)
    with torch.no_grad():
        pos = model.score_link(z, u, v)
        cand = torch.tensor(rng.choice(pool, size=(len(data.sources), Q)),
                            dtype=torch.long, device=device)
        us = u.unsqueeze(1).expand(-1, Q).reshape(-1)
        cs = cand.reshape(-1)
        sc = model.score_link(z, us, cs).reshape(-1, Q)
        greater = (sc > pos.unsqueeze(1)).sum(1).float()
        ties = (sc == pos.unsqueeze(1)).sum(1).float()
        r = (1 + greater + 0.5 * ties).cpu().numpy()
    return {'link_mrr': float(np.mean(1 / r)), 'link_hits1': float(np.mean(r <= 1)),
            'link_hits3': float(np.mean(r <= 3)), 'link_hits10': float(np.mean(r <= 10))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--model', choices=['gcn', 'graphsage'], default='gcn')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-runs', type=int, default=5)
    p.add_argument('--link-ranking-candidates', type=int, default=100)
    p.add_argument('--day-seconds', type=float, default=86400.0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)

    df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    ef = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    nf = np.load(os.path.join(args.processed_dir, 'node_features.npy'))
    nfeat, efeat, full, train, val, test, _, nntest = get_data(
        df, ef, nf, different_new_nodes_between_val_and_test=True)
    n_nodes, n_cat = nfeat.shape[0], int(df.label.max()) + 1
    X = torch.tensor(nfeat, dtype=torch.float32, device=device)
    E = torch.tensor(efeat, dtype=torch.float32, device=device)
    A = build_adjacency(train.sources, train.destinations, n_nodes, device,
                        'gcn' if args.model == 'gcn' else 'sage')
    print(f'{args.model}: {n_nodes:,} nodes, adjacency from the training window only '
          f'({len(train.sources):,} interactions)')

    hist_s = np.concatenate([train.sources, val.sources])
    hist_d = np.concatenate([train.destinations, val.destinations])
    pool = np.unique(full.destinations)
    summaries = []

    for run in range(args.n_runs):
        torch.manual_seed(run); np.random.seed(run)
        model = StaticGNN(nfeat.shape[1], args.hidden, n_cat, efeat.shape[1],
                          args.model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        tr_sampler = RandEdgeSampler(train.sources, train.destinations, seed=run)
        va_sampler = RandEdgeSampler(full.sources, full.destinations, seed=1)
        best_ap, best_state, bad = -1, None, 0

        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            z = model.embed(A, X)
            neg = sample_neg(tr_sampler, train)
            u = torch.tensor(train.sources, dtype=torch.long, device=device)
            v = torch.tensor(train.destinations, dtype=torch.long, device=device)
            n = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bce(model.score_link(z, u, v), torch.ones(len(u), device=device)) + \
                   bce(model.score_link(z, u, n), torch.zeros(len(u), device=device))
            eidx = torch.tensor(train.edge_idxs, dtype=torch.long, device=device)
            loss = loss + ce(model.score_cat(z, u, v, E[eidx]),
                             torch.tensor(train.labels, dtype=torch.long, device=device))
            loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                zv = model.embed(A, X)
            pv, nv = link_scores(model, zv, val, sample_neg(va_sampler, val), device)
            ap = average_precision_score(np.r_[np.ones_like(pv), np.zeros_like(nv)],
                                         np.r_[pv, nv])
            if ap > best_ap:
                best_ap, best_state, bad = ap, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            z = model.embed(A, X)

        # threshold tuned on validation, applied unchanged to test
        pv, nv = link_scores(model, z, val, sample_neg(va_sampler, val), device)
        yv = np.r_[np.ones_like(pv), np.zeros_like(nv)]; sv = np.r_[pv, nv]
        thr = max(np.linspace(.01, .99, 99), key=lambda t: f1_score(yv, (sv >= t).astype(int), zero_division=0))

        rec = {'run': run, 'seed': run, 'tuned_link_threshold': thr, 'best_val_ap': best_ap}
        regimes = {
            '': RandEdgeSampler(full.sources, full.destinations, seed=2),
            '_random_cc': CollisionCheckedRandEdgeSampler(full.sources, full.destinations, seed=2),
            '_historical': HistoricalEdgeSampler(hist_s, hist_d, full.destinations, seed=2),
            '_inductive': InductiveEdgeSampler(hist_s, hist_d, test.sources,
                                               test.destinations, full.destinations, seed=2),
        }
        for suffix, sampler in regimes.items():
            pt, nt = link_scores(model, z, test, sample_neg(sampler, test), device)
            for k, v_ in metrics_from(pt, nt, thr).items():
                rec[f'test_{k}{suffix}'] = v_
        rng = np.random.RandomState(123)
        rec.update({f'test_{k}': v_ for k, v_ in
                    ranking(model, z, test, pool, args.link_ranking_candidates, device, rng).items()})
        pn, nn_ = link_scores(model, z, nntest, sample_neg(regimes[''], nntest), device)
        for k, v_ in metrics_from(pn, nn_, thr).items():
            rec[f'nn_test_{k}'] = v_

        # category prediction on ground-truth edges only
        with torch.no_grad():
            ut = torch.tensor(test.sources, dtype=torch.long, device=device)
            vt = torch.tensor(test.destinations, dtype=torch.long, device=device)
            et = torch.tensor(test.edge_idxs, dtype=torch.long, device=device)
            pred = model.score_cat(z, ut, vt, E[et]).argmax(1).cpu().numpy()
        yt = test.labels.astype(int)
        rec['test_category_accuracy'] = accuracy_score(yt, pred)
        rec['test_f1_macro'] = f1_score(yt, pred, average='macro', zero_division=0)

        # per-day metrics for Figure 12
        day = ((test.timestamps - full.timestamps.min()) // args.day_seconds).astype(int)
        per_day = []
        for d in np.unique(day):
            m = day == d
            if m.sum() < 50:
                continue
            sub = type(test)(test.sources[m], test.destinations[m], test.timestamps[m],
                             test.edge_idxs[m], test.labels[m])
            pt, nt = link_scores(model, z, sub, sample_neg(regimes[''], sub), device)
            row = {'run': run, 'day': int(d), 'n': int(m.sum())}
            row.update(metrics_from(pt, nt, thr))
            row.update(ranking(model, z, sub, pool, args.link_ranking_candidates,
                               device, np.random.RandomState(123)))
            row['category_accuracy'] = accuracy_score(yt[m], pred[m])
            per_day.append(row)
        pd.DataFrame(per_day).to_csv(
            os.path.join(os.path.expanduser(args.out_dir), f'per_day_run{run}.csv'),
            index=False) if per_day else None

        summaries.append(rec)
        print(f"  run {run}: AUC {rec['test_auc']:.4f}  AP {rec['test_ap']:.4f}  "
              f"MRR {rec['test_link_mrr']:.4f}  cat acc {rec['test_category_accuracy']:.4f}")
        pd.DataFrame(summaries).set_index('run').to_csv(
            os.path.join(os.path.expanduser(args.out_dir), 'test_metrics_multi_seed.csv'))

    s = pd.DataFrame(summaries)
    print(f'\n=== {args.model} over {len(s)} seeds ===')
    for c in ('test_auc', 'test_ap', 'test_link_mrr', 'test_category_accuracy', 'test_f1_macro'):
        print(f'  {c:26s} {s[c].mean():.4f} ± {s[c].std(ddof=1) if len(s) > 1 else 0:.4f}')
    print(f'wrote {args.out_dir}/test_metrics_multi_seed.csv and per_day_run*.csv')


if __name__ == '__main__':
    main()
