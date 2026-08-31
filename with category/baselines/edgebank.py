"""
EdgeBank baseline (Poursafaei et al., NeurIPS 2022 — reference [15]).

A pure memorization heuristic: an edge (u, v) is predicted positive iff the pair
was observed before. No learning, no parameters. Poursafaei showed it is a
deceptively strong baseline under RANDOM negatives and collapses under
HISTORICAL negatives — which is exactly the point Reviewer 6's Comment 2 makes,
so this baseline carries a lot of argumentative weight in the revision.

It is evaluated here through the SAME protocol as BiTA: identical temporal
splits from get_data, identical negative samplers, identical threshold-based
link metrics, and the identical link-ranking procedure (each positive ranked
against Q candidate destinations, ties counted 0.5). Only that way is the
comparison in the SOTA figure meaningful.

Two memory modes (both from the paper):
  --mode unlimited   remember every pair ever seen
  --mode window      remember pairs seen in the last --window-frac of the
                     observed time span (default 0.15, i.e. recent history)

Output: test_metrics_multi_seed.csv with the same column names the training
runs produce, so scripts/collect_results.py merges it with everything else.
EdgeBank does not predict categories, so category columns are absent (NaN).

USAGE
    python -m baselines.edgebank --processed-dir data/processed_cat \
        --link-ranking-candidates 100 --out-dir results_edgebank
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import get_data
from utils.utils import RandEdgeSampler
from utils.negative_samplers import (CollisionCheckedRandEdgeSampler,
                                     HistoricalEdgeSampler, InductiveEdgeSampler)


class EdgeBank:
    def __init__(self, mode='unlimited', window_frac=0.15):
        self.mode = mode
        self.window_frac = window_frac
        self.seen = {}           # (u, v) -> most recent timestamp
        self.window = None

    def fit(self, sources, destinations, timestamps):
        span = float(timestamps.max() - timestamps.min())
        self.window = span * self.window_frac
        for s, d, t in zip(sources, destinations, timestamps):
            self.seen[(int(s), int(d))] = float(t)

    def score(self, sources, destinations, now):
        """1.0 if the pair is in memory (and, in window mode, recent)."""
        out = np.zeros(len(sources), dtype=np.float32)
        for k, (s, d) in enumerate(zip(sources, destinations)):
            last = self.seen.get((int(s), int(d)))
            if last is None:
                continue
            if self.mode == 'unlimited' or (now - last) <= self.window:
                out[k] = 1.0
        return out

    def observe(self, sources, destinations, timestamps):
        for s, d, t in zip(sources, destinations, timestamps):
            self.seen[(int(s), int(d))] = float(t)


def evaluate(model, data, sampler, batch_size=200, ranking_candidates=0,
             ranking_seed=123, candidate_pool=None, stream_update=True):
    """Mirror of utils.evaluation for a scoring function with no parameters."""
    n = len(data.sources)
    n_batches = (n + batch_size - 1) // batch_size
    pos_scores, neg_scores, ranks = [], [], []
    rank_rng = np.random.RandomState(ranking_seed)
    if ranking_candidates and candidate_pool is None:
        candidate_pool = np.unique(data.destinations)

    for b in range(n_batches):
        lo, hi = b * batch_size, min(n, (b + 1) * batch_size)
        src, dst = data.sources[lo:hi], data.destinations[lo:hi]
        ts = data.timestamps[lo:hi]
        size = len(src)
        now = float(ts[-1])

        if getattr(sampler, 'needs_batch_context', False):
            _, neg = sampler.sample(size, sources=src, destinations=dst)
        else:
            _, neg = sampler.sample(size)

        p = model.score(src, dst, now)
        q = model.score(src, neg, now)
        pos_scores.append(p)
        neg_scores.append(q)

        if ranking_candidates:
            Q = ranking_candidates
            cand = rank_rng.choice(candidate_pool, size=(size, Q))
            clash = cand == np.asarray(dst)[:, None]
            if clash.any():
                cand[clash] = rank_rng.choice(candidate_pool, size=int(clash.sum()))
            cs = np.stack([model.score(np.repeat(src[i], Q), cand[i], now)
                           for i in range(size)])
            greater = (cs > p[:, None]).sum(axis=1)
            ties = (cs == p[:, None]).sum(axis=1)
            ranks.append(1.0 + greater + 0.5 * ties)

        # EdgeBank streams: the batch is memorized only AFTER being scored.
        if stream_update:
            model.observe(src, dst, ts)

    pos = np.concatenate(pos_scores)
    neg = np.concatenate(neg_scores)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    pred = (scores >= 0.5).astype(int)

    out = {
        'auc': roc_auc_score(labels, scores),
        'ap': average_precision_score(labels, scores),
        'link_acc': accuracy_score(labels, pred),
        'link_precision': precision_score(labels, pred, zero_division=0),
        'link_recall': recall_score(labels, pred, zero_division=0),
        'link_f1': f1_score(labels, pred, zero_division=0),
    }
    if ranks:
        r = np.concatenate(ranks)
        out.update({'link_mrr': float(np.mean(1.0 / r)),
                    'link_hits1': float(np.mean(r <= 1)),
                    'link_hits3': float(np.mean(r <= 3)),
                    'link_hits10': float(np.mean(r <= 10))})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--out-dir', default='.')
    p.add_argument('--mode', choices=['unlimited', 'window'], default='unlimited')
    p.add_argument('--window-frac', type=float, default=0.15)
    p.add_argument('--link-ranking-candidates', type=int, default=100)
    p.add_argument('--prefix', default='edgebank')
    args = p.parse_args()

    new_df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    edge_features = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    node_features = np.load(os.path.join(args.processed_dir, 'node_features.npy'))

    _, _, full_data, train_data, val_data, test_data, _, nn_test_data = get_data(
        new_df, edge_features, node_features)

    hist_src = np.concatenate([train_data.sources, val_data.sources])
    hist_dst = np.concatenate([train_data.destinations, val_data.destinations])
    hist_ts = np.concatenate([train_data.timestamps, val_data.timestamps])

    def fresh():
        m = EdgeBank(args.mode, args.window_frac)
        m.fit(hist_src, hist_dst, hist_ts)
        return m

    Q = args.link_ranking_candidates
    row = {'run': 0, 'seed': 0, 'model': f'EdgeBank-{args.mode}'}

    r = evaluate(fresh(), test_data, RandEdgeSampler(full_data.sources, full_data.destinations),
                 ranking_candidates=Q)
    row.update({f'test_{k}': v for k, v in r.items()})

    r = evaluate(fresh(), nn_test_data, RandEdgeSampler(full_data.sources, full_data.destinations),
                 ranking_candidates=Q)
    row.update({f'nn_test_{k}': v for k, v in r.items()})

    strategies = {
        'random_cc': CollisionCheckedRandEdgeSampler(full_data.sources, full_data.destinations, seed=2),
        'historical': HistoricalEdgeSampler(hist_src, hist_dst, full_data.destinations, seed=2),
        'inductive': InductiveEdgeSampler(hist_src, hist_dst, test_data.sources,
                                          test_data.destinations, full_data.destinations, seed=2),
    }
    for name, sampler in strategies.items():
        r = evaluate(fresh(), test_data, sampler)
        for k in ('auc', 'ap', 'link_acc', 'link_f1'):
            row[f'test_{k}_{name}'] = r[k]
        print(f'{name:12s} auc {r["auc"]:.4f}  ap {r["ap"]:.4f}  '
              f'acc {r["link_acc"]:.4f}  f1 {r["link_f1"]:.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'test_metrics_multi_seed.csv')
    pd.DataFrame([row]).set_index('run').to_csv(out)
    print('\nEdgeBank has no parameters and no seed dependence: one run is exact.')
    print('Wrote', out)


if __name__ == '__main__':
    main()
