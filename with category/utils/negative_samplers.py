# =====================================================================
# negative_samplers.py — NEW (Reviewer 6, Comment 2)
#
# The paper's evaluation used RandEdgeSampler: uniform random negative
# destinations, 1:1, no collision checking. Poursafaei et al. (NeurIPS
# 2022, [15]) showed this protocol inflates AUC/AP for dynamic link
# prediction. This cell adds:
#   * CollisionCheckedRandEdgeSampler — random negatives that are
#     guaranteed not to collide with a positive edge of the current batch;
#   * HistoricalEdgeSampler — negatives drawn from edges OBSERVED in the
#     history (train ∪ val): "will this previously-seen pair interact
#     again right now?" — the hard case for memory-based models;
#   * InductiveEdgeSampler — negatives drawn from evaluation-period edges
#     NEVER seen in the history.
# All samplers keep the destination-corruption convention of the
# pipeline (the source of each positive edge is kept, only the
# destination is replaced) and fall back to collision-checked random
# sampling when a source has no usable candidate, as in [15].
# =====================================================================

import numpy as np
from collections import defaultdict


class _BatchContextSampler:
    """Base: samplers that need the current batch's positive edges."""
    needs_batch_context = True

    def __init__(self, all_dst, seed=None):
        self.all_dst = np.unique(all_dst)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

    def _random_non_colliding(self, source, batch_edges):
        for _ in range(100):
            d = self.all_dst[self.random_state.randint(0, len(self.all_dst))]
            if (source, d) not in batch_edges:
                return d
        return d  # graph nearly complete for this source; accept collision

    def _candidates(self, source):
        return None  # override

    def sample(self, size, sources=None, destinations=None):
        assert sources is not None and destinations is not None, \
            "this sampler needs the current batch (sources, destinations)"
        batch_edges = set(zip(np.asarray(sources).tolist(),
                              np.asarray(destinations).tolist()))
        negatives = np.empty(size, dtype=self.all_dst.dtype)
        for i in range(size):
            s = sources[i]
            cands = self._candidates(s)
            if cands is not None and len(cands) > 0:
                cands = cands[~np.isin(cands, destinations[i:i+1])] if len(cands) else cands
                valid = [d for d in cands if (s, d) not in batch_edges]
                if valid:
                    negatives[i] = valid[self.random_state.randint(0, len(valid))]
                    continue
            negatives[i] = self._random_non_colliding(s, batch_edges)
        return None, negatives


class CollisionCheckedRandEdgeSampler(_BatchContextSampler):
    """Uniform random negatives with collision checking against the
    current batch's positive edges."""
    def __init__(self, src_list, dst_list, seed=None):
        super().__init__(dst_list, seed)


class HistoricalEdgeSampler(_BatchContextSampler):
    """Negatives from the HISTORICAL edge set (Poursafaei et al. [15]):
    for a positive (s, d, t), the negative destination d' is sampled from
    {d' : (s, d') was observed in train ∪ val}, excluding the current
    batch's positives of s. Falls back to collision-checked random when
    s has no history (e.g. new nodes)."""
    def __init__(self, hist_src, hist_dst, all_dst, seed=None):
        super().__init__(all_dst, seed)
        by_src = defaultdict(set)
        for s, d in zip(hist_src, hist_dst):
            by_src[s].add(d)
        self.hist_by_src = {s: np.array(sorted(ds)) for s, ds in by_src.items()}

    def _candidates(self, source):
        return self.hist_by_src.get(source)


class InductiveEdgeSampler(_BatchContextSampler):
    """Negatives from INDUCTIVE edges (Poursafaei et al. [15]): edges that
    occur during the evaluation period but were NEVER observed in the
    history (train ∪ val). Falls back to collision-checked random when a
    source has no inductive candidate."""
    def __init__(self, hist_src, hist_dst, eval_src, eval_dst, all_dst, seed=None):
        super().__init__(all_dst, seed)
        hist_set = set(zip(np.asarray(hist_src).tolist(), np.asarray(hist_dst).tolist()))
        by_src = defaultdict(set)
        for s, d in zip(eval_src, eval_dst):
            if (s, d) not in hist_set:
                by_src[s].add(d)
        self.ind_by_src = {s: np.array(sorted(ds)) for s, ds in by_src.items()}

    def _candidates(self, source):
        return self.ind_by_src.get(source)
