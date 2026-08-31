import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, Data
from model.extended_tgn import ExtendedTGN


def build(args, node_features, edge_features, train_data, full_data, device):
    m_s, s_s, m_d, s_d = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)
    model = ExtendedTGN(
        neighbor_finder=get_neighbor_finder(train_data, uniform=False),
        node_features=node_features, edge_features=edge_features, device=device,
        n_layers=1, n_heads=2, dropout=0.1, use_memory=True,
        memory_update_at_start=True, message_dimension=100,
        memory_dimension=node_features.shape[1],
        embedding_module_type='graph_attention', message_function='mlp',
        mean_time_shift_src=m_s, std_time_shift_src=s_s,
        mean_time_shift_dst=m_d, std_time_shift_dst=s_d,
        n_neighbors=args.n_neighbors, aggregator_type='bigru_transformer',
        memory_updater_type='gru', num_categories=args.num_categories,
        aggregator_hidden_dim=64, aggregator_n_layers=1,
        aggregator_max_seq_len=128).to(device)
    if args.model and os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print('loaded', args.model)
    model.eval()
    return model


def predict(model, data, sl, finder, n_neighbors, snapshot):
    """Score one slice from a fixed memory snapshot under a given neighbour finder."""
    model.memory.restore_memory(snapshot)
    model.embedding_module.neighbor_finder = finder
    model.neighbor_finder = finder
    with torch.no_grad():
        pos, _, cat = model.compute_edge_probabilities_and_categories(
            data.sources[sl], data.destinations[sl],
            data.destinations[sl][::-1].copy(),
            data.timestamps[sl], data.edge_idxs[sl], n_neighbors=n_neighbors)
    return pos.detach().cpu().numpy().ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed')
    p.add_argument('--model', default=None)
    p.add_argument('--n-neighbors', type=int, default=10)
    p.add_argument('--num-categories', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--n-probe-edges', type=int, default=100,
                   help='edges probed for the per-edge causality test (Part A)')
    p.add_argument('--n-permutations', type=int, default=5, help='Part B repetitions')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='artifacts')
    args = p.parse_args()

    new_df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    edge_features = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    node_features = np.load(os.path.join(args.processed_dir, 'node_features.npy'))
    device = torch.device(args.device)
    _, _, full_data, train_data, val_data, test_data, _, _ = get_data(
        new_df, edge_features, node_features,
        different_new_nodes_between_val_and_test=True)
    # The neighbour finders must all span the FULL node range: a past-only
    # subset can have a smaller maximum node id, and indexing its adjacency
    # list with a later node then raises IndexError.
    _max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
    model = build(args, node_features, edge_features, train_data, full_data, device)
    os.makedirs(args.out_dir, exist_ok=True)

    # warm the memory over train+val so the test predictions are realistic
    warm = Data(np.concatenate([train_data.sources, val_data.sources]),
                np.concatenate([train_data.destinations, val_data.destinations]),
                np.concatenate([train_data.timestamps, val_data.timestamps]),
                np.concatenate([train_data.edge_idxs, val_data.edge_idxs]),
                np.concatenate([train_data.labels, val_data.labels]))
    full_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=_max_node)
    model.memory.__init_memory__()
    model.embedding_module.neighbor_finder = full_finder
    model.neighbor_finder = full_finder
    bs = args.batch_size
    with torch.no_grad():
        for b in range((len(warm.sources) + bs - 1) // bs):
            sl = slice(b * bs, min(len(warm.sources), (b + 1) * bs))
            if sl.stop <= sl.start:
                continue
            model.compute_edge_probabilities_and_categories(
                warm.sources[sl], warm.destinations[sl],
                warm.destinations[sl][::-1].copy(), warm.timestamps[sl],
                warm.edge_idxs[sl], n_neighbors=args.n_neighbors)
    snapshot = model.memory.backup_memory()

    # ---------------- PART A: causality --------------------------------
    # The cut must be made at EACH EDGE's own timestamp. Cutting at the batch
    # start instead would remove interactions that lie strictly in that edge's
    # past, and the resulting difference would be a legitimate change of
    # available history, not a causality violation.
    idx = np.linspace(0, len(test_data.sources) - 1, args.n_probe_edges).astype(int)
    deltas, p_full_all, p_past_all = [], [], []
    cache = {}
    for k in idx:
        t_i = float(test_data.timestamps[k])
        if t_i not in cache:
            keep = full_data.timestamps < t_i
            cache[t_i] = get_neighbor_finder(
                Data(full_data.sources[keep], full_data.destinations[keep],
                     full_data.timestamps[keep], full_data.edge_idxs[keep],
                     full_data.labels[keep]), uniform=False, max_node_idx=_max_node)
        past_finder = cache[t_i]
        sl = slice(int(k), int(k) + 1)
        p_full = predict(model, test_data, sl, full_finder, args.n_neighbors, snapshot)
        p_past = predict(model, test_data, sl, past_finder, args.n_neighbors, snapshot)
        deltas.append(np.abs(p_full - p_past))
        p_full_all.append(p_full)
        p_past_all.append(p_past)

    d = np.concatenate(deltas)
    p_full_all = np.concatenate(p_full_all)
    p_past_all = np.concatenate(p_past_all)
    r = float(np.corrcoef(p_full_all, p_past_all)[0, 1]) if d.max() > 0 else 1.0

    stats = {
        'n_edges': int(d.size),
        'delta_max': float(d.max()),
        'delta_mean': float(d.mean()),
        'delta_median': float(np.median(d)),
        'frac_below_1e-6': float((d < 1e-6).mean()),
        'frac_below_1e-4': float((d < 1e-4).mean()),
        'frac_below_1e-2': float((d < 1e-2).mean()),
        'pearson_r': r,
        'strictly_causal': bool(d.max() == 0.0),
    }
    print('\n--- PART A: temporal causality (per-edge cut at t_i) ---')
    for k1, v1 in stats.items():
        print(f'  {k1:20s} {v1}')
    print('  ' + ('STRICTLY CAUSAL: removing every interaction at or after t_i leaves\n'
                  '  the prediction bit-identical.' if stats['strictly_causal'] else
                  'NOT bit-identical — report the measured maximum honestly.'))

    np.save(os.path.join(args.out_dir, 'causality_delta.npy'), d)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(p_full_all, p_past_all, s=12, alpha=.6)
    ax[0].plot([0, 1], [0, 1], 'r--', lw=1)
    ax[0].set_xlabel('prediction with future edges present')
    ax[0].set_ylabel('prediction, past-only graph')
    ax[0].set_title(f'Pearson r = {r:.4f}')
    ax[1].hist(d, bins=40)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$|\Delta|$')
    ax[1].set_ylabel('count (log)')
    ax[1].set_title(rf'$\Delta_{{\max}}$ = {d.max():.2e}')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'causality.png'), dpi=200)

    # ---------------- PART B: batch-grouping invariance ----------------
    # TGN forbids updating memory to an earlier timestamp, so mini-batches
    # cannot simply be shuffled: chronological order is structural, not a
    # choice. What CAN legitimately vary is where the batch boundaries fall.
    # We therefore keep chronological order and shift the batching offset,
    # then measure per-edge prediction variance across groupings. This is a
    # DIFFERENT quantity from Part A (a variance, not a mean difference) and
    # must be reported separately.
    _n_test = len(test_data.sources)
    _p0 = min(6 * bs, max(0, _n_test - bs))
    probe = slice(_p0, min(_p0 + bs, _n_test))
    preds = []
    offsets = [o % bs for o in [0, 17, 41, 73, 101]][:args.n_permutations]
    for off in offsets:
        model.memory.restore_memory(snapshot)
        model.embedding_module.neighbor_finder = full_finder
        model.neighbor_finder = full_finder
        pos = 0
        with torch.no_grad():
            while pos < probe.start:
                step = max(1, (bs - off) if (pos == 0 and off) else bs)
                stop = min(probe.start, pos + step)
                sl = slice(pos, stop)
                if stop > pos:
                    model.compute_edge_probabilities_and_categories(
                        test_data.sources[sl], test_data.destinations[sl],
                        test_data.destinations[sl][::-1].copy(),
                        test_data.timestamps[sl], test_data.edge_idxs[sl],
                        n_neighbors=args.n_neighbors)
                pos = stop
        preds.append(predict(model, test_data, probe, full_finder,
                             args.n_neighbors, model.memory.backup_memory()))
    var = np.var(np.stack(preds), axis=0, ddof=1)
    order_stats = {'n_groupings': len(preds), 'n_edges': int(var.size),
                   'variance_mean': float(var.mean()),
                   'variance_max': float(var.max()),
                   'frac_below_1e-2': float((var < 1e-2).mean())}
    print('\n--- PART B: batch-grouping invariance ---')
    for k2, v2 in order_stats.items():
        print(f'  {k2:20s} {v2}')
    print('  NOTE: Part A reports a mean ABSOLUTE DIFFERENCE, Part B a per-edge')
    print('  VARIANCE. They are different quantities and must not share a value.')

    np.save(os.path.join(args.out_dir, 'order_variance.npy'), var)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(var, bins=40)
    ax2.set_xlabel('per-edge prediction variance across batch groupings')
    ax2.set_ylabel('count')
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, 'order_invariance.png'), dpi=200)

    json.dump({'causality': stats, 'order_invariance': order_stats},
              open(os.path.join(args.out_dir, 'causality_stats.json'), 'w'), indent=2)
    print('\nWrote', args.out_dir + '/causality_stats.json and two figures')


if __name__ == '__main__':
    main()
