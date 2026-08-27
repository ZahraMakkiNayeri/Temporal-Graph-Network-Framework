"""
Interpretability analysis for BiTA, using behavioural attribution rather than
attention weights.

WHY NOT ATTENTION MAPS. Reviewer 1 Comment 19 asks for interpretability evidence
"addressing known limitations of attention-as-explanation", and Reviewer 8
Comment 8 makes the same point. Attention weights are not a faithful account of
what a model used: they can be permuted without changing predictions, and high
attention does not imply high influence. Both analyses here measure what the
model DOES, by intervening on inputs and observing the change in output, so they
are faithful by construction.

TWO ANALYSES

1. Feature-block permutation importance. Each block of the edge-feature vector
   (flow statistics, protocol, destination port, and the optional historical
   category channel) is permuted across the evaluation set while everything else
   is held fixed; the drop in AUC, AP and category macro-F1 measures how much
   the model relies on that block. This also quantifies the category channel
   directly, which is the substance of Reviewer 6 Comment 3.

2. Message-level occlusion. For individual predictions, each message in the
   source node's pending sequence is removed in turn and the prediction is
   recomputed from an identical memory state. This answers the claim actually
   made in the abstract -- which historical interactions influence an alert
   prediction -- and yields a recency profile showing whether the aggregator
   depends on recent messages, distant ones, or the whole window.

USAGE
    python scripts/interpretability.py --processed-dir data/processed_cat \
        --model saved_models --n-probe-edges 25 --out-dir artifacts
"""

import argparse
import glob
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

from utils.data_processing import get_data, compute_time_statistics
from utils.utils import get_neighbor_finder, RandEdgeSampler
from utils.evaluation import eval_edge_prediction_with_categories
from model.extended_tgn import ExtendedTGN


def default_blocks(width):
    """Column spans of the edge-feature vector, matching the preprocessing."""
    if width in (9, 13):        # Warden: flow(1) | port(4) | proto(4) [| category(4)]
        b = {'flow statistics': (0, 1), 'destination port': (1, 5), 'protocol': (5, 9)}
        if width == 13:
            b['historical category'] = (9, 13)
    elif width in (13 + 4, 13):  # NF-UNSW: flow(5) | proto(4) | port(4) [| category(4)]
        b = {'flow statistics': (0, 5), 'protocol': (5, 9), 'destination port': (9, 13)}
        if width == 17:
            b['historical category'] = (13, 17)
    else:
        b = {f'cols {i}-{min(i+4, width)}': (i, min(i + 4, width))
             for i in range(0, width, 4)}
    return b


def build_model(args, nfeat, efeat, train_data, full_data, device):
    m_s, s_s, m_d, s_d = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)
    model = ExtendedTGN(
        neighbor_finder=get_neighbor_finder(full_data, uniform=False),
        node_features=nfeat, edge_features=efeat, device=device,
        n_layers=1, n_heads=args.n_head, dropout=0.1, use_memory=True,
        memory_update_at_start=True, message_dimension=100,
        memory_dimension=nfeat.shape[1], embedding_module_type='graph_attention',
        message_function='mlp', mean_time_shift_src=m_s, std_time_shift_src=s_s,
        mean_time_shift_dst=m_d, std_time_shift_dst=s_d,
        n_neighbors=args.n_neighbors, aggregator_type=args.aggregator,
        memory_updater_type='gru', num_categories=args.num_categories,
        aggregator_hidden_dim=64, aggregator_n_layers=1,
        aggregator_max_seq_len=128).to(device)
    ckpts = []
    for m in (args.model or []):
        ckpts += sorted(glob.glob(os.path.join(m, '*.pth'))) if os.path.isdir(m) else [m]
    if ckpts:
        try:
            model.load_state_dict(torch.load(ckpts[0], map_location=device))
            print('loaded checkpoint:', ckpts[0])
        except RuntimeError as e:
            print('WARNING: checkpoint incompatible with this --processed-dir;',
                  'attribution on an untrained model is meaningless.')
            raise SystemExit(str(e).splitlines()[0])
    else:
        print('WARNING: no checkpoint given — attribution on an untrained model '
              'is meaningless. Pass --model <run>/saved_models.')
    model.eval()
    return model


# ------------------------------------------------------------------ analysis 1
def permutation_importance(model, data, sampler_seed, blocks, repeats, rng, n_neighbors):
    def score():
        model.memory.__init_memory__()
        r = eval_edge_prediction_with_categories(
            model=model, negative_edge_sampler=RandEdgeSampler(
                data.sources, data.destinations, seed=sampler_seed),
            data=data, n_neighbors=n_neighbors)
        return {'auc': r[1], 'ap': r[0], 'category_accuracy': r[4], 'macro_f1': r[8]}

    base = score()
    print('\nbaseline: ' + '  '.join(f'{k} {v:.4f}' for k, v in base.items()))
    original = model.edge_raw_features.clone()
    rows = []
    for name, (lo, hi) in blocks.items():
        drops = []
        for _ in range(repeats):
            perm = torch.from_numpy(
                rng.permutation(original.shape[0] - 1) + 1).to(original.device)
            model.edge_raw_features[1:, lo:hi] = original[perm, lo:hi]
            drops.append(score())
            model.edge_raw_features.copy_(original)
        rec = {'block': name, 'columns': f'{lo}:{hi}'}
        for k in base:
            vals = np.array([d[k] for d in drops])
            rec[f'{k}_drop'] = base[k] - vals.mean()
            rec[f'{k}_drop_std'] = vals.std(ddof=1) if repeats > 1 else 0.0
        rows.append(rec)
        print(f"  {name:22s} AUC {-rec['auc_drop']:+.4f}   "
              f"AP {-rec['ap_drop']:+.4f}   macro-F1 {-rec['macro_f1_drop']:+.4f}"
              f"   (negative = performance falls when this block is destroyed)")
    return base, pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis 2
def message_occlusion(model, data, warm, n_probes, min_messages, n_neighbors,
                      batch_size=200):
    """Remove one pending message at a time and measure the prediction shift.

    Probe edges are selected by walking the evaluation stream and picking edges
    whose source currently holds at least `min_messages` pending messages.
    Uniform sampling does not work: with memory_update_at_start the message
    store is cleared for every node in a batch, so a node's queue only holds
    what it accumulated since its last appearance, and most edges have exactly
    one. Only nodes recurring within a batch build a sequence worth occluding.
    """
    model.memory.__init_memory__()
    with torch.no_grad():
        for b in range((len(warm.sources) + batch_size - 1) // batch_size):
            sl = slice(b * batch_size, min(len(warm.sources), (b + 1) * batch_size))
            if sl.stop <= sl.start:
                continue
            model.compute_edge_probabilities_and_categories(
                warm.sources[sl], warm.destinations[sl],
                warm.destinations[sl][::-1].copy(), warm.timestamps[sl],
                warm.edge_idxs[sl], n_neighbors=n_neighbors)

    def predict_one(k):
        sl = slice(int(k), int(k) + 1)
        with torch.no_grad():
            pos, _, cat = model.compute_edge_probabilities_and_categories(
                data.sources[sl], data.destinations[sl],
                data.destinations[sl][::-1].copy(), data.timestamps[sl],
                data.edge_idxs[sl], n_neighbors=n_neighbors)
        return float(pos.reshape(-1)[0]), cat.reshape(-1).cpu().numpy()

    records, probed, seq_lengths = [], 0, []
    n = len(data.sources)
    for b in range((n + batch_size - 1) // batch_size):
        if probed >= n_probes:
            break
        lo, hi = b * batch_size, min(n, (b + 1) * batch_size)
        if hi <= lo:
            continue
        snap = model.memory.backup_memory()

        for k in range(lo, hi):
            if probed >= n_probes:
                break
            src = int(data.sources[k])
            n_msg = len(model.memory.messages[src])
            if n_msg < min_messages:
                continue
            model.memory.restore_memory(snap)
            full_link, full_cat = predict_one(k)
            model.memory.restore_memory(snap)
            msgs = list(model.memory.messages[src])
            for pos in range(len(msgs)):
                model.memory.restore_memory(snap)
                model.memory.messages[src] = [m for j, m in enumerate(msgs) if j != pos]
                link, cat = predict_one(k)
                records.append({
                    'edge': int(k), 'source': src, 'n_messages': len(msgs),
                    'position_from_end': len(msgs) - 1 - pos,
                    'delta_link': abs(link - full_link),
                    'delta_category': float(np.abs(cat - full_cat).max()),
                })
            seq_lengths.append(len(msgs))
            probed += 1
            model.memory.restore_memory(snap)

        # advance the stream normally from the pre-batch state
        model.memory.restore_memory(snap)
        with torch.no_grad():
            sl = slice(lo, hi)
            model.compute_edge_probabilities_and_categories(
                data.sources[sl], data.destinations[sl],
                data.destinations[sl][::-1].copy(), data.timestamps[sl],
                data.edge_idxs[sl], n_neighbors=n_neighbors)

    if seq_lengths:
        print(f'  probed {probed} edges whose source held >= {min_messages} '
              f'pending messages (median sequence {np.median(seq_lengths):.0f}, '
              f'max {max(seq_lengths)})')
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--model', nargs='*', default=None)
    p.add_argument('--aggregator', default='bigru_transformer')
    p.add_argument('--n-head', type=int, default=2)
    p.add_argument('--n-neighbors', type=int, default=10)
    p.add_argument('--num-categories', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3, help='permutations per block')
    p.add_argument('--n-probe-edges', type=int, default=25)
    p.add_argument('--min-messages', type=int, default=3,
                   help='only probe edges whose source holds at least this '
                        'many pending messages; 1 makes occlusion trivial')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='artifacts')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    new_df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    ef = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    nf = np.load(os.path.join(args.processed_dir, 'node_features.npy'))
    nfeat, efeat, full, train, val, test, _, _ = get_data(
        new_df, ef, nf, different_new_nodes_between_val_and_test=True)
    model = build_model(args, nfeat, efeat, train, full, device)
    os.makedirs(args.out_dir, exist_ok=True)

    blocks = default_blocks(ef.shape[1])
    print(f'edge-feature blocks: ' + ', '.join(f'{k} [{v[0]}:{v[1]}]' for k, v in blocks.items()))
    print('\n=== 1. feature-block permutation importance ===')
    base, imp = permutation_importance(model, test, 2, blocks, args.repeats,
                                       rng, args.n_neighbors)
    imp.to_csv(os.path.join(args.out_dir, 'permutation_importance.csv'), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(imp))
    ax.bar(x - 0.2, imp['auc_drop'], 0.4, yerr=imp['auc_drop_std'], label='AUC drop', capsize=3)
    ax.bar(x + 0.2, imp['macro_f1_drop'], 0.4, yerr=imp['macro_f1_drop_std'],
           label='category macro-F1 drop', capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(imp['block'], rotation=15, ha='right')
    ax.axhline(0, color='k', lw=.8); ax.legend(); ax.set_ylabel('performance drop')
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'permutation_importance.png'), dpi=200)

    print('\n=== 2. message-level occlusion ===')
    from utils.data_processing import Data
    warm = Data(np.concatenate([train.sources, val.sources]),
                np.concatenate([train.destinations, val.destinations]),
                np.concatenate([train.timestamps, val.timestamps]),
                np.concatenate([train.edge_idxs, val.edge_idxs]),
                np.concatenate([train.labels, val.labels]))
    occ = message_occlusion(model, test, warm, args.n_probe_edges,
                            args.min_messages, args.n_neighbors)
    if occ.empty:
        print('  no probed source node had pending messages; nothing to occlude')
        return
    occ.to_csv(os.path.join(args.out_dir, 'message_occlusion.csv'), index=False)

    prof = occ.groupby('position_from_end').agg(
        n=('delta_link', 'size'), mean_delta_link=('delta_link', 'mean'),
        mean_delta_category=('delta_category', 'mean')).head(20)
    print('  influence by message recency (0 = most recent):')
    print(prof.round(5).to_string())
    print(f"\n  edges probed: {occ.edge.nunique()}   "
          f"median sequence length: {occ.groupby('edge').n_messages.first().median():.0f}")
    print(f"  mean |change| from removing one message: link {occ.delta_link.mean():.5f}, "
          f"category logit {occ.delta_category.mean():.5f}")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(prof.index, prof['mean_delta_link'], marker='o', label='link probability')
    ax2.plot(prof.index, prof['mean_delta_category'], marker='s', label='category logit')
    ax2.set_xlabel('message position from most recent (0 = latest)')
    ax2.set_ylabel('mean |change| when removed')
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, 'message_occlusion.png'), dpi=200)

    json.dump({'baseline': base,
               'mean_delta_link': float(occ.delta_link.mean()),
               'mean_delta_category': float(occ.delta_category.mean()),
               'edges_probed': int(occ.edge.nunique())},
              open(os.path.join(args.out_dir, 'interpretability.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/permutation_importance.{{csv,png}}, '
          f'message_occlusion.{{csv,png}}, interpretability.json')
    print('\nBoth analyses are behavioural: they intervene on inputs and measure the')
    print('change in output, so they are faithful by construction and do not rely on')
    print('attention weights as explanations.')


if __name__ == '__main__':
    main()
