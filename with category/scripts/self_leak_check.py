"""
self_leak_check.py — empirical causality test for the historical-category variant
(Reviewer 6, Comment 3: "specify which features of which events are visible when
each category is predicted").

CLAIM UNDER TEST
    When the category channel is enabled in EDGE features
    (preprocess --include-category-features), the model may read the categories
    of interactions that occurred strictly BEFORE time t, but never the category
    of the very edge whose category it is predicting at t.

METHOD — a counterfactual on the label itself. Memory is warmed over the
preceding batches and snapshotted. Then the SAME batch is run twice from that
identical state, the second time after flipping the category values stored in
the edge-feature matrix for exactly the edges of that batch:

  TEST 1  (no self-leak)      ONE edge's category is flipped at a time and only
                              THAT edge's prediction is compared; it must be
                              BIT-IDENTICAL. Flipping a whole batch would not
                              isolate self-leakage: TGN cuts temporal
                              neighbours at each edge's own timestamp, not at
                              batch boundaries, so an edge legitimately sees
                              the categories of edges EARLIER IN THE SAME BATCH
                              (t_j < t_i). That is past information, and the
                              per-edge design separates it from self-leakage.

  TEST 2  (not vacuous)       predictions for SUBSEQUENT batches must DIFFER.
                              This proves the channel is genuinely wired: the
                              flipped labels do reach the model — one batch
                              later, as history. Without this control, TEST 1
                              would also "pass" for a model that ignores the
                              channel entirely.

  TEST 3  (strictly past)     flipping the categories of edges that occur
                              AFTER the tested batch must leave that batch's
                              predictions BIT-IDENTICAL — no lookahead through
                              temporal neighbour sampling.

Together: the model uses past categories (TEST 2) and only past categories
(TESTS 1 and 3).

USAGE
    python scripts/self_leak_check.py --processed-dir data/processed_cat
    # optional: --batch-index 12 --batch-size 128 --n-following 3

Run it on the category-enabled preprocessing. On the default (category-free)
preprocessing there is no channel to flip, and the script says so and exits.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from model.extended_tgn import ExtendedTGN

EMB_DIM = 4          # category block width used by preprocessing/preprocess_data.py
BASE_EDGE_DIM = 9    # flow(1) + port(4) + proto(4)


def build_model(args, node_features, edge_features, train_data, full_data, device):
    mean_src, std_src, mean_dst, std_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)
    model = ExtendedTGN(
        neighbor_finder=get_neighbor_finder(train_data, uniform=False),
        node_features=node_features, edge_features=edge_features, device=device,
        n_layers=1, n_heads=2, dropout=0.1,
        use_memory=True, memory_update_at_start=True,
        message_dimension=100, memory_dimension=node_features.shape[1],
        embedding_module_type='graph_attention', message_function='mlp',
        mean_time_shift_src=mean_src, std_time_shift_src=std_src,
        mean_time_shift_dst=mean_dst, std_time_shift_dst=std_dst,
        n_neighbors=args.n_neighbors, aggregator_type='bigru_transformer',
        memory_updater_type='gru', num_categories=args.num_categories,
        aggregator_hidden_dim=64, aggregator_n_layers=1,
        aggregator_max_seq_len=128).to(device)
    model.eval()   # dropout off -> the two worlds are comparable bit for bit
    return model


def category_blocks(edge_features, labels, n_categories):
    """Recover the frozen embedding block used for each category value."""
    blocks = {}
    for c in range(n_categories):
        rows = np.where(labels == c)[0]
        if len(rows):
            # edge_features row i+1 corresponds to interaction i (row 0 is padding)
            blocks[c] = edge_features[rows[0] + 1, BASE_EDGE_DIM:].copy()
    return blocks


def run_batch(model, data, sl, n_neighbors):
    with torch.no_grad():
        pos, neg, cat = model.compute_edge_probabilities_and_categories(
            data.sources[sl], data.destinations[sl],
            data.destinations[sl][::-1].copy(),          # fixed "negatives": no RNG
            data.timestamps[sl], data.edge_idxs[sl], n_neighbors=n_neighbors)
    return pos.detach().cpu().numpy(), cat.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed_cat')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--batch-index', type=int, default=10,
                   help='which batch to flip (must be > 0 so memory is warm)')
    p.add_argument('--n-following', type=int, default=3,
                   help='how many subsequent batches to check for the expected change')
    p.add_argument('--n-probe-edges', type=int, default=12,
                   help='how many individual edges to probe in TEST 1')
    p.add_argument('--n-neighbors', type=int, default=10)
    p.add_argument('--num-categories', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    new_df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    edge_features = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    node_features = np.load(os.path.join(args.processed_dir, 'node_features.npy'))

    cfg_path = os.path.join(args.processed_dir, 'preprocess_config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    has_channel = edge_features.shape[1] >= BASE_EDGE_DIM + EMB_DIM

    print(f'edge_features {edge_features.shape}, node_features {node_features.shape}')
    print(f'preprocess config: {cfg}')
    if not has_channel:
        print('\nThis preprocessing has NO category channel in the edge features '
              '(the default, leakage-free setting) — there is nothing to flip, and '
              'the category cannot reach the model at all.\n'
              'Run preprocess_data.py with --include-category-features to test the '
              'historical-category variant.')
        return

    device = torch.device(args.device)
    _, _, full_data, train_data, val_data, test_data, _, _ = get_data(
        new_df, edge_features, node_features)
    model = build_model(args, node_features, edge_features, train_data, full_data, device)

    labels = new_df.label.values.astype(int)
    blocks = category_blocks(edge_features, labels, args.num_categories)

    data = train_data
    bs, k = args.batch_size, args.batch_index
    assert (k + args.n_following + 1) * bs <= len(data.sources), 'batch index out of range'
    target = slice(k * bs, (k + 1) * bs)
    following = [slice((k + j) * bs, (k + j + 1) * bs) for j in range(1, args.n_following + 1)]
    future = slice((k + 1) * bs, (k + 1 + args.n_following) * bs)

    # ---- warm memory over batches 0..k-1, then snapshot -------------------
    model.memory.__init_memory__()
    for b in range(k):
        run_batch(model, data, slice(b * bs, (b + 1) * bs), args.n_neighbors)
    snapshot = model.memory.backup_memory()

    def world(perturb_rows=None):
        """Replay the target batch (and the following ones) from the snapshot,
        optionally with the category block of `perturb_rows` flipped."""
        model.memory.restore_memory(snapshot)
        original = model.edge_raw_features.clone()
        if perturb_rows is not None:
            for row in perturb_rows:
                lab = labels[row - 1]                       # row = edge_idx (1-based)
                flipped = blocks[(lab + 1) % args.num_categories]
                model.edge_raw_features[row, BASE_EDGE_DIM:BASE_EDGE_DIM + EMB_DIM] = \
                    torch.from_numpy(flipped).to(model.edge_raw_features.dtype).to(device)
        out = [run_batch(model, data, target, args.n_neighbors)]
        for sl in following:
            out.append(run_batch(model, data, sl, args.n_neighbors))
        model.edge_raw_features.copy_(original)
        return out

    target_rows = data.edge_idxs[target]
    future_rows = data.edge_idxs[future]

    base = world(None)
    flipped_batch = world(target_rows)
    flipped_future = world(future_rows)

    def dmax(a, b):
        return max(float(np.abs(a[i] - b[i]).max()) for i in range(2))

    # ---- TEST 1: per-edge counterfactual ---------------------------------
    # Flip ONE edge's category, compare ONLY that edge's own prediction.
    probe_positions = np.linspace(0, len(target_rows) - 1, min(args.n_probe_edges,
                                  len(target_rows))).astype(int)
    worst, worst_pos = 0.0, None
    for pos in probe_positions:
        out = world([target_rows[pos]])
        d = max(float(np.abs(out[0][i][pos] - base[0][i][pos]).max()) for i in range(2))
        if d > worst:
            worst, worst_pos = d, int(pos)

    print('\n' + '=' * 72)
    ok1 = worst == 0.0
    print(f'TEST 1  own category flipped, one edge at a time '
          f'({len(probe_positions)} probe edges) -> that edge\'s own prediction '
          f'{"UNCHANGED" if ok1 else "CHANGED"}')
    print(f'        worst max |diff| over probes = {worst:.3e}'
          + ('' if ok1 else f' (at batch position {worst_pos})'))
    print(f'        {"PASS: an edge cannot see its own category." if ok1 else "FAIL: SELF-LEAK."}')

    deltas = [dmax(base[j], flipped_batch[j]) for j in range(1, len(base))]
    ok2 = any(d > 0.0 for d in deltas)
    print(f'\nTEST 2  whole batch flipped -> following batches changed by '
          f'{["%.3e" % d for d in deltas]}')
    print(f'        {"PASS: past categories DO reach later predictions (channel is live)." if ok2 else "FAIL: channel appears inert — TEST 1 would be vacuous."}')

    d3 = dmax(base[0], flipped_future[0])
    ok3 = d3 == 0.0
    print(f'\nTEST 3  FUTURE edges\' categories flipped -> tested batch '
          f'{"UNCHANGED" if ok3 else "CHANGED"} (max |diff| = {d3:.3e})')
    print(f'        {"PASS: no lookahead through neighbour sampling." if ok3 else "FAIL: future information reaches the prediction."}')

    print('=' * 72)
    verdict = ok1 and ok2 and ok3
    print('VERDICT:', 'ALL CHECKS PASSED — the model uses past categories, and only past '
          'categories.' if verdict else 'CHECKS FAILED — see above.')
    print('\nNote for the manuscript: TGN cuts temporal neighbours at each edge\'s own')
    print('timestamp, so an edge may read the categories of edges earlier in the same')
    print('batch. That is past information, but it should be stated explicitly.')
    print('\nSuggested manuscript sentence:')
    print('  "When predicting the category of interaction (u, v, t), the model observes the')
    print('   categories of interactions strictly earlier than t and never that of the')
    print('   interaction itself; we verify this empirically by flipping the label of each')
    print('   evaluated edge and confirming its prediction is unchanged, while predictions')
    print('   for subsequent interactions do change."')
    sys.exit(0 if verdict else 1)


if __name__ == '__main__':
    main()
