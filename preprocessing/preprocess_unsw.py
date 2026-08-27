"""
Preprocess NF-UNSW-NB15-v2 into the artifacts run_train.py consumes, using the
same leakage-free protocol as the Warden pipeline.

TWO PROPERTIES OF THIS DATASET MUST BE DISCLOSED IN THE MANUSCRIPT
------------------------------------------------------------------
1. NF-UNSW-NB15-v2 has NO timestamp column. The temporal order used here is the
   order of rows in the released file, which reflects the order in which flows
   were exported. This is a proxy for arrival order, not a recorded time, and
   the paper must say so. Timestamps are synthesised as one second per row so
   that the chronological split and the temporal machinery are well defined.

2. The full file has 2.39M flows; one epoch over it takes hours. We therefore
   subsample. The default is a uniform random subsample that PRESERVES the
   natural class proportions and the relative order of the retained rows; the
   sampling rate is recorded in preprocess_config.json and must be stated.

WHAT IS AND IS NOT IN THE FEATURES (Reviewer 6, Comment 3)
----------------------------------------------------------
  edge features = [ standardised log flow statistics | L7_PROTO emb | dst-port emb ]
                  (+ 4-dim historical-category channel only with
                   --include-category-features)
  node features = [ role (+1 source / -1 destination) | modal dst-port emb |
                    modal protocol emb ]  -- built per NODE from TRAINING-window
                  interactions only, never per row, and never label-derived.
All vocabularies and standardisation statistics are fit on the training window
alone; values unseen there map to a reserved UNK index.

The attack category is the prediction target. It enters the inputs only through
the optional historical-category channel on EDGE features, which is causally
bounded (an edge's own category never reaches its own prediction; verified by
scripts/self_leak_check.py).

USAGE
    python -m preprocessing.preprocess_unsw --raw data/NF-UNSW-NB15.csv \
        --out-dir data/unsw_processed --n-interactions 150000
    python -m preprocessing.preprocess_unsw --raw data/NF-UNSW-NB15.csv \
        --out-dir data/unsw_processed_cat --n-interactions 150000 \
        --include-category-features
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EMB_DIM = 4
FLOW_COLS = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
             'FLOW_DURATION_MILLISECONDS']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/NF-UNSW-NB15.csv')
    p.add_argument('--out-dir', default='data/unsw_processed')
    p.add_argument('--n-interactions', type=int, default=150000,
                   help='subsample size; 0 keeps every row (very slow to train)')
    p.add_argument('--include-category-features', action='store_true')
    p.add_argument('--balance', action='store_true',
                   help='reproduce the old balance-before-split protocol (not recommended)')
    p.add_argument('--keep-all-attacks', action='store_true',
                   help='retain every attack flow and subsample only Benign, so the '
                        'rare classes stay evaluable (changes the class balance; disclose it)')
    p.add_argument('--benign-label', default='Benign')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    usecols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL',
               'L7_PROTO', 'Attack'] + FLOW_COLS
    df = pd.read_csv(args.raw, usecols=usecols)
    print(f'loaded {len(df):,} flows')
    print('class distribution:\n', df['Attack'].value_counts().to_string(), '\n')

    # ---- subsample, preserving order and class proportions -----------------
    if args.n_interactions and args.n_interactions < len(df):
        n_before = len(df)
        if args.keep_all_attacks:
            # Uniform subsampling leaves the rare classes unevaluable (Worms
            # would retain ~10 flows). Keep every attack flow and subsample the
            # Benign majority instead. This deliberately changes the class
            # balance and MUST be stated in the manuscript.
            atk = np.where(df['Attack'].values != args.benign_label)[0]
            ben = np.where(df['Attack'].values == args.benign_label)[0]
            n_ben = max(0, args.n_interactions - len(atk))
            keep = np.sort(np.concatenate(
                [atk, rng.choice(ben, min(n_ben, len(ben)), replace=False)]))
            print(f'stratified subsample: kept all {len(atk):,} attack flows + '
                  f'{len(keep)-len(atk):,} of {len(ben):,} Benign')
        else:
            keep = np.sort(rng.choice(n_before, args.n_interactions, replace=False))
            print(f'uniform subsample {n_before:,} -> {args.n_interactions:,} '
                  f'({100*args.n_interactions/n_before:.2f}%), class proportions preserved')
        df = df.iloc[keep].reset_index(drop=True)
        print(f'  -> {len(df):,} flows, order preserved')
    print('class distribution after subsampling:\n',
          df['Attack'].value_counts().to_string(), '\n')

    if args.balance:
        from sklearn.utils import resample
        target = int(df['Attack'].value_counts().median())
        parts = [resample(g, replace=len(g) < target, n_samples=target,
                          random_state=args.seed) for _, g in df.groupby('Attack')]
        df = pd.concat(parts).sort_index().reset_index(drop=True)
        print(f'WARNING: balance-before-split active, {target} per class, '
              f'{int(df.duplicated().sum()):,} duplicated rows')

    # ---- synthetic timeline: row order, one second apart ------------------
    df['ts'] = np.arange(len(df), dtype=np.float64)

    # ---- bipartite reindex: sources 1..S, destinations S+1..S+D -----------
    src_codes, _ = pd.factorize(df['IPV4_SRC_ADDR'])
    dst_codes, _ = pd.factorize(df['IPV4_DST_ADDR'])
    n_src = src_codes.max() + 1
    df['u'] = src_codes + 1
    df['i'] = dst_codes + n_src + 1
    labels, cat_index = pd.factorize(df['Attack'])
    df['label'] = labels
    df['idx'] = np.arange(1, len(df) + 1)          # 1-based; row 0 is padding
    print(f'graph: {df.u.nunique():,} sources, {df.i.nunique():,} destinations, '
          f'{len(cat_index)} categories {list(cat_index)}')

    # ---- training window: same 70th-percentile rule as get_data -----------
    val_time = float(np.quantile(df.ts.values, 0.70))
    train_mask = df.ts.values <= val_time
    print(f'{train_mask.sum():,} / {len(df):,} flows in the training window')

    def build_vocab(values):
        seen, out = set(), {}
        for v in values[train_mask]:
            if v not in seen:
                seen.add(v); out[v] = len(out) + 1
        return out

    # Destination ports are bucketed rather than embedded raw: the raw column
    # holds tens of thousands of distinct values, most of them ephemeral
    # per-flow numbers, so embedding them directly injects noise and leaves
    # most val/test values unseen. Well-known ports keep their identity.
    def bucket_port(v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 'other'
        if v < 1024:
            return f'wellknown:{v}'
        if v < 49152:
            return 'registered'
        return 'ephemeral'
    df['PORT_BUCKET'] = [bucket_port(v) for v in df['L4_DST_PORT'].values]

    proto_vocab = build_vocab(df['L7_PROTO'].values)
    port_vocab = build_vocab(df['PORT_BUCKET'].values)
    proto_idx = np.array([proto_vocab.get(v, 0) for v in df['L7_PROTO'].values])
    port_idx = np.array([port_vocab.get(v, 0) for v in df['PORT_BUCKET'].values])
    print(f'train-fitted vocabularies: L7_PROTO {len(proto_vocab)}, '
          f'dst-port buckets {len(port_vocab)}; '
          f'{int((proto_idx==0).sum() + (port_idx==0).sum()):,} val/test values -> UNK')

    proto_tab = nn.Embedding(len(proto_vocab) + 1, EMB_DIM, padding_idx=0)
    port_tab = nn.Embedding(len(port_vocab) + 1, EMB_DIM, padding_idx=0)
    with torch.no_grad():
        proto_emb = proto_tab(torch.tensor(proto_idx, dtype=torch.long)).numpy()
        port_emb = port_tab(torch.tensor(port_idx, dtype=torch.long)).numpy()

    # ---- flow statistics: log1p, standardised with TRAIN statistics -------
    flow = np.log1p(df[FLOW_COLS].astype(float).values)
    mu, sd = flow[train_mask].mean(0), flow[train_mask].std(0) + 1e-9
    flow_z = (flow - mu) / sd

    edge_rows = np.concatenate([flow_z, proto_emb, port_emb], axis=1)
    if args.include_category_features:
        n_cat = int(df.label.max()) + 1
        cat_tab = nn.Embedding(n_cat + 1, EMB_DIM, padding_idx=0)
        with torch.no_grad():
            cat_emb = cat_tab(torch.tensor(df.label.values + 1, dtype=torch.long)).numpy()
        edge_rows = np.concatenate([edge_rows, cat_emb], axis=1)
        print('WARNING: historical-category channel INCLUDED in edge features')
    edge_features = np.vstack([np.zeros((1, edge_rows.shape[1])),
                               edge_rows]).astype(np.float32)

    # ---- node features: per NODE, training window only, label-free --------
    u_arr, i_arr = df.u.values, df.i.values
    max_node = int(i_arr.max())
    cnt_port, cnt_proto = defaultdict(Counter), defaultdict(Counter)
    for r in np.where(train_mask)[0]:
        for node in (u_arr[r], i_arr[r]):
            cnt_port[node][port_idx[r]] += 1
            cnt_proto[node][proto_idx[r]] += 1
    sources, dests = set(u_arr.tolist()), set(i_arr.tolist())

    node_dim = 1 + 2 * EMB_DIM
    node_features = np.zeros((max_node + 1, node_dim), dtype=np.float32)
    with torch.no_grad():
        for node in range(1, max_node + 1):
            role = 1.0 if node in sources else (-1.0 if node in dests else 0.0)
            pm = cnt_port[node].most_common(1)[0][0] if node in cnt_port else 0
            rm = cnt_proto[node].most_common(1)[0][0] if node in cnt_proto else 0
            node_features[node] = np.concatenate(
                [[role], port_tab(torch.tensor(pm)).numpy(),
                 proto_tab(torch.tensor(rm)).numpy()])

    cold = int((np.abs(node_features[1:, 1:]).sum(1) == 0).sum())
    print(f'edge_features {edge_features.shape}, node_features {node_features.shape} '
          f'({cold:,} nodes without training-window history)')

    # ---- save -------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[['u', 'i', 'ts', 'label', 'idx']].copy()
    out.to_csv(os.path.join(args.out_dir, 'new_df.csv'), index=False)
    np.save(os.path.join(args.out_dir, 'edge_features.npy'), edge_features)
    np.save(os.path.join(args.out_dir, 'node_features.npy'), node_features)
    json.dump({'dataset': 'NF-UNSW-NB15-v2',
               'balance': args.balance,
               'include_category_features': args.include_category_features,
               'n_interactions': int(len(df)),
               'n_nodes': int(max_node),
               'n_categories': int(len(cat_index)),
               'categories': list(map(str, cat_index)),
               'subsample_n': args.n_interactions,
               'subsample_seed': args.seed,
               'subsample_scheme': ('all attacks + Benign subsample'
                                    if args.keep_all_attacks else 'uniform'),
               'port_encoding': 'bucketed (well-known / registered / ephemeral)',
               'n_distinct_destinations': int(df.i.nunique()),
               'timeline': 'SYNTHETIC: row order of the released file, 1 s per row; '
                           'the dataset has no timestamp column',
               'node_feature_dim': node_dim},
              open(os.path.join(args.out_dir, 'preprocess_config.json'), 'w'), indent=2)
    json.dump({str(k): str(v) for k, v in enumerate(cat_index)},
              open(os.path.join(args.out_dir, 'category_mapping.json'), 'w'), indent=2)
    print('saved to', args.out_dir)
    n_dst = int(df.i.nunique())
    print(f'\nIMPORTANT: only {n_dst} distinct destinations exist, so link-ranking '
          f'candidates must not exceed {n_dst - 1}: pass '
          f'--link-ranking-candidates {n_dst - 1} to run_train.py (sampling more '
          f'than that repeats nodes and distorts MRR/Hits).')
    print(f'IMPORTANT: node feature dimension is {node_dim}; pass '
          f'--memory-dim {node_dim} to run_train.py (TGN adds memory to node '
          f'features, so the two must match).')


if __name__ == '__main__':
    main()
