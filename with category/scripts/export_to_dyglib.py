"""
Export the processed dataset into DyGLib format, so that the standard dynamic
graph baselines can be trained and evaluated under an identical protocol.

WHY: Reviewer 1 #8, Reviewer 6 #12 and Reviewer 8 #4 all require that baselines
be run with the authors' official implementations on the same splits, the same
negative sampling and the same features — and that no comparison rest on a
baseline scoring at or below chance. DyGLib (Yu et al., NeurIPS 2023) provides
JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL, GraphMixer and DyGFormer behind
one training pipeline, with exactly the random / historical / inductive negative
sampling strategies we already report, and the same 70/15/15 chronological split.

WHAT THIS DOES NOT COVER: TIDFormer, EasyDGL, TF-TGN and TransformerG2G are not
in DyGLib. Run their own repositories if their code is public, or drop them from
the numeric comparison and discuss them conceptually — which is the honest
option for any model whose released code cannot be run faithfully.

DyGLib expects, under DG_data/ (or processed_data/<name>/):
    ml_<name>.csv        columns u, i, ts, label, idx   (node ids start at 1)
    ml_<name>.npy        edge features, shape [#edges + 1, d_e]
    ml_<name>_node.npy   node features, shape [#nodes + 1, d_n]
Row/index 0 is the reserved padding slot in both feature matrices, which is
already how our preprocessing writes them.

USAGE
    python scripts/export_to_dyglib.py --processed-dir data/processed \\
        --name warden --out-dir ~/DyGLib/processed_data/warden
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', default='data/processed')
    p.add_argument('--name', default='warden', help='dataset name DyGLib will refer to')
    p.add_argument('--out-dir', required=True,
                   help='e.g. ~/DyGLib/processed_data/warden')
    args = p.parse_args()

    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.processed_dir, 'new_df.csv'))
    edge_features = np.load(os.path.join(args.processed_dir, 'edge_features.npy'))
    node_features = np.load(os.path.join(args.processed_dir, 'node_features.npy'))

    missing = {'u', 'i', 'ts', 'label', 'idx'} - set(df.columns)
    assert not missing, f'new_df.csv is missing columns: {missing}'

    # DyGLib reads the csv with a leading unnamed index column, exactly as the
    # TGN preprocessing writes it.
    out_df = df[['u', 'i', 'ts', 'label', 'idx']].copy()
    out_df['u'] = out_df['u'].astype(int)
    out_df['i'] = out_df['i'].astype(int)
    out_df['idx'] = out_df['idx'].astype(int)

    assert out_df['u'].min() >= 1 and out_df['i'].min() >= 1, \
        'node ids must start at 1 (index 0 is the padding slot)'
    assert out_df['idx'].min() >= 1, 'edge idx must start at 1'
    assert edge_features.shape[0] == len(out_df) + 1, \
        f'edge features {edge_features.shape[0]} != #edges+1 {len(out_df) + 1}'
    n_nodes = int(max(out_df['u'].max(), out_df['i'].max()))
    assert node_features.shape[0] >= n_nodes + 1, \
        f'node features {node_features.shape[0]} < #nodes+1 {n_nodes + 1}'
    assert out_df['ts'].is_monotonic_increasing, \
        'interactions must be in chronological order'

    csv_path = os.path.join(out, f'ml_{args.name}.csv')
    out_df.to_csv(csv_path, index=True)
    np.save(os.path.join(out, f'ml_{args.name}.npy'), edge_features)
    np.save(os.path.join(out, f'ml_{args.name}_node.npy'), node_features)

    print(f'wrote {csv_path}')
    print(f'  interactions {len(out_df):,}   nodes {n_nodes:,}')
    print(f'  edge features {edge_features.shape}   node features {node_features.shape}')
    print(f'  label column carries the attack category (DyGLib uses it only for')
    print(f'  node classification; link-prediction baselines ignore it)')
    print(f"""
Next, in the DyGLib checkout:

  python train_link_prediction.py --dataset_name {args.name} --model_name TGN \\
      --num_runs 10 --gpu 0 --negative_sample_strategy random --load_best_configs

Repeat with --model_name in: JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL,
GraphMixer, DyGFormer; and --negative_sample_strategy in: random, historical,
inductive. Use --num_runs 10 so the seed count matches the BiTA runs.

Then bring the numbers back with:
  python scripts/import_dyglib_results.py --dyglib-dir ~/DyGLib --name {args.name}
""")


if __name__ == '__main__':
    main()
