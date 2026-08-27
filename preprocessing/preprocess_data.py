"""Preprocess the raw Warden CSVs into the artifacts run_train.py consumes.

Usage:
    python -m preprocessing.preprocess_data --raw-dir data/raw --out-dir data/processed
Flags:
    --balance                    reproduce the old median-resampling protocol (Comment 4)
    --include-category-features  historical-category ablation channel (Comment 3)
    --emit-table2                write verbatim Table 2 LaTeX rows to <out-dir>/table2.tex
"""
import argparse, os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--raw-dir', default='data/raw')
parser.add_argument('--out-dir', default='data/processed')
parser.add_argument('--balance', action='store_true')
parser.add_argument('--include-category-features', action='store_true')
parser.add_argument('--emit-table2', action='store_true')
ARGS = parser.parse_args()


# File: preprocessing/balance_classes.py — REVISED (Reviewer 6, Comment 4)
#
# The released pipeline resampled every class to the MEDIAN class size
# (undersampling majorities, oversampling minorities WITH replacement) over
# the full week, BEFORE the temporal split. Consequences the reviewer named:
#   * every classification metric is computed on an artificial test
#     distribution, not the real one;
#   * oversampled minority rows are exact duplicates sharing one timestamp,
#     so a test-window minority class may consist of a handful of unique
#     events copied many times — classifying those few events correctly
#     yields the per-class accuracies of 1.0 seen in Table 4;
#   * undersampling discards most of the majority-class evidence.
#
# FIX: BALANCE_CLASSES = False (default) keeps the ORIGINAL distribution;
# class imbalance is instead handled in the LOSS via per-class weights
# computed on the TRAINING split only (see the training cells). Set the
# flag to True to reproduce the old protocol for comparison — if reported,
# it must be labelled as a resampled-distribution result.

import pandas as pd
from sklearn.utils import resample

BALANCE_CLASSES = ARGS.balance

# FIX (1c): portable paths — DATA_DIR is set in the first cell (Colab or local).
file_paths = [os.path.join(ARGS.raw_dir, name) for name in [
    '11March_e.csv', '12March_e.csv', '13March_e.csv', '14March_e.csv',
    '15March_e.csv', '16March_e.csv', '17March_e.csv',
]]

dfs = [pd.read_csv(fp) for fp in file_paths]
df = pd.concat(dfs, ignore_index=True)

# Keep a verbatim string copy BEFORE any type conversion — used by the
# Table 2 generator below to print records exactly as released.
raw_records_df = df.copy()

# Convert 'DetectTime' column to datetime
df['DetectTime'] = pd.to_datetime(df['DetectTime'], format='ISO8601', utc=True)

print('Original class distribution:')
print(df['Category'].value_counts())

if BALANCE_CLASSES:
    grouped = df.groupby('Category')
    class_sizes = grouped.size().sort_values()
    target_size = int(class_sizes.median())

    balanced_dfs = []
    for label, group in grouped:
        if len(group) > target_size:
            balanced_dfs.append(resample(group, replace=False, n_samples=target_size, random_state=42))
        elif len(group) < target_size:
            balanced_dfs.append(resample(group, replace=True, n_samples=target_size, random_state=42))
        else:
            balanced_dfs.append(group)

    balanced_df = pd.concat(balanced_dfs).sort_values('DetectTime').reset_index(drop=True)
    n_dup = int(balanced_df.duplicated().sum())
    print('\nWARNING: resampled-distribution protocol active (Comment 4).')
    print(f'Balanced class distribution (target = {target_size} per class, {n_dup} duplicated rows):')
    print(balanced_df['Category'].value_counts())
else:
    balanced_df = df.sort_values('DetectTime').reset_index(drop=True)
    print('\nBALANCE_CLASSES = False: original distribution preserved; '
          'class imbalance is handled by loss weighting (see training cells).')


# ---- glue (was: df = balanced_df; sort) ----
df = balanced_df
df = df.sort_values('DetectTime')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # Convert SourceIP and TargetIP to strings to ensure they are hashable
    df['SourceIP'] = df['SourceIP'].astype(str)
    df['TargetIP'] = df['TargetIP'].astype(str)

    # Initialize label encoders for categorical features
    proto_encoder = LabelEncoder()
    attack_type_encoder = LabelEncoder()
    port_type_encoder = LabelEncoder()

    # Fit and transform categorical features to numeric values
    df['Proto_encoded'] = proto_encoder.fit_transform(df['Proto'])
    # NOTE (Reviewer 6, Comment 3): fitting the LABEL encoder on the full dataset is
    # acceptable — it only fixes the label vocabulary (closed-set assumption, Sec. 4.2)
    # and the encoded value is used as the prediction TARGET, never as an input feature
    # (see the leakage-free feature-construction cell below).
    df['Category_encoded'] = attack_type_encoder.fit_transform(df['Category'])
    df['Port_encoded'] = port_type_encoder.fit_transform(df['Port'])  # FIX (Comment 3): was refitting proto_encoder

    # Map encoded categories back to their original labels
    category_mapping = {index: label for index, label in enumerate(attack_type_encoder.classes_)}

    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    # Encode SourceIP and TargetIP as integers
    df['SourceIP_encoded'], _ = pd.factorize(df['SourceIP'])
    df['TargetIP_encoded'], _ = pd.factorize(df['TargetIP'])

    for idx, row in df.iterrows():
        u = row['SourceIP_encoded']
        i = row['TargetIP_encoded']
        ts = row['DetectTime'].timestamp()
        label = row['Category_encoded']

        # Extracting FlowCount, Port, and Proto as features
        flow_count = row['FlowCount']
        port = row['Port_encoded']
        proto = row['Proto_encoded']
        attack_type = row['Category_encoded']

        #feat = np.zeros(172)
        feat = np.array([flow_count, port, proto])

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        feat_l.append(feat)

    processed_df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'label': label_list,
        'idx': idx_list
    })
    # Adding a column to help identify the original date
    processed_df['original_date'] = pd.to_datetime(processed_df['ts'], unit='s')
    return processed_df, np.array(feat_l), category_mapping
# Example usage:
# Assuming df is your DataFrame loaded in Google Colab with the appropriate columns
#preprocessed_df, feature_array = preprocess(df)


preprocessed_df, feature_array, category_mapping = preprocess(df)

def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique())), "u column not contiguous integers"
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique())), "i column not contiguous integers"

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

# Call the reindex function on the preprocessed DataFrame
#balanced_reindexed_df = reindex(preprocessed_df)

# Display the reindexed DataFrame
#print(balanced_reindexed_df)


def run(df, bipartite=True):
    # Preprocess the DataFrame
    preprocessed_df, feat,category_mapping = preprocess(df)
    # Reindex the DataFrame
    new_df = reindex(preprocessed_df, bipartite)

    # Create empty feature for padding
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    # Create random features for nodes
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    # Instead of saving to files, return the processed data
    return new_df, feat, rand_feat

# Example usage with your DataFrame `df`
#balanced_reindexed_df, feature_array, node_features = run(balanced_df)
new_df, feat, rand_feat = run(df, bipartite=True)
#
## Now balanced_reindexed_df, feature_array, and node_features are in memory
#print(balanced_reindexed_df.head())
#print(feature_array.shape)
#print(node_features.shape)

# FIX (Reviewer 6, Comment 3, code detail): edge indices are 1-based so that
# index 0 — which TGN's NeighborFinder returns for "no neighbor" — maps to a
# genuine zero padding row of the edge-feature matrix instead of colliding
# with the first real interaction's features.
new_df.idx = pd.RangeIndex(start=1, stop=len(new_df) + 1, step=1)


# =====================================================================
# feature_construction.py — REVISED (Reviewer 6, Comment 3)
#
# The previous pipeline leaked the prediction target into the inputs:
#   * a bijective 4-dim nn.Embedding of the Category LABEL was concatenated
#     into the EDGE features;
#   * NODE features were built per interaction ROW (port/proto/CATEGORY
#     embeddings) but consumed per NODE ID, so node k received the
#     attributes — including the label — of the k-th interaction of the
#     week (an arbitrary, possibly future/test event);
#   * all vocabularies/embedding tables were fit on the FULL week before
#     the 70/85-percentile temporal split.
#
# This cell rebuilds both feature sets without the target:
#   EDGE features  = [ standardized log FlowCount | Port emb | Proto emb ]
#                    (+ optional HISTORICAL-category channel, see flag)
#   NODE features  = [ role (+1 attacker / -1 victim) | node's most frequent
#                      Port emb | most frequent Proto emb ]  — aggregated
#                    over TRAIN-window interactions only; label-free.
#
# Train-only fitting: vocabularies, FlowCount statistics, and per-node
# profiles use ONLY interactions with ts <= 70th percentile (the training
# cutoff used in get_data). Unseen ports/protocols at val/test map to a
# reserved UNK index (0). Categorical embeddings are FROZEN random
# projections (fixed seed), as in the original pipeline, but deterministic.
#
# ABLATION (requested by the reviewer): INCLUDE_CATEGORY_IN_EDGE_FEATURES
#   False (default) -> the label appears nowhere in any input feature.
#   True            -> a category channel is appended to EDGE features.
#     With the single-forward-pass fix (Comment 1e) and TGN's strictly-past
#     neighbor sampling, an edge's own features never enter its own
#     prediction, so this variant exposes only the categories of EARLIER
#     edges ("past labels as features"). If used, this protocol must be
#     disclosed explicitly in the paper; the main results should use False.
#
# Row 0 of both matrices is a zero padding row (node id 0 / edge idx 0).
# Node feature dimension is 9 and MUST equal memory_dim (TGN adds memory
# to node features inside the embedding module).
# =====================================================================

import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict

INCLUDE_CATEGORY_IN_EDGE_FEATURES = ARGS.include_category_features

torch.manual_seed(0)
np.random.seed(0)

# --- training cutoff: same 70th-percentile rule as get_data ---
VAL_TIME = float(np.quantile(new_df.ts.values, 0.70))
train_row_mask = new_df.ts.values <= VAL_TIME
print(f'Feature construction: {train_row_mask.sum()} / {len(new_df)} rows in the training window')

# --- vocabularies fit on the TRAIN window only; 0 is reserved for UNK/padding ---
def build_vocab(values, mask):
    seen = []
    seen_set = set()
    for v in values[mask]:
        if v not in seen_set:
            seen_set.add(v); seen.append(v)
    return {v: i + 1 for i, v in enumerate(seen)}

port_vocab = build_vocab(df['Port'].values, train_row_mask)
proto_vocab = build_vocab(df['Proto'].values, train_row_mask)

port_idx = np.array([port_vocab.get(v, 0) for v in df['Port'].values])
proto_idx = np.array([proto_vocab.get(v, 0) for v in df['Proto'].values])
n_unseen = int((port_idx == 0).sum() + (proto_idx == 0).sum())
print(f'Vocabulary sizes (train-fitted): port={len(port_vocab)}, proto={len(proto_vocab)}; '
      f'{n_unseen} val/test values mapped to UNK')

# --- frozen random embedding tables (deterministic; padding_idx 0 stays zero) ---
EMB_DIM = 4
port_table = nn.Embedding(len(port_vocab) + 1, EMB_DIM, padding_idx=0)
proto_table = nn.Embedding(len(proto_vocab) + 1, EMB_DIM, padding_idx=0)
with torch.no_grad():
    port_emb_all = port_table(torch.tensor(port_idx, dtype=torch.long)).numpy()
    proto_emb_all = proto_table(torch.tensor(proto_idx, dtype=torch.long)).numpy()

# --- FlowCount: log-scaled, standardized with TRAIN statistics only ---
flow = np.log1p(df['FlowCount'].astype(float).values)
flow_mu = flow[train_row_mask].mean()
flow_sd = flow[train_row_mask].std() + 1e-9
flow_z = ((flow - flow_mu) / flow_sd)[:, None]

# --- EDGE features (aligned with 1-based edge idx via zero padding row 0) ---
edge_feat_rows = np.concatenate([flow_z, port_emb_all, proto_emb_all], axis=1)

if INCLUDE_CATEGORY_IN_EDGE_FEATURES:
    # HISTORICAL-category ablation channel (see header comment).
    n_categories = int(new_df.label.max()) + 1
    cat_table = nn.Embedding(n_categories + 1, EMB_DIM, padding_idx=0)
    with torch.no_grad():
        cat_emb_all = cat_table(torch.tensor(new_df.label.values + 1, dtype=torch.long)).numpy()
    edge_feat_rows = np.concatenate([edge_feat_rows, cat_emb_all], axis=1)
    print('WARNING: category channel INCLUDED in edge features (disclosed ablation variant).')

edge_features = np.vstack([np.zeros((1, edge_feat_rows.shape[1])), edge_feat_rows]).astype(np.float32)

# --- NODE features: per NODE (not per row), TRAIN window only, label-free ---
u_arr = new_df.u.values
i_arr = new_df.i.values
max_node_id = int(max(u_arr.max(), i_arr.max()))

attacker_ids = set(u_arr.tolist())          # bipartite construction: sources are attackers
victim_ids = set(i_arr.tolist())

node_port_counts = defaultdict(Counter)
node_proto_counts = defaultdict(Counter)
for r in np.where(train_row_mask)[0]:
    for node in (u_arr[r], i_arr[r]):
        node_port_counts[node][port_idx[r]] += 1
        node_proto_counts[node][proto_idx[r]] += 1

NODE_FEATURE_DIM = 1 + 2 * EMB_DIM          # role + port profile + proto profile = 9
node_features = np.zeros((max_node_id + 1, NODE_FEATURE_DIM), dtype=np.float32)
with torch.no_grad():
    for node in range(1, max_node_id + 1):
        role = 1.0 if node in attacker_ids else (-1.0 if node in victim_ids else 0.0)
        p_mode = node_port_counts[node].most_common(1)[0][0] if node in node_port_counts else 0
        pr_mode = node_proto_counts[node].most_common(1)[0][0] if node in node_proto_counts else 0
        node_features[node] = np.concatenate([
            [role],
            port_table(torch.tensor(p_mode)).numpy(),
            proto_table(torch.tensor(pr_mode)).numpy(),
        ])

n_cold = int((np.abs(node_features[1:, 1:]).sum(axis=1) == 0).sum())
print(f'edge_features: {edge_features.shape}, node_features: {node_features.shape} '
      f'({n_cold} nodes without training-window history -> zero profiles, role only)')
assert NODE_FEATURE_DIM == 9, "node feature dim must equal memory_dim (=9) — TGN adds memory to node features"


# VERIFICATION (Reviewer 6, Comment 3) — the per-row node-feature construction
# that previously lived in this cell assigned each NODE the attributes (including
# the category-label embedding) of an arbitrary interaction ROW; it has been
# replaced by the leakage-free, per-node construction in the cell above.
assert edge_features.shape[0] == len(new_df) + 1, "edge features: one row per interaction + padding row 0"
assert node_features.shape[0] == int(max(new_df.u.max(), new_df.i.max())) + 1, "node features: one row per node id"
assert node_features.shape[1] == 9, "node feature dim must equal memory_dim"
print('Feature matrices verified: node features are per-node, label-free; '
      'edge features contain the category channel only if the ablation flag is set.')



# ----------------------------------------------------------------------
# Save artifacts for run_train.py
# ----------------------------------------------------------------------
os.makedirs(ARGS.out_dir, exist_ok=True)
new_df.to_csv(os.path.join(ARGS.out_dir, 'new_df.csv'), index=False)
np.save(os.path.join(ARGS.out_dir, 'edge_features.npy'), edge_features)
np.save(os.path.join(ARGS.out_dir, 'node_features.npy'), node_features)
json.dump({str(k): v for k, v in category_mapping.items()},
          open(os.path.join(ARGS.out_dir, 'category_mapping.json'), 'w'), indent=2)
json.dump({'balance': ARGS.balance,
           'include_category_features': ARGS.include_category_features,
           'n_interactions': int(len(new_df)),
           'n_nodes': int(max(new_df.u.max(), new_df.i.max()))},
          open(os.path.join(ARGS.out_dir, 'preprocess_config.json'), 'w'), indent=2)
print('Saved processed artifacts to', ARGS.out_dir)

if ARGS.emit_table2:
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # =====================================================================
        # make_table2.py — NEW (Reviewer 6, Comment 4)
        #
        # Table 2 ("A sample of typical alert data") must contain VERBATIM records
        # from the released dataset. The reviewer verified against the released
        # files that the current table matches the data in none of: timestamp
        # offset (+02:00 never occurs; CET in March 2019 was UTC+01:00 — DST began
        # March 31), category taxonomy (Recon.Scanning / Anomaly.Traffic /
        # Availability.DoS / Availability.DDoS), protocol casing, or port
        # conventions (predominantly 'OtherPort'). This cell prints LaTeX rows
        # copied exactly as stored — paste them into Table 2 unmodified.
        # =====================================================================

        def latex_escape(s):
            return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')

        # first chronological genuine record of each category, strings untouched
        _parsed = pd.to_datetime(raw_records_df['DetectTime'], format='ISO8601', utc=True)
        _order = _parsed.sort_values().index
        _seen, _rows = set(), []
        for _idx in _order:
            _cat = raw_records_df.loc[_idx, 'Category']
            if _cat not in _seen:
                _seen.add(_cat)
                _rows.append(raw_records_df.loc[_idx])

        _cols = ['DetectTime', 'FlowCount', 'SourceIP', 'TargetIP', 'Port', 'Proto', 'Category']
        print('% Table 2 replacement — verbatim records from the released dataset')
        print(r'\begin{tabular}{lllllll}')
        print(r'\toprule')
        print(' & '.join([r'\textbf{' + c + '}' for c in _cols]) + r' \\')
        print(r'\midrule')
        for _r in _rows:
            print(' & '.join(latex_escape(_r[c]) for c in _cols) + r' \\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
        print('\n% NOTE: verify Sec. 4.4 prose against these rows (offsets, taxonomy,')
        print('%       protocol casing, ports) so text and table stay consistent.')

    open(os.path.join(ARGS.out_dir, 'table2.tex'), 'w').write(buf.getvalue())
    print('Wrote verbatim Table 2 rows to', os.path.join(ARGS.out_dir, 'table2.tex'))
