"""
leakage_probe.py — Falsification test for Reviewer 6, Comment 3 (BiTA / Warden).

QUESTION: is the attack category recoverable from the ORIGINAL static feature
tables alone — no TGN, no graph, no temporal modeling?

If the features carried no target information, a linear probe reading ONLY a
node's static feature row should score near the majority-class baseline on the
test split. If it scores near the paper's Table 4 numbers instead, the label
is in the inputs, and the temporal model's contribution cannot be assessed
from the current results.

This script mirrors the released pipeline exactly:
  1. (optional) median-balancing of classes, then chronological re-sort
     — as described in Sec. 5.4 of the manuscript;
  2. preprocess(): IP factorization, label encoding (as in the notebook);
  3. reindex(): bipartite ids — sources 1..n_src, victims n_src+1..n_src+n_dst;
  4. ORIGINAL feature construction, verbatim logic:
       edge_features = [FlowCount emb | Proto emb | CATEGORY emb]   (per row)
       node_features = [Port emb | Proto emb | CATEGORY emb]        (per ROW,
                        but consumed per NODE ID by TGN — the indexing bug);
  5. temporal split at the 70th/85th timestamp percentiles (as in get_data);
  6. linear probes (multinomial logistic regression), trained on train-split
     edges, evaluated on test-split edges:
       P0: edge_features[edge_idx]        -> category   (bijectivity check)
       P1: node_features[destination_id]  -> category   (the silent channel)
       P2: node_features[source_id]       -> category
       P3: P1 + P2 concatenated           -> category
       C : CONTROL — leakage-free node features (role + train-window port/proto
           profiles, no category anywhere) with the same probes.

USAGE:
    python leakage_probe.py /path/to/warden.csv
    (expects columns: DetectTime, FlowCount, SourceIP, TargetIP, Port, Proto, Category)

Set BALANCE=False below to skip the median-resampling step.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BALANCE = True          # mirror the paper's median-balancing (Sec. 5.4)
SEED = 0

# ----------------------------------------------------------------------
# 1. Load + (optional) balancing, exactly as described in the manuscript
# ----------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    df['DetectTime'] = pd.to_datetime(df['DetectTime'], format='ISO8601', utc=True)
    df = df.sort_values('DetectTime').reset_index(drop=True)
    if BALANCE:
        sizes = df['Category'].value_counts()
        median = int(sizes.median())
        rng = np.random.RandomState(SEED)
        parts = []
        for cat, group in df.groupby('Category'):
            if len(group) >= median:
                parts.append(group.sample(n=median, random_state=SEED))          # undersample
            else:
                parts.append(group.sample(n=median, replace=True, random_state=SEED))  # oversample
        df = pd.concat(parts).sort_values('DetectTime').reset_index(drop=True)
        print(f'[balance] aligned all classes to the median size ({median}); {len(df)} rows after re-sort')
    return df

# ----------------------------------------------------------------------
# 2-3. preprocess + bipartite reindex (as in the notebook)
# ----------------------------------------------------------------------
def preprocess_and_reindex(df):
    df = df.copy()
    df['SourceIP'] = df['SourceIP'].astype(str)
    df['TargetIP'] = df['TargetIP'].astype(str)
    df['u0'], _ = pd.factorize(df['SourceIP'])
    df['i0'], _ = pd.factorize(df['TargetIP'])
    df['label'], cat_index = pd.factorize(df['Category'])
    upper_u = df['u0'].max() + 1
    df['u'] = df['u0'] + 1                       # sources: 1..n_src
    df['i'] = df['i0'] + upper_u + 1             # victims: n_src+1..n_src+n_dst
    df['ts'] = df['DetectTime'].astype('int64') / 1e9
    df['edge_idx'] = np.arange(len(df))          # 0-based, as in the released notebook
    print(f'[graph] {df.u.nunique()} attackers, {df.i.nunique()} victims, {len(df)} interactions, '
          f'{len(cat_index)} categories: {list(cat_index)}')
    return df, list(cat_index)

# ----------------------------------------------------------------------
# 4. ORIGINAL feature construction (verbatim logic of the released cells)
# ----------------------------------------------------------------------
def original_features(df):
    torch.manual_seed(SEED)
    # --- edge features: FlowCount emb | Proto emb | Category emb (dim 4 each) ---
    fc_map = {v: k for k, v in enumerate(df['FlowCount'].unique())}
    pr_map = {v: k for k, v in enumerate(df['Proto'].unique())}
    ct_map = {v: k for k, v in enumerate(df['Category'].unique())}
    e_fc = nn.Embedding(len(fc_map), 4); e_pr = nn.Embedding(len(pr_map), 4); e_ct = nn.Embedding(len(ct_map), 4)
    with torch.no_grad():
        edge_features = torch.cat((
            e_fc(torch.tensor(df['FlowCount'].map(fc_map).values)),
            e_pr(torch.tensor(df['Proto'].map(pr_map).values)),
            e_ct(torch.tensor(df['Category'].map(ct_map).values)),
        ), dim=1).numpy()
    # --- node features: Port emb | Proto emb | Category emb (dim 3 each), PER ROW ---
    po_map = {v: k for k, v in enumerate(df['Port'].unique())}
    n_po = nn.Embedding(len(po_map), 3); n_pr = nn.Embedding(len(pr_map), 3); n_ct = nn.Embedding(len(ct_map), 3)
    with torch.no_grad():
        node_features_per_row = torch.cat((
            n_po(torch.tensor(df['Port'].map(po_map).values)),
            n_pr(torch.tensor(df['Proto'].map(pr_map).values)),
            n_ct(torch.tensor(df['Category'].map(ct_map).values)),
        ), dim=1).numpy()
    # TGN indexes this table by NODE ID — reproduce exactly that lookup.
    max_node_id = int(df['i'].max())
    if len(node_features_per_row) <= max_node_id:
        raise SystemExit('fewer rows than node ids — indexing would crash, as in some runs')
    node_feature_table = node_features_per_row      # row k <- k-th interaction of the week
    return edge_features, node_feature_table

# ----------------------------------------------------------------------
# CONTROL: leakage-free node features (as in the revised notebook)
# ----------------------------------------------------------------------
def clean_node_features(df, val_time):
    torch.manual_seed(SEED)
    train_mask = df['ts'].values <= val_time
    po_vocab = {v: k + 1 for k, v in enumerate(pd.unique(df['Port'].values[train_mask]))}
    pr_vocab = {v: k + 1 for k, v in enumerate(pd.unique(df['Proto'].values[train_mask]))}
    po_idx = np.array([po_vocab.get(v, 0) for v in df['Port'].values])
    pr_idx = np.array([pr_vocab.get(v, 0) for v in df['Proto'].values])
    t_po = nn.Embedding(len(po_vocab) + 1, 4, padding_idx=0)
    t_pr = nn.Embedding(len(pr_vocab) + 1, 4, padding_idx=0)
    u_arr, i_arr = df['u'].values, df['i'].values
    max_node_id = int(i_arr.max())
    cnt_po, cnt_pr = defaultdict(Counter), defaultdict(Counter)
    for r in np.where(train_mask)[0]:
        for node in (u_arr[r], i_arr[r]):
            cnt_po[node][po_idx[r]] += 1
            cnt_pr[node][pr_idx[r]] += 1
    attackers, victims = set(u_arr), set(i_arr)
    table = np.zeros((max_node_id + 1, 9), np.float32)
    with torch.no_grad():
        for node in range(1, max_node_id + 1):
            role = 1.0 if node in attackers else (-1.0 if node in victims else 0.0)
            pm = cnt_po[node].most_common(1)[0][0] if node in cnt_po else 0
            rm = cnt_pr[node].most_common(1)[0][0] if node in cnt_pr else 0
            table[node] = np.concatenate([[role], t_po(torch.tensor(pm)).numpy(), t_pr(torch.tensor(rm)).numpy()])
    return table

# ----------------------------------------------------------------------
# 6. Probes
# ----------------------------------------------------------------------
def probe(name, X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro', zero_division=0)
    print(f'  {name:52s} acc = {acc:.4f}   macro-F1 = {f1:.4f}')
    return acc

def main(path):
    df = load(path)
    df, categories = preprocess_and_reindex(df)
    val_time, test_time = np.quantile(df['ts'].values, [0.70, 0.85])
    train = df[df['ts'] <= val_time]
    test = df[df['ts'] > test_time]
    y_tr, y_te = train['label'].values, test['label'].values
    print(f'[split] train {len(train)} / test {len(test)} edges '
          f'(70th/85th percentile cutoffs, as in get_data)')

    maj = Counter(y_tr).most_common(1)[0][0]
    base = accuracy_score(y_te, np.full_like(y_te, maj))
    print(f'\n=== BASELINE (no features at all) ===')
    print(f'  {"majority-class":52s} acc = {base:.4f}')

    ef, nf = original_features(df)
    print(f'\n=== ORIGINAL features (released pipeline) — probes see ONE static vector, no graph, no time ===')
    probe('P0: edge_features[edge_idx] -> category', ef[train.edge_idx.values], y_tr, ef[test.edge_idx.values], y_te)
    probe('P1: node_features[destination_id] -> category', nf[train.i.values], y_tr, nf[test.i.values], y_te)
    probe('P2: node_features[source_id] -> category', nf[train.u.values], y_tr, nf[test.u.values], y_te)
    probe('P3: node_features[src] + node_features[dst] -> category',
          np.hstack([nf[train.u.values], nf[train.i.values]]), y_tr,
          np.hstack([nf[test.u.values], nf[test.i.values]]), y_te)

    cnf = clean_node_features(df, val_time)
    print(f'\n=== CONTROL: leakage-free node features (revised pipeline) ===')
    probe('C1: clean node_features[destination_id] -> category', cnf[train.i.values], y_tr, cnf[test.i.values], y_te)
    probe('C3: clean node feats [src]+[dst] -> category',
          np.hstack([cnf[train.u.values], cnf[train.i.values]]), y_tr,
          np.hstack([cnf[test.u.values], cnf[test.i.values]]), y_te)

    print(f"""
=== HOW TO READ THIS ===
* P0 near 1.0            -> the category embedding in edge features is bijective:
                            the label is literally stored in the input vector.
* P1/P3 >> majority acc  -> a linear model with NO graph and NO temporal modeling
                            approaches Table 4 from static node features alone;
                            the per-row node-feature table carries the target.
* C1/C3 near majority    -> with the label removed, static features alone cannot
                            predict the category — any performance above this in
                            the full model is then attributable to BiTA/TGN.
If instead P1/P3 land near the majority baseline, the leakage concern for the
node channel is empirically refuted on this dataset — report exactly that.
""")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python leakage_probe.py /path/to/warden.csv')
    main(sys.argv[1])
