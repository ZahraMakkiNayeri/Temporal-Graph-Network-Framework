# BiTA: A BiGRU–Transformer Message Aggregator for Temporal Graph Networks

Code and reproduction artifact for *"BiTA: Bidirectional GRU–Transformer
Aggregation in a Temporal Graph Network Framework for Alert Prediction in
Computer Networks"* (under review, Applied Soft Computing).

This repository replaces the message aggregation step of the Temporal Graph
Network (TGN) framework with a learnable sequence encoder, and evaluates whether
doing so improves proactive alert prediction. It contains everything needed to
reproduce every number in the paper, together with a mapping from each table and
figure to the command that produces it.

---

## What this repository provides

- A single entry point (`run_train.py`) with resumable multi-seed training
- Ten message aggregators behind one flag, including the non-parametric
  baselines, so that comparisons differ only in the aggregation function
- Preprocessing for both datasets, with leakage-free feature construction and
  all encoders fit on the training window alone
- Evaluation under four negative-sampling regimes, genuine link-ranking metrics,
  and a disclosed threshold-tuning protocol
- Statistical analysis: paired *t*-tests, Wilcoxon signed-rank tests, Cohen's
  *d<sub>z</sub>*, confidence intervals, and Holm–Bonferroni correction
- Verification scripts for temporal causality and feature-level causality
- `RELEASE.md`, mapping every table and figure in the paper to its command

---

## Installation

```bash
git clone <repository-url> && cd bita-tgn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A single GPU is sufficient. All reported results were produced on one NVIDIA
A40; peak memory stays below 16 GB.

---

## Data

Neither dataset is redistributed here.

**Warden.** Download the seven daily CSV files (`11March_e.csv` …
`17March_e.csv`) from the dataset record and place them in `data/`. Columns:
`DetectTime, FlowCount, SourceIP, TargetIP, Category, Proto, Port`.

**NF-UNSW-NB15-v2.** Download `NF-UNSW-NB15-v2.csv` and place it in `data/`.

```bash
# Warden, with and without the historical-category channel
python -m preprocessing.preprocess_data --raw-dir data \
       --out-dir data/processed_cat --include-category-features --emit-table2
python -m preprocessing.preprocess_data --raw-dir data --out-dir data/processed

# NF-UNSW-NB15-v2, attack-enriched and natural class proportions
python -m preprocessing.preprocess_unsw --raw data/NF-UNSW-NB15-v2.csv \
       --out-dir data/unsw_cat --n-interactions 150000 --keep-all-attacks \
       --include-category-features
python -m preprocessing.preprocess_unsw --raw data/NF-UNSW-NB15-v2.csv \
       --out-dir data/unsw_nat_cat --n-interactions 150000 \
       --include-category-features
```

Each output directory contains `new_df.csv`, `edge_features.npy`,
`node_features.npy` and `preprocess_config.json`. The edge-feature width
identifies the configuration: 13 columns with the category channel and 9
without on Warden, 17 and 13 on NF-UNSW-NB15-v2.

---

## Running

```bash
# proposed configuration
python run_train.py --processed-dir data/processed_cat \
       --n-runs 25 --link-ranking-candidates 100

# any other aggregator: last, mean, gru, bigru, bitransformer,
# gru_transformer, bitransformer_temporal, relative_transformer,
# stacked_bitransformer, tcn
python run_train.py --processed-dir data/processed_cat --aggregator bigru \
       --n-runs 25 --link-ranking-candidates 100

# NF-UNSW-NB15-v2 (10 categories; 38 distinct destinations)
python run_train.py --processed-dir data/unsw_cat --num-categories 10 \
       --memory-dim 9 --n-runs 10 --link-ranking-candidates 37
```

Results are written incrementally to `test_metrics_multi_seed.csv`, one row per
seed, so an interrupted run resumes from the first unfinished seed. Confusion
matrices, ROC and PR curve data, and sequence-length bins are written to
`artifacts/`.

`reproduce.sh` runs the full pipeline; `bash reproduce.sh --dry-run` prints the
commands without executing them.

---

## Baselines

```bash
python -m baselines.edgebank   --processed-dir data/processed_cat \
       --link-ranking-candidates 100 --out-dir runs/edgebank
python -m baselines.classical  --processed-dir data/processed_cat \
       --out-dir runs/classical            # RF, linear SVM, logistic regression
python -m baselines.static_gnn --processed-dir data/processed_cat \
       --model gcn --n-runs 5 --out-dir runs/gcn        # also: graphsage
```

DyRep and JODIE are configurations of the same backbone rather than separate
implementations:

```bash
python run_train.py --processed-dir data/processed_cat --dyrep true \
       --memory-updater rnn --use-source-embedding-in-message true
python run_train.py --processed-dir data/processed_cat \
       --embedding-module time --memory-updater rnn --aggregator last
```

---

## Analysis

```bash
# comparison tables and figures
python scripts/collect_results.py --out-dir comparison --run "BiTA=." --run "TGN-Last=runs/tgn_last"
python scripts/make_comparison_table.py --proposed "BiTA=." --baseline "TGN-Last=runs/tgn_last" --out comparison/table_main
python scripts/make_table4.py --processed-dir data/processed_cat --run "BiTA=."

# statistics
python scripts/paired_tests.py --a "TGN-Last=runs/tgn_last" --b "BiTA=." --all-metrics
python scripts/significance_table.py --in "TGN-Last=comparison/vs_tgn_last.csv"
python scripts/seed_sensitivity.py --pair "BiTA vs BiGRU=.:runs/agg_bigru"

# verification
python scripts/causality_experiment.py --processed-dir data/processed_cat --n-probe-edges 200
python scripts/self_leak_check.py      --processed-dir data/processed_cat
python leakage_probe.py data/11March_e.csv

# cost and attribution
python scripts/scalability_benchmark.py --processed-dir data/processed_cat --model saved_models
python scripts/interpretability.py --processed-dir data/processed_cat --model saved_models
python scripts/dataset_stats.py --data "Warden=data/processed_cat" --data "NF-UNSW=data/unsw_cat"
```

`RELEASE.md` maps each of these to the table or figure it produces.

---

## Protocol

Fixed across every method and configuration:

| | |
|---|---|
| Splits | chronological, 70th / 85th timestamp percentiles |
| Inductive set | 10% of nodes first appearing after the validation cutoff, removed from training entirely |
| Negative sampling | 1:1, destination corrupted only, so attacker→victim direction is preserved |
| Test-time regimes | random, collision-checked random, historical, inductive |
| Link ranking | 100 candidates (37 on NF-UNSW-NB15-v2), shared pool, true destination excluded, ties counted 0.5 |
| Threshold | tuned for F1 on the validation split of the selected epoch, applied unchanged to test |
| Class imbalance | per-class loss weights from the training split only; no resampling at any stage |
| Seeds | run *i* sets both PyTorch and NumPy seeds to *i*, so runs with the same index are paired across methods |
| Early stopping | validation average precision, patience 5 |
| Optimiser | Adam, lr 1e-4, batch size 128, dropout 0.1 |

Two dimensions are architecturally constrained rather than tuned. The memory
dimension must equal the node-feature dimension, because memory is combined with
node features element-wise; both are 9. The number of attention heads must
therefore divide 18, admitting only 1, 2, 3, 6, 9 and 18.

---

## Provenance

Two files are written automatically at every run and record what actually
executed, independently of directory names:

- `preprocess_config.json` — whether the category channel is included, and
  whether any resampling was applied
- `parameter_breakdown.csv` — trainable parameters per module

The parameter total distinguishes the configurations: 1,240,264 with the
category channel and 1,237,714 without, the difference being the cost of four
additional edge-feature dimensions.

---

## Notes and limitations

- The aggregator operates on a bounded window of the most recent
  `--aggregator-max-seq-len` pending messages per node (default 128).
- Trainable parameters are **not** equal across compared aggregators: 8,352 for
  last and mean, 54,216 for BiGRU, 1,240,264 for the proposed aggregator.
  Per-module counts are reported rather than an equal-capacity claim.
- NF-UNSW-NB15-v2 provides no timestamp column; the temporal order is the row
  order of the released file. It also contains only 44 distinct IP addresses,
  so as an interaction graph over hosts it is small and dense.
- Throughput figures extrapolated to a daily rate are an idealised ceiling under
  the stated hardware and graph size, not a measured operational rate.

## Citation

```bibtex
@article{bita2026,
  title  = {BiTA: Bidirectional GRU--Transformer Aggregation in a Temporal
            Graph Network Framework for Alert Prediction in Computer Networks},
  author = {...},
  year   = {2026},
  note   = {Under review}
}
```




