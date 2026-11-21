# RELEASE: tables and figures → commands, configurations and seed sets

This document maps every table and figure of the manuscript
**ASOC-D-26-06900, second revision**, to the command that produces it, the
configuration it uses and the seeds it averages over. Table and figure numbers
refer to that version of the manuscript.

The step labels inside `reproduce.sh` and some script docstrings use the
numbering of the first revision; where they disagree with this file, this file
is authoritative.

---

## 1. Common setup

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

All temporal runs use `run_train.py` with `--link-ranking-candidates 100`
(37 on NF-UNSW-NB15-v2) and the fixed protocol listed in `README.md`.

## 2. Seed sets

Run *i* sets both the PyTorch and NumPy seeds to *i*, so runs with the same
index are paired across methods.

| Seed set | Seeds | Used for |
|---|---|---|
| **25-seed** | 0–24 | principal comparisons: Table 9 (BiTA row), Section 5.10 text, Section 5.18 |
| **10-seed** | 0–9 | secondary analyses: Tables 5–8, 10, 11 (all but two rows), 14–17 |
| **3-seed** | 0–2 | hyperparameter scan, Table 23 |
| **5-seed** | 0–4 | static graph baselines (GCN, GraphSAGE) in Table 9 |
| **single run** | — | EdgeBank (deterministic); interpretability (one checkpoint) |

A 10-seed result for a configuration that also has 25 seeds uses the first ten
seeds of the same run directory. Every table states its seed count.

## 3. Run directories

| Run directory | Configuration | Seeds | Command |
|---|---|---|---|
| `runs/cat_full` | BiTA, with category channel | 25 | `python run_train.py --processed-dir data/processed_cat --n-runs 25` |
| `runs/warden-nocat` | BiTA, without category channel | 10 | `python run_train.py --processed-dir data/processed --n-runs 10` |
| `runs/cat_tgn_last` | last-message aggregation | 25 | `... --processed-dir data/processed_cat --aggregator last --n-runs 25` |
| `runs/cat_tgn_mean` | mean aggregation | 10 | `... --aggregator mean --n-runs 10` |
| `runs/agg_bitransformer` | Transformer only (TGN-Attention) | 25 | `... --aggregator bitransformer --n-runs 25` |
| `runs/agg_bigru` | BiGRU only | 25 | `... --aggregator bigru --n-runs 25` |
| `runs/agg_gru` | GRU only | 10 | `... --aggregator gru --n-runs 10` |
| `runs/agg_gru_transformer` | GRU + Transformer | 10 | `... --aggregator gru_transformer --n-runs 10` |
| `runs/cat_dyrep` | DyRep-style | 10 | `... --dyrep true --memory-updater rnn --use-source-embedding-in-message true --n-runs 10` |
| `runs/cat_jodie` | JODIE-style | 10 | `... --embedding-module time --memory-updater rnn --aggregator last --n-runs 10` |
| `runs/edgebank` | EdgeBank | 1 | `python -m baselines.edgebank --processed-dir data/processed_cat --out-dir runs/edgebank` |
| `runs/classical` | Random Forest, linear SVM, logistic regression | 10 | `python -m baselines.classical --processed-dir data/processed_cat --seed $i` for `i` in 0–9 |
| `runs/gcn`, `runs/graphsage` | static GNNs | 5 | `python -m baselines.static_gnn --processed-dir data/processed_cat --model gcn --n-runs 5` (and `--model graphsage`) |
| `runs/sens_*` | Table 23 variants | 3 | `run_train.py` with one setting changed per variant (see Table 23) |

Each run writes one row per seed to `<run>/test_metrics_multi_seed.csv`, and
confusion matrices, ROC/PR curve data and sequence-length bins to
`<run>/artifacts/`.

## 4. Tables

| Table | Content | Produced by | Output | Seeds |
|---|---|---|---|---|
| 1 | Related-work comparison | — (descriptive) | — | — |
| 2 | Notation | — (descriptive) | — | — |
| 3 | Sample of alert data | rows of the released Warden file `data/11March_e.csv` | — | — |
| 4 | Example interaction stream | — (illustrative) | — | — |
| 5 | BiTA vs random ranker | `scripts/collect_results.py`; random ranker computed analytically for 100 candidates | `comparison/comparison.csv` (row `BiTA`, columns `test_link_*`) | 10 |
| 6 | Paired comparison against four alternatives | `scripts/paired_tests.py --a <baseline> --b BiTA=runs/cat_full` for each baseline | `comparison/cat10_vs_cat_tgn_last.csv`, `cat10_vs_cat_tgn_mean.csv`, `cat10_vs_agg_bitransformer.csv`, `cat10_vs_cat_dyrep.csv` | 10 |
| 7 | Historical-category channel | Warden: `scripts/paired_tests.py` (with vs without channel); NF-UNSW-NB15-v2: `scripts/collect_results.py` on the UNSW runs | `comparison/ablation_category.csv`; `comparison_unsw/comparison.csv`| 10 |
| 8 | With- vs no-category, paired | `scripts/paired_tests.py --a "no-category=runs/warden-nocat" --b "with-category=runs/cat_full"` | `comparison/ablation_category.csv` | 10 |
| 9 | Non-temporal baselines | `baselines/classical.py`, `baselines/static_gnn.py`; BiTA row from `runs/cat_full` | per-run `test_metrics_multi_seed.csv` | as in the table's Seeds column |
| 10 | Aggregation strategies | `scripts/collect_results.py` over the 10-seed runs of Section 3 and EdgeBank | regenerate with the command in Section 6 | 10 (EdgeBank 1) |
| 11 | Per-class metrics | `scripts/make_table4.py --processed-dir data/processed_cat` | `comparison/table4.csv`, `table4.tex` | 25 for BiTA and Transformer only, 10 otherwise |
| 12 | Aggregator properties | — (descriptive) | — | — |
| 13 | Learnable aggregators, transductive AUC/AP | original-submission pipeline (Section 7 below) | — | not averaged over seeds |
| 14 | Ablation of architectural components | `scripts/collect_results.py` over `agg_gru`, `agg_bigru`, `agg_bitransformer`, `agg_gru_transformer`, `cat_full` | regenerate with the command in Section 6 | 10 |
| 15 | Per-category precision and recall | `scripts/make_table15.py` over the same five runs | printed LaTeX rows | 10 |
| 16 | Inductive performance | as Table 14, columns `test_*_inductive` | as Table 14 | 10 |
| 17 | New-node performance | as Table 14, columns `nn_test_*` | as Table 14 | 10 |
| 18 | Temporal causality and batch-grouping invariance | `scripts/causality_experiment.py --processed-dir data/processed_cat --n-probe-edges 200` | `causality_stats.json` | one checkpoint, 200 probed edges |
| 19 | Feature-level causality | `scripts/self_leak_check.py --processed-dir data/processed_cat` | printed | one checkpoint |
| 20 | Training configuration | defaults of `run_train.py` | — | — |
| 21 | Dataset statistics | `scripts/dataset_stats.py --data "Warden=data/processed_cat" --data "NF-UNSW natural=data/unsw_nat_cat" --data "NF-UNSW enriched=data/unsw_cat"`; the row *Removed from training* is the total minus the train, validation and test counts | `comparison/dataset_stats.csv`, `.tex` | — |
| 22 | Computational cost | `scripts/scalability_benchmark.py` (inference, batch size 500); training time from the run logs | printed | — |
| 23 | Hyperparameter sensitivity | `scripts/collect_results.py --out-dir comparison_sensitivity` over the `sens_*` runs | `comparison_sensitivity/comparison.csv` | 3 |
| 24 | Inference cost vs batch size | `scripts/scalability_benchmark.py --batch-sizes 32,64,128,256,500` | printed | — |
| 25 | Per-edge latency vs graph size | `scripts/scalability_benchmark.py --graph-sizes 2000,3000,5000,6500` (batch size 128) | printed | — |
| 26 | Per-module parameter counts | written by `run_train.py` at the first seed | `<run>/parameter_breakdown.csv` | — |
| 27 | Feature-block permutation importance | `scripts/interpretability.py --processed-dir data/processed_cat` | printed | one checkpoint |
| 28 | Message-level occlusion | `scripts/interpretability.py` (same run) | printed | one checkpoint, 300 edges |

**Note on Table 26.** Summing `p.numel()` over `model.parameters()` gives
1,240,264 for BiTA. This includes a copy of the Transformer layer (593,024
parameters) that `nn.TransformerEncoder` keeps but never uses in the forward
pass. Table 26 reports the 647,240 parameters that take part in computing a
prediction.

## 5. Figures

| Figure | Content | Produced by | Seeds |
|---|---|---|---|
| 1–4 | Architecture and task illustrations | — (drawn) | — |
| 5, 6 | Attack intervals and accumulation | computed from the raw Warden files | — |
| 7 | Effect size and significance vs number of seeds | `scripts/seed_sensitivity.py` → `comparison/seed_sensitivity.pdf` | 25 |
| 8, 9 | ROC and precision–recall curves | original-submission pipeline (Section 7 below) | not averaged over seeds |
| 10 | Inductive confusion matrix | averaged from `<run>/artifacts/confusion_matrix_nntest_run*.csv` of `runs/cat_full`| as stated in the caption |
| 11 | Time-window comparison | original-submission pipeline (Section 7 below) | single run per window |
| 12, 13, 14 | Baseline comparisons, comparison over time | original-submission pipeline (Section 7 below) | not averaged over seeds |
| 15 | NF-UNSW-NB15-v2 aggregators (supplementary) | — | — |
| 16 | Performance by sequence length | original-submission pipeline (Section 7 below) | not averaged over seeds |
| 17 | Temporal causality and batch-grouping invariance | `scripts/causality_experiment.py` → `causality.png`, `order_invariance.png` | one checkpoint |

## 6. Numbers quoted in the text

| Where | Quantity | Source |
|---|---|---|
| Section 5.10, Section 5.21 | BiTA vs last-message aggregation, +0.0007 AUC, 95% CI [−0.0125, +0.0138] | `comparison/table_main_detail.csv` (`scripts/make_comparison_table.py`) |
| Section 5.18 | BiTA vs BiGRU-only and vs Transformer-only, paired over 25 seeds | `comparison/transformer_contribution_25.csv`, `comparison/recurrence_contribution_25.csv` |
| Section 5.18 | link MRR and category accuracy vs BiGRU-only | `comparison/rank_vs_agg_bigru.csv` |
| Section 5.6 | effect-size collapse and power | `comparison/seed_sensitivity.csv` |
| Section 5.17 | test interactions by number of pending messages (2,319 / 2,849 / 1,741 / 1,226 / 115 / 564; 63% with two or more) | `python pending_lengths.py --processed-dir data/processed_cat` (data only, no model) |
| Table 9 caption, Section 7 | standard deviation of BiTA over 25 seeds (0.0265 AUC, 0.0224 AP, 0.0304 AUC under historical negatives) | `comparison/table_main.csv` |
| Section 6 | precision 0.8722, recall 0.8786 and false-positive rate 0.1287 over 25 seeds, and the implied precision at other base rates | `comparison/cat_full_vs_cat_tgn_last.csv` (BiTA columns `test_precision`, `test_recall`); false-positive rate = recall × (1 − precision) / precision |

To regenerate Tables 10, 14, 16 and 17:

```bash
python scripts/collect_results.py --out-dir comparison_tables \
  --run "BiTA=runs/cat_full" --run "TGN-Last=runs/cat_tgn_last" \
  --run "TGN-Mean=runs/cat_tgn_mean" --run "TGN-Attention=runs/agg_bitransformer" \
  --run "DyRep-style=runs/cat_dyrep" --run "JODIE-style=runs/cat_jodie" \
  --run "GRU=runs/agg_gru" --run "BiGRU=runs/agg_bigru" \
  --run "GRU+Transformer=runs/agg_gru_transformer" --run "EdgeBank=runs/edgebank"
```
