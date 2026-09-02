#!/bin/bash
# Reproduce every reported result. See RELEASE.md for the table/figure mapping.
#
#   bash reproduce.sh            run everything (long: multi-seed training)
#   bash reproduce.sh --dry-run  print the commands without running them
#   SEEDS=3 bash reproduce.sh    fewer seeds for a quick check
#
# Steps 0-2 must run in order. Steps 3-6 are independent of each other.
set -euo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
SEEDS=${SEEDS:-10}
Q=${Q:-100}
RUNS=${RUNS:-runs}
RAW=${RAW:-data}
MAIN=data/processed_cat        # main configuration: historical category channel
ABL=data/processed             # ablation: category absent from all inputs

run () { echo; echo "\$ $*"; [ $DRY -eq 1 ] || eval "$@"; }
step () { echo; echo "=============== $* ==============="; }

# ---------------------------------------------------------------- 0. data
step "0. preprocessing (Table 2, dataset statistics)"
run "python -m preprocessing.preprocess_data --raw-dir $RAW --out-dir $MAIN \
     --include-category-features --emit-table2"
run "python -m preprocessing.preprocess_data --raw-dir $RAW --out-dir $ABL"

# ------------------------------------------------- 1. main + category ablation
step "1. main results and category ablation (Table 4, Reviewer 6 Comment 3)"
for cfg in "warden-cat:$MAIN" "warden-nocat:$ABL"; do
  name=${cfg%%:*}; dir=${cfg#*:}
  run "mkdir -p $RUNS/$name && cp -r model modules utils preprocessing baselines scripts \
       run_train.py $RUNS/$name/ 2>/dev/null || true"
  run "(cd $RUNS/$name && ln -sfn ../../data data && \
       python run_train.py --processed-dir $dir --n-runs $SEEDS \
       --link-ranking-candidates $Q --prefix $name)"
done

# ------------------------------------------------------- 2. aggregator ablation
step "2. aggregator ablation (Tables 6-8, Experiment 10)"
for A in last mean gru bigru bitransformer gru_transformer \
         bitransformer_temporal relative_transformer stacked_bitransformer tcn; do
  run "mkdir -p $RUNS/agg_$A && cp -r model modules utils preprocessing baselines scripts \
       run_train.py $RUNS/agg_$A/ 2>/dev/null || true"
  run "(cd $RUNS/agg_$A && ln -sfn ../../data data && \
       python run_train.py --processed-dir $MAIN --aggregator $A \
       --n-runs $SEEDS --link-ranking-candidates $Q --prefix $A)"
done

# ------------------------------------------------------------- 3. DyRep variant
step "3. DyRep (memory-based baseline within the same backbone)"
run "mkdir -p $RUNS/dyrep && cp -r model modules utils preprocessing baselines scripts \
     run_train.py $RUNS/dyrep/ 2>/dev/null || true"
run "(cd $RUNS/dyrep && ln -sfn ../../data data && \
     python run_train.py --processed-dir $MAIN --dyrep true --memory-updater rnn \
     --use-source-embedding-in-message true --n-runs $SEEDS \
     --link-ranking-candidates $Q --prefix dyrep)"

# ----------------------------------------------------------------- 4. EdgeBank
step "4. EdgeBank (parameter-free memorisation baseline; one run is exact)"
run "python -m baselines.edgebank --processed-dir $MAIN \
     --link-ranking-candidates $Q --out-dir $RUNS/edgebank"

# ------------------------------------------------- 5. analyses needing no training
step "5. leakage, causality, scalability (Reviewer 6 C3, Experiment 11, Section 5.19)"
run "python leakage_probe.py $RAW/11March_e.csv"
run "python scripts/self_leak_check.py --processed-dir $MAIN"
run "python scripts/causality_experiment.py --processed-dir $MAIN --n-probe-edges 200"
run "python scripts/scalability_benchmark.py --processed-dir $MAIN"

# ------------------------------------------------------- 6. tables and figures
step "6. comparison table, Figure 26 replacement, significance tests"
run "python scripts/collect_results.py --out-dir comparison \
     --run \"BiTA=$RUNS/warden-cat\" \
     --run \"BiTA (no category)=$RUNS/warden-nocat\" \
     --run \"TGN-Last=$RUNS/agg_last\" --run \"TGN-Mean=$RUNS/agg_mean\" \
     --run \"TGN-Attention=$RUNS/agg_bitransformer\" \
     --run \"BiGRU only=$RUNS/agg_bigru\" --run \"GRU only=$RUNS/agg_gru\" \
     --run \"DyRep=$RUNS/dyrep\" --run \"EdgeBank=$RUNS/edgebank\""

run "python scripts/paired_tests.py --a \"no-category=$RUNS/warden-nocat\" \
     --b \"with-category=$RUNS/warden-cat\" --out comparison/ablation_category.csv"
run "python scripts/paired_tests.py --a \"BiGRU=$RUNS/agg_bigru\" \
     --b \"BiTA=$RUNS/warden-cat\" --out comparison/transformer_contribution.csv"
run "python scripts/paired_tests.py --a \"TGN-Last=$RUNS/agg_last\" \
     --b \"BiTA=$RUNS/warden-cat\" --out comparison/vs_tgn_last.csv"

echo
echo "Done. Tables in comparison/, per-seed metrics in $RUNS/*/test_metrics_multi_seed.csv,"
echo "figures and per-seed artifacts in each run's artifacts/ directory."
echo "For the DyGLib baselines see RELEASE.md section 4."
