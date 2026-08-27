#!/bin/bash
# Submit every in-framework baseline under the IDENTICAL protocol as BiTA.
# Each model gets its own directory so the summary CSVs don't collide.
#
#   bash scripts/run_all_baselines.sh data/processed_cat 100
#
set -e
PROCESSED=${1:-data/processed_cat}
Q=${2:-100}
SRC=$(pwd)
RUNS=${RUNS_DIR:-$HOME/runs}
mkdir -p "$RUNS"

submit () {                       # submit <name> <extra run_train.py flags...>
  local name=$1; shift
  local dir="$RUNS/$name"
  rm -rf "$dir"; cp -r "$SRC" "$dir"
  rm -f "$dir/test_metrics_multi_seed.csv"
  ( cd "$dir" && sbatch --job-name="$name" scripts/bita_train.sbatch \
        --processed-dir "$PROCESSED" --link-ranking-candidates "$Q" \
        --prefix "$name" "$@" )
  echo "submitted $name -> $dir"
}

# --- TGN variants: only the aggregator differs (the paper's fair comparison) --
submit tgn_last  --aggregator last
submit tgn_mean  --aggregator mean
submit tgn_attn  --aggregator bitransformer

# --- DyRep (Rossi et al.'s emulation: RNN memory + source embedding messages) --
submit dyrep --dyrep true --memory-updater rnn --use-source-embedding-in-message true

# --- BiTA aggregator ablations (Tables 6-8, Experiment 10) --------------------
submit bita_temporal --aggregator bitransformer_temporal
submit bita_relative --aggregator relative_transformer
submit bita_stacked  --aggregator stacked_bitransformer
submit bita_tcn      --aggregator tcn

echo
echo "watch:   squeue -u $USER"
echo "collect: python scripts/collect_results.py --run BiTA=$SRC \\"
echo "             --run TGN-Last=$RUNS/tgn_last --run TGN-Mean=$RUNS/tgn_mean \\"
echo "             --run DyRep=$RUNS/dyrep --run EdgeBank=$RUNS/edgebank"
