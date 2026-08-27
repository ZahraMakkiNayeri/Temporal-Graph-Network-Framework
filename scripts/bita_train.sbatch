#!/bin/bash
# FULL multi-seed suite (seed count set by --n-runs). Long job -> slowlane.
#   sbatch scripts/bita_train.sbatch
#   sbatch scripts/bita_train.sbatch --processed-dir data/processed --prefix nocat
# Resumable: if the job is killed or times out, resubmitting continues with the
# seeds not yet written to test_metrics_multi_seed.csv.
#SBATCH --job-name=bita
#SBATCH --partition=slowlane
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Unbuffered stdout: otherwise print() output is lost when a job is killed
# by the wall clock, which hides exactly the lines you need to diagnose it.
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/bita-tgn-v3/bita_v3_env/bin/activate

echo "node: $(hostname)"; nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python run_train.py "$@"
