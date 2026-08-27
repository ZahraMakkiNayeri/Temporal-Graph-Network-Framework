#!/bin/bash
# PILOT: sanity + timing measurement. Short job -> fastlane (4h limit).
#   sbatch scripts/bita_pilot.sbatch
#SBATCH --job-name=bita-pilot
#SBATCH --partition=fastlane
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Unbuffered stdout: otherwise print() output is lost when a job is killed
# by the wall clock, which hides exactly the lines you need to diagnose it.
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/bita-tgn-v3/bita_v3_env/bin/activate

echo "node: $(hostname)"; nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

python run_train.py --pilot "$@"
