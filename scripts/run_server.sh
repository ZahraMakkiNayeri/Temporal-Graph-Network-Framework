#!/bin/bash
# Plain-SSH alternative to SLURM (nohup keeps it alive after logout):
#   bash scripts/run_server.sh --pilot
#   bash scripts/run_server.sh                # full suite
set -e
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs
nohup python run_train.py "$@" > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "started PID $! — follow with: tail -f logs/train_*.log"
