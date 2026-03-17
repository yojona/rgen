#!/bin/bash
# Phase 1: Pretrain RGEN small (~70M params, K=1) on full dataset.
# TinyStories + Wikipedia ES + OpenWebMath
#
# Launch:
#   nohup caffeinate -s bash scripts/train_small.sh > logs/train_small.log 2>&1 &
# Monitor:
#   tail -f logs/train_small.log

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p logs

echo "=== RGEN small phase 1 (K=1) ==="
echo "Started: $(date)"

python3 train/train.py \
    --config config/small.yaml \
    --data-dir data/tokenized \
    --output-dir checkpoints/small_phase1

echo "Finished: $(date)"
