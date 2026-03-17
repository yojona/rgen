#!/bin/bash
# Phase 2: Fine-tune RGEN small (K=8) on reasoning data.
# Resumes from phase 1 checkpoint, increases Reasoner depth.
#
# Launch:
#   nohup caffeinate -s bash scripts/train_finetune.sh > logs/train_finetune.log 2>&1 &
# Monitor:
#   tail -f logs/train_finetune.log

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p logs

# Find latest phase 1 checkpoint
CKPT=$(ls -t checkpoints/small_phase1/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No phase 1 checkpoint found in checkpoints/small_phase1/"
    exit 1
fi

echo "=== RGEN small phase 2 (K=8) ==="
echo "Resuming from: $CKPT"
echo "Started: $(date)"

python3 train/train.py \
    --config config/small_phase2.yaml \
    --data-dir data/tokenized \
    --output-dir checkpoints/small_phase2 \
    --resume "$CKPT"

echo "Finished: $(date)"
