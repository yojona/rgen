#!/bin/bash
# Train RGEN tiny (30M params) on TinyStories only for 2000 steps.
# Validates the full training pipeline end-to-end.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=.

# Create a temp dir with just tinystories
mkdir -p data/tokenized/tinystories_only
ln -sf "$(pwd)/data/tokenized/tinystories.bin" data/tokenized/tinystories_only/tinystories.bin

python3 train/train.py \
    --config config/tiny.yaml \
    --data-dir data/tokenized/tinystories_only \
    --output-dir checkpoints/tiny_tinystories
