#!/usr/bin/env python3
"""
CLI entry point for RGEN training.

Usage:
  python train/train.py --config config/tiny.yaml
  python train/train.py --config config/tiny.yaml --max-steps 10
  python train/train.py --config config/small.yaml --resume checkpoints/checkpoint_step_5000.pt
  python train/train.py --config config/tiny.yaml --eval-only
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from dataclasses import fields
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rgen.config import RGENConfig
from train.trainer import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train RGEN model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (model + training params)")
    parser.add_argument("--data-dir", type=str, default="data/tokenized",
                        help="Directory with .bin tokenized files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint output directory (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--log-interval", type=int, default=None,
                        help="Override log_interval from config")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, do not train")
    args = parser.parse_args()

    # ----- Load YAML -----
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # ----- Model config (filter to RGENConfig fields) -----
    model_fields = {fld.name for fld in fields(RGENConfig)}
    model_kwargs = {k: v for k, v in raw.items() if k in model_fields}
    mcfg = RGENConfig(**model_kwargs)

    # ----- Training config (filter to TrainConfig fields) -----
    train_fields = {fld.name for fld in fields(TrainConfig)}
    train_kwargs = {k: v for k, v in raw.items() if k in train_fields}

    # Discover .bin files in data-dir
    data_dir = Path(args.data_dir)
    bin_files = sorted(glob.glob(str(data_dir / "*.bin")))
    if not bin_files:
        print(f"ERROR: no .bin files in {data_dir}", file=sys.stderr)
        sys.exit(1)
    train_kwargs["data_paths"] = bin_files
    train_kwargs["max_seq_len"] = mcfg.max_seq_len

    # CLI overrides
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    if args.log_interval is not None:
        train_kwargs["log_interval"] = args.log_interval
    if args.output_dir is not None:
        train_kwargs["output_dir"] = args.output_dir

    tcfg = TrainConfig(**train_kwargs)

    # ----- Print summary -----
    params = mcfg.estimated_params()
    print("=" * 60)
    print("RGEN Training")
    print("=" * 60)
    print(f"  Config:        {config_path}")
    print(f"  Model params:  {params['total_M']}M")
    print(f"    embedding:   {params['embedding']/1e6:.1f}M")
    print(f"    reasoner:    {params['reasoner']/1e6:.1f}M")
    print(f"    generator:   {params['generator']/1e6:.1f}M")
    print(f"  d_model:       {mcfg.d_model}")
    print(f"  max_seq_len:   {mcfg.max_seq_len}")
    print(f"  Data files:    {len(bin_files)}")
    for bf in bin_files:
        print(f"    - {bf}")
    print(f"  Batch size:    {tcfg.batch_size} × {tcfg.gradient_accumulation_steps} = {tcfg.batch_size * tcfg.gradient_accumulation_steps}")
    print(f"  LR:            {tcfg.learning_rate} -> {tcfg.min_lr}")
    print(f"  Max steps:     {tcfg.max_steps:,}")
    print(f"  Warmup:        {tcfg.warmup_steps}")
    print(f"  Curriculum:    generator_only={tcfg.generator_only_steps}, reasoner_warmup={tcfg.reasoner_warmup_steps}")
    print(f"  Output:        {tcfg.output_dir}")
    if args.resume:
        print(f"  Resume from:   {args.resume}")
    print("=" * 60)
    print()

    if args.eval_only:
        print("--eval-only: evaluation not yet implemented.")
        sys.exit(0)

    # ----- Train -----
    trainer = Trainer(mcfg, tcfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ----- Summary -----
    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Steps completed: {trainer.global_step:,}")
    print(f"  Tokens seen:     {trainer.tokens_seen:,}")
    print(f"  Wall time:       {elapsed/60:.1f} min")
    print(f"  Tokens/sec:      {trainer.tokens_seen/elapsed:.0f}")
    print(f"  Checkpoints:     {tcfg.output_dir}")

    metrics_path = Path(tcfg.output_dir) / "metrics.jsonl"
    if metrics_path.exists():
        import json
        lines = metrics_path.read_text().strip().split("\n")
        if lines:
            first = json.loads(lines[0])
            last = json.loads(lines[-1])
            print(f"  Loss start:      {first.get('loss', 'n/a'):.4f} (step {first['step']})")
            print(f"  Loss end:        {last.get('loss', 'n/a'):.4f} (step {last['step']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
