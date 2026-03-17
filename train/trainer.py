"""
RGEN training loop (section 5.4).

Features:
  - Mixed precision (float16) via torch.cuda.amp / torch.cpu.amp
  - Gradient accumulation over configurable steps
  - Gradient clipping (max_norm=1.0)
  - EMA of Reasoner weights (decay=0.999, inference only)
  - Curriculum unfreezing:
      Phase A (steps 0–5K):    Reasoner frozen, only Generator trains
      Phase B (steps 5K–20K):  Reasoner unfrozen at 0.1× learning rate
      Phase C (steps 20K+):    Full joint training
  - Checkpoint saving every N steps
  - Metric logging every N steps
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Force line-buffered output so logs appear immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rgen.config import RGENConfig
from rgen.model import RGEN
from train.dataset import build_dataset
from train.losses import ReconstructionHead, compute_loss
from train.scheduler import cosine_with_warmup


@dataclass
class TrainConfig:
    """Training hyperparameters (section 5.5)."""

    # Data
    data_paths: list = field(default_factory=list)
    max_seq_len: int = 512

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    num_workers: int = 0  # macOS fork() is unsafe; use 0 by default

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000

    # Curriculum unfreezing thresholds
    generator_only_steps: int = 5000
    reasoner_warmup_steps: int = 20000
    reasoner_lr_scale: float = 0.1

    # EMA
    ema_decay: float = 0.999

    # Logging & checkpoints
    log_interval: int = 100
    save_interval: int = 5000
    output_dir: str = "checkpoints"

    # Device
    device: str = "cpu"
    use_amp: bool = False  # auto-detected if cuda available


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the Reasoner weights updated as:
        ema = decay * ema + (1 - decay) * current
    Used only for inference, not for backprop.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in module.named_parameters()
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, module: nn.Module) -> dict:
        """Replace module params with EMA values. Returns originals for restore."""
        originals = {}
        for name, param in module.named_parameters():
            originals[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return originals

    def restore(self, module: nn.Module, originals: dict) -> None:
        """Restore original params after EMA was applied."""
        for name, param in module.named_parameters():
            param.data.copy_(originals[name])

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# Curriculum unfreezing
# ---------------------------------------------------------------------------

def _apply_curriculum(
    model: RGEN,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
) -> str:
    """Manage Reasoner freeze/unfreeze schedule.

    Returns the current phase name for logging.
    """
    if step < cfg.generator_only_steps:
        # Phase A: freeze Reasoner entirely
        for p in model.reasoner.parameters():
            p.requires_grad = False
        return "generator_only"

    elif step < cfg.reasoner_warmup_steps:
        # Phase B: unfreeze Reasoner with reduced LR
        for p in model.reasoner.parameters():
            p.requires_grad = True
        # Set Reasoner param group LR to base * scale
        for group in optimizer.param_groups:
            if group.get("name") == "reasoner":
                group["lr"] = optimizer.param_groups[0]["lr"] * cfg.reasoner_lr_scale
        return "reasoner_warmup"

    else:
        # Phase C: full training, all at same LR
        for p in model.reasoner.parameters():
            p.requires_grad = True
        return "joint"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Main training loop for RGEN."""

    def __init__(self, model_config: RGENConfig, train_config: TrainConfig):
        self.mcfg = model_config
        self.tcfg = train_config
        self.device = torch.device(train_config.device)

        # Auto-detect AMP
        if self.device.type == "cuda":
            self.tcfg.use_amp = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        # Model
        self.model = RGEN(model_config).to(self.device)
        self.recon_head = ReconstructionHead(model_config.d_model).to(self.device)

        # Optimizer — separate param groups for curriculum LR
        reasoner_params = list(self.model.reasoner.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith("reasoner.")
        ] + list(self.recon_head.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "name": "main"},
                {"params": reasoner_params, "name": "reasoner"},
            ],
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            weight_decay=train_config.weight_decay,
        )

        # Scheduler
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            max_lr=train_config.learning_rate,
            min_lr=train_config.min_lr,
        )

        # EMA for Reasoner
        self.ema = EMA(self.model.reasoner, decay=train_config.ema_decay)

        # AMP scaler
        self.scaler = torch.amp.GradScaler(enabled=self.tcfg.use_amp)

        # Dataset
        self.dataset = build_dataset(
            train_config.data_paths,
            max_seq_len=model_config.max_seq_len,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        # State
        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")

        # Output dir
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.tcfg
        model = self.model
        model.train()

        print(f"Training RGEN: {self.mcfg.estimated_params()['total_M']}M params")
        print(f"  Device: {self.device}")
        print(f"  AMP: {cfg.use_amp}")
        print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")
        print(f"  Dataset samples: {len(self.dataset):,}")
        print(f"  Max steps: {cfg.max_steps:,}")
        print()

        data_iter = iter(self.dataloader)
        self.optimizer.zero_grad()

        metrics_accum = {}
        steps_since_log = 0
        t_start = time.time()

        while self.global_step < cfg.max_steps:
            # --- Curriculum (applied once per optimizer step) ---
            phase = _apply_curriculum(model, self.optimizer, self.global_step, cfg)

            # --- Accumulate gradients over N micro-batches ---
            for micro in range(cfg.gradient_accumulation_steps):
                try:
                    input_ids, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    input_ids, targets = next(data_iter)

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                amp_device = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type=amp_device, enabled=cfg.use_amp):
                    x_emb = model.embedding(input_ids)

                    if self.global_step < cfg.generator_only_steps:
                        z_star = torch.zeros_like(x_emb)
                    else:
                        z_star = model.reasoner(x_emb)

                    logits = model.generator(
                        x_emb, z_star,
                        causal_mask=model.causal_mask,
                        rope_freqs=model.rope_freqs,
                    )

                    loss, metrics = compute_loss(
                        logits, targets, z_star, x_emb,
                        self.recon_head,
                        lambda_reconstruction=self.mcfg.lambda_reconstruction,
                        lambda_diversity=self.mcfg.lambda_diversity,
                    )
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                for k, v in metrics.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v / cfg.gradient_accumulation_steps

                self.tokens_seen += input_ids.numel()

            # --- Optimizer step (once per global_step) ---
            self.scaler.unscale_(self.optimizer)
            all_params = list(model.parameters()) + list(self.recon_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # EMA update (only when Reasoner is training)
            if self.global_step >= cfg.generator_only_steps:
                self.ema.update(model.reasoner)

            self.global_step += 1
            steps_since_log += 1

            # --- Logging ---
            if self.global_step % cfg.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                lr = self.optimizer.param_groups[0]["lr"]

                # Average metrics over steps since last log
                avg_metrics = {k: v / steps_since_log for k, v in metrics_accum.items()}

                log_line = (
                    f"step {self.global_step:>6d}/{cfg.max_steps} | "
                    f"loss {avg_metrics.get('loss', 0):.4f} | "
                    f"lm {avg_metrics.get('lm_loss', 0):.4f} | "
                    f"recon {avg_metrics.get('reconstruction_loss', 0):.6f} | "
                    f"div {avg_metrics.get('diversity_loss', 0):.4f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tokens_per_sec:.0f} | "
                    f"phase {phase}"
                )
                print(log_line)

                log_entry = {
                    "step": self.global_step,
                    "phase": phase,
                    "lr": lr,
                    "tokens_seen": self.tokens_seen,
                    "elapsed_s": elapsed,
                    **avg_metrics,
                }
                with open(self.output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                metrics_accum = {}
                steps_since_log = 0

            # --- Checkpoint ---
            if self.global_step % cfg.save_interval == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(tag="final")
        print(f"\nTraining complete. {self.global_step:,} steps, {self.tokens_seen:,} tokens.")

    def _save_checkpoint(self, tag: Optional[str] = None) -> None:
        name = tag or f"step_{self.global_step}"
        path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save({
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "model": self.model.state_dict(),
            "recon_head": self.recon_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
            "model_config": self.mcfg.__dict__,
            "train_config": self.tcfg.__dict__,
        }, path)
        print(f"  [checkpoint] saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.recon_head.load_state_dict(ckpt["recon_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.ema.load_state_dict(ckpt["ema"])
        self.global_step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        print(f"Resumed from {path} at step {self.global_step:,}")
