"""
Cosine decay learning rate scheduler with linear warmup (section 5.4).

Schedule:
  - Steps [0, warmup_steps):  linear ramp from 0 to max_lr
  - Steps [warmup_steps, max_steps]: cosine decay from max_lr to min_lr
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> LambdaLR:
    """Create a LambdaLR scheduler with linear warmup + cosine decay."""

    def lr_lambda(step: int) -> float:
        # Warmup phase: linear ramp
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay phase
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale between min_lr and max_lr, return as multiplier
        # (optimizer base lr should be set to max_lr)
        target_lr = min_lr + (max_lr - min_lr) * cosine
        return target_lr / max_lr

    return LambdaLR(optimizer, lr_lambda)
