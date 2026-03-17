"""
Loss functions for RGEN training (section 5.3).

Total loss = lm_loss
           + lambda_reconstruction * reconstruction_loss
           + lambda_diversity * diversity_loss

The auxiliary losses prevent Reasoner collapse:
  - reconstruction_loss: forces z* to encode information about the input
  - diversity_loss: penalizes z* collapsing to a constant across the batch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    """Small MLP that predicts the mean input embedding from mean z*.

    Trained jointly — gradient flows back through z* to the Reasoner,
    forcing it to retain information about the input.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_mean: torch.Tensor) -> torch.Tensor:
        return self.proj(z_mean)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_star: torch.Tensor,
    x_embedding: torch.Tensor,
    reconstruction_head: ReconstructionHead,
    lambda_reconstruction: float = 0.1,
    lambda_diversity: float = 0.01,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict]:
    """Compute total RGEN loss with auxiliary terms.

    Args:
        logits: [batch, seq_len, vocab_size] — generator output
        targets: [batch, seq_len] — target token ids
        z_star: [batch, seq_len, d_model] — reasoner output
        x_embedding: [batch, seq_len, d_model] — input embeddings
        reconstruction_head: MLP for reconstruction loss
        lambda_reconstruction: weight for reconstruction loss
        lambda_diversity: weight for diversity loss
        ignore_index: token id to ignore in LM loss (padding)

    Returns:
        (total_loss, metrics_dict)
    """
    # 1. Language modeling loss — standard cross-entropy
    lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )

    # 2. Reconstruction loss — z* must encode input information
    #    Mean-pool over sequence, then predict input mean from z* mean
    z_mean = z_star.mean(dim=1)          # [batch, d_model]
    x_mean = x_embedding.mean(dim=1)     # [batch, d_model]
    z_reconstructed = reconstruction_head(z_mean)
    reconstruction_loss = F.mse_loss(z_reconstructed, x_mean.detach())

    # 3. Diversity loss — z* must vary across batch examples
    #    Penalize low variance across the batch dimension
    #    z_star: [batch, seq_len, d_model] → std across batch dim
    z_std = z_star.std(dim=0).mean()     # scalar: avg std per position/dim
    diversity_loss = -torch.log(z_std + 1e-8)

    # Total
    total_loss = (
        lm_loss
        + lambda_reconstruction * reconstruction_loss
        + lambda_diversity * diversity_loss
    )

    metrics = {
        "loss": total_loss.item(),
        "lm_loss": lm_loss.item(),
        "reconstruction_loss": reconstruction_loss.item(),
        "diversity_loss": diversity_loss.item(),
    }

    return total_loss, metrics
