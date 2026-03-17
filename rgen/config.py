"""RGEN model configuration with YAML loading support."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RGENConfig:
    """All hyperparameters for the RGEN model (section 3.1 of spec)."""

    # Tokenizer
    vocab_size: int = 32000

    # Shared embedding
    d_model: int = 512

    # Reasoner
    reasoner_layers: int = 2
    reasoner_heads: int = 8
    reasoner_d_ff: int = 1024
    reasoner_iterations: int = 8
    reasoner_max_iterations: int = 16  # upper bound for ACT (v2)

    # Generator
    generator_layers: int = 6
    generator_heads: int = 8
    generator_d_ff: int = 2048
    max_seq_len: int = 2048

    # Training
    dropout: float = 0.1
    norm_eps: float = 1e-5

    # Auxiliary loss weights
    lambda_reconstruction: float = 0.1
    lambda_diversity: float = 0.01

    def estimated_params(self) -> dict:
        """Rough parameter count breakdown."""
        embed = self.vocab_size * self.d_model
        # Reasoner: each layer has self-attn (4 * d^2) + FFN (3 * d * d_ff)
        reasoner_per_layer = 4 * self.d_model ** 2 + 3 * self.d_model * self.reasoner_d_ff
        reasoner = self.reasoner_layers * reasoner_per_layer + self.d_model ** 2  # z_proj
        # Generator: each layer has self-attn + cross-attn (8 * d^2) + FFN (3 * d * d_ff)
        gen_per_layer = 8 * self.d_model ** 2 + 3 * self.d_model * self.generator_d_ff
        generator = self.generator_layers * gen_per_layer + self.d_model * self.vocab_size  # lm_head
        total = embed + reasoner + generator
        return {
            "embedding": embed,
            "reasoner": reasoner,
            "generator": generator,
            "total": total,
            "total_M": round(total / 1e6, 1),
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> RGENConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

# ~30M params — fast experiments and tests
TINY_CONFIG = RGENConfig(
    d_model=256,
    reasoner_layers=2,
    reasoner_heads=4,
    reasoner_d_ff=512,
    reasoner_iterations=4,
    reasoner_max_iterations=8,
    generator_layers=4,
    generator_heads=4,
    generator_d_ff=1024,
    max_seq_len=512,
)

# ~111M params — Apple Silicon target (section 3.1 defaults)
SMALL_CONFIG = RGENConfig()
