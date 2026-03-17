"""RGEN model: Reasoner + Generator."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RGENConfig
from .layers import TransformerLayer, RMSNorm, precompute_rope_freqs


class Reasoner(nn.Module):
    """
    Recursive latent reasoning loop.

    Uses the SAME layers in every iteration (weight sharing).
    z is initialized to zeros, not from the input embedding.
    The input x is injected in EVERY iteration (not just the first).
    """

    def __init__(self, config: RGENConfig):
        super().__init__()
        self.K = config.reasoner_iterations
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=config.d_model,
                n_heads=config.reasoner_heads,
                d_ff=config.reasoner_d_ff,
                dropout=config.dropout,
                use_cross_attention=False,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.reasoner_layers)
        ])
        self.z_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model] — embedded input
            return_intermediates: if True, also return list of z at each iteration
        Returns:
            z_star: [batch, seq_len, d_model] — final reasoning state
            intermediates (optional): list of K tensors, one per iteration
        """
        z = torch.zeros_like(x)
        intermediates = [] if return_intermediates else None

        for _ in range(self.K):
            # Concatenate x and z along sequence dim so self-attention sees both
            xz = torch.cat([x, z], dim=1)  # [batch, 2*seq_len, d_model]

            for layer in self.layers:
                xz = layer(xz)

            # Take only the z half
            z_new = xz[:, x.size(1):, :]  # [batch, seq_len, d_model]
            z = self.z_proj(z_new)

            if return_intermediates:
                intermediates.append(z.clone())

        if return_intermediates:
            return z, intermediates
        return z


class Generator(nn.Module):
    """
    Autoregressive decoder conditioned on z* via cross-attention.

    Each layer has: self-attention (causal) -> cross-attention(z*) -> FFN.
    RoPE is applied in self-attention for positional encoding.
    """

    def __init__(self, config: RGENConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=config.d_model,
                n_heads=config.generator_heads,
                d_ff=config.generator_d_ff,
                dropout=config.dropout,
                use_cross_attention=True,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.generator_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        z_star: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        rope_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len, d_model] — token embeddings
            z_star: [batch, src_len, d_model] — reasoner output
            causal_mask: [max_seq, max_seq] bool, True = masked
            rope_freqs: precomputed RoPE complex frequencies
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = tokens
        for layer in self.layers:
            x = layer(
                x,
                cross_attn_context=z_star,
                causal_mask=causal_mask,
                rope_freqs=rope_freqs,
            )
        x = self.norm(x)
        return self.lm_head(x)


class RGEN(nn.Module):
    """Full RGEN model: Embedding + Reasoner + Generator."""

    def __init__(self, config: RGENConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.reasoner = Reasoner(config)
        self.generator = Generator(config)

        # Precompute RoPE frequencies (buffer, not parameter)
        rope_freqs = precompute_rope_freqs(
            d_head=config.d_model // config.generator_heads,
            max_seq_len=config.max_seq_len,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Causal mask (upper-triangular = True means masked)
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token ids
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)
        z_star = self.reasoner(x)
        logits = self.generator(
            x, z_star,
            causal_mask=self.causal_mask,
            rope_freqs=self.rope_freqs,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        1. Run the Reasoner ONCE on the prompt to get z*.
        2. Generate tokens one at a time, conditioning on z* (fixed).

        Args:
            input_ids: [batch, prompt_len] token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: if >0, keep only top-k logits before sampling
        Returns:
            tokens: [batch, prompt_len + max_new_tokens] full sequence
        """
        self.eval()
        device = input_ids.device

        # Step 1: compute z* once from the prompt
        prompt_emb = self.embedding(input_ids)
        z_star = self.reasoner(prompt_emb)

        generated = input_ids

        for _ in range(max_new_tokens):
            seq_len = generated.size(1)
            if seq_len > self.config.max_seq_len:
                break

            # Embed all tokens so far
            token_emb = self.embedding(generated)

            # Run generator with fixed z*
            logits = self.generator(
                token_emb, z_star,
                causal_mask=self.causal_mask,
                rope_freqs=self.rope_freqs,
            )

            # Take logits at last position
            next_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Greedy
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_vals, _ = next_logits.topk(top_k, dim=-1)
                    threshold = top_vals[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated
