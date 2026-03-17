"""
RGEN low-level layers: RMSNorm, RoPE, SwiGLUFFN, MultiHeadAttention, TransformerLayer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE.

    Returns a tensor of shape [max_seq_len, d_head // 2] with complex values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, d_head // 2]
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: [batch, seq_len, n_heads, d_head]
        freqs_cis: [seq_len, d_head // 2] complex
    """
    # Reshape x into pairs: [batch, seq_len, n_heads, d_head//2, 2]
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    # Convert to complex: [batch, seq_len, n_heads, d_head//2]
    x_complex = torch.view_as_complex(x_pairs)
    # Broadcast freqs: [1, seq_len, 1, d_head//2]
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    # Apply rotation
    x_rotated = x_complex * freqs
    # Back to real: [batch, seq_len, n_heads, d_head//2, 2]
    x_out = torch.view_as_real(x_rotated)
    # Flatten last two dims back to d_head
    return x_out.reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: FFN(x) = (W1·x ⊙ SiLU(W3·x)) · W2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.w1(x) * F.silu(self.w3(x))))


# ---------------------------------------------------------------------------
# Multi-Head Attention (with RoPE)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with RoPE support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        rope_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: query input [batch, seq_len, d_model]
            key: optional key input for cross-attention [batch, src_len, d_model]
            value: optional value input for cross-attention [batch, src_len, d_model]
            causal_mask: boolean mask [seq_len, seq_len], True = masked positions
            rope_freqs: precomputed RoPE frequencies (applied only to self-attention)
        """
        is_cross = key is not None
        if key is None:
            key = x
        if value is None:
            value = key

        B, T, _ = x.shape
        S = key.shape[1]

        # Project to heads: [B, seq, n_heads, d_head]
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(key).view(B, S, self.n_heads, self.d_head)
        v = self.wv(value).view(B, S, self.n_heads, self.d_head)

        # Apply RoPE to Q and K (only for self-attention)
        if rope_freqs is not None and not is_cross:
            q = apply_rope(q, rope_freqs[:T])
            k = apply_rope(k, rope_freqs[:S])

        # Transpose to [B, n_heads, seq, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, S]

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask[:T, :S].unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with optional cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_cross_attention: bool,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm3 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cross_attn_context: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        rope_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), causal_mask=causal_mask, rope_freqs=rope_freqs)

        # Cross-attention
        if self.use_cross_attention and cross_attn_context is not None:
            normed = self.norm2(x)
            x = x + self.cross_attn(
                normed,
                key=cross_attn_context,
                value=cross_attn_context,
            )

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x
