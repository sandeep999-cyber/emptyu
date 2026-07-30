"""Pre-LN Transformer block with RoPE in self-attention."""

import torch
from torch import nn
import torch.nn.functional as F
from src.models.teacher.positional_encoding import RotaryPositionalEncoding


class TransformerBlock(nn.Module):
    """Pre-LN block: LN → QKV proj + RoPE → SDPA → output proj → residual → LN → MLP → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, rope: RotaryPositionalEncoding):
        super().__init__()
        self.rope = rope
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Pre-LN attention
        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm)  # [B, T, D]
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q, k = self.rope.apply_rope(q, k, positions)

        # Reshape to [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # key_padding_mask: [B, T] — True = valid (attend), False = padding
        # SDPA with bool mask: True = allowed
        attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T] for broadcast
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # Pre-LN MLP
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x
