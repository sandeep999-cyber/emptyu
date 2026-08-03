"""Rotary Position Embeddings (RoPE) with time-aware positions.

Positions are derived from timestamps (epoch ms) so that relative
distances reflect actual time gaps, not just array index.
CLS token gets position 0; data tokens get 1 + minute_offset.
"""

import torch
from torch import nn


class RotaryPositionalEncoding(nn.Module):
    """Pre-computed RoPE cos/sin cache. Apply to Q and K."""

    def __init__(self, d_model: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        self.register_buffer("_cache_cos", None, persistent=False)
        self.register_buffer("_cache_sin", None, persistent=False)
        self._cache_max_len = 0
        # Set by the encoder forward pass once it has pre-sized the cache for
        # the current batch. When False, apply_rope self-checks every call so
        # direct callers (e.g. tests) can grow the cache safely.
        self._cache_ready = False

    def build_cache(self, max_seq_len: int, device: torch.device):
        if max_seq_len <= self._cache_max_len:
            return
        d_half = self.d_model // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_half, 2, dtype=torch.float32, device=device) / d_half))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self._cache_cos = angles.cos()
        self._cache_sin = angles.sin()
        self._cache_max_len = max_seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # Cache sizing is hoisted to the encoder forward pass (one host sync per
        # forward instead of one per transformer block). Direct callers that never
        # pre-built the cache fall back to the self-checking path.
        if not self._cache_ready:
            self.build_cache(int(positions.max().item()) + 1, q.device)

        cos = self._cache_cos[positions.long()]  # [..., T, d_half]
        sin = self._cache_sin[positions.long()]  # [..., T, d_half]

        # Ensure [B, T, d_half] by broadcasting
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).expand(q.shape[0], -1, -1)
            sin = sin.unsqueeze(0).expand(q.shape[0], -1, -1)

        # Interleave cos/sin -> [B, T, d_model]
        cos = torch.stack([cos, cos], dim=-1).view(*cos.shape[:-1], -1)[..., : self.d_model]
        sin = torch.stack([sin, sin], dim=-1).view(*sin.shape[:-1], -1)[..., : self.d_model]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
