"""Teacher Encoder: projection → CLS → N transformer blocks → latent."""

from typing import Dict
import torch
from torch import nn
from src.config import config as cfg
from src.models.teacher.projection import FeatureProjection, ReconstructionHead
from src.models.teacher.positional_encoding import RotaryPositionalEncoding
from src.models.teacher.transformer import TransformerBlock


class TeacherEncoder(nn.Module):
    """Encoder-only transformer producing per-timestep latent representations.

    Input: normalized features [B, T_data, 15] (T_data = context_length)
    Output: latent [B, T_data+1, d_model] + key_padding_mask [B, T_data+1]
    """

    def __init__(self, model_cfg: Dict):
        super().__init__()
        self.context_length = model_cfg["context_length"]
        self.d_model = model_cfg["d_model"]
        self.n_layers = model_cfg["n_layers"]
        self.n_heads = model_cfg["n_heads"]
        self.d_ff = model_cfg["d_ff"]
        self.dropout = model_cfg["dropout"]
        rope_theta = model_cfg.get("rope_theta", 10000.0)
        self.feature_dim = model_cfg["feature_dim"]

        self.projection = FeatureProjection(self.feature_dim, self.d_model)
        self.cls_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.rope = RotaryPositionalEncoding(self.d_model, theta=rope_theta)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout, self.rope)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

        # SSL reconstruction head
        calendar_spec = model_cfg.get("loss", {}).get("masked_modeling", {}).get("calendar", {})
        if calendar_spec:
            self.reconstruction_head = ReconstructionHead(self.d_model, calendar_spec)
        else:
            self.reconstruction_head = None

    def forward(self, features: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            features: [B, T_data, feature_dim] — normalized
            timestamps: [B, T_data] — int64 epoch ms
            mask: [B, T_data] — bool, True = valid timestep, False = padding
        Returns:
            latent: [B, T_total, d_model] where T_total = T_data
            key_padding_mask: [B, T_total] — bool, True = valid
            positions: [B, T_total]
            T_data: int — number of data positions (for slicing outside)
        """
        B, T_data = features.shape[:2]
        device = features.device

        # Project
        x = self.projection(features)  # [B, T_data, d_model]
        x = self.dropout_layer(x)

        # CLS token
        cls = self.cls_embed.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, T_total, d_model]
        T_total = T_data + 1

        # RoPE positions: CLS=0, data=1+minute_offset
        ts_start = timestamps[:, 0:1]  # [B, 1]
        minute_offsets = ((timestamps - ts_start) / 60000).long()  # [B, T_data]
        data_positions = 1 + minute_offsets
        cls_position = torch.zeros(B, 1, dtype=torch.long, device=device)
        positions = torch.cat([cls_position, data_positions], dim=1)  # [B, T_total]

        # key_padding_mask: CLS always valid, data uses sample mask
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # [B, T_total]

        # Encoder blocks
        for block in self.blocks:
            x = block(x, key_padding_mask, positions)

        x = self.norm(x)
        return x, key_padding_mask, positions, T_data

    def reconstruct(self, latent: torch.Tensor):
        if self.reconstruction_head is None:
            raise RuntimeError("Reconstruction head not initialized — missing calendar spec in config.")
        return self.reconstruction_head(latent)
