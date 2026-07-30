"""Feature Projection (15→d_model) and SSL reconstruction heads."""

from typing import Dict
import torch
from torch import nn


class FeatureProjection(nn.Module):
    """Maps 15-dim normalized market features to d_model."""

    def __init__(self, feature_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ReconstructionHead(nn.Module):
    """SSL-only reconstruction head for Masked Market Modeling.

    Predicts masked features from latent representations.
    Price group → Huber; funding/OI → MSE; calendar → CE per field.
    """

    def __init__(self, d_model: int, calendar_spec: Dict):
        super().__init__()
        self.price_head = nn.Linear(d_model, 5)
        self.funding_oi_head = nn.Linear(d_model, 2)
        self.calendar_heads = nn.ModuleDict()
        for field, spec in calendar_spec.items():
            self.calendar_heads[field] = nn.Linear(d_model, spec["classes"])

    def forward(self, latent: torch.Tensor) -> Dict:
        return {
            "price": self.price_head(latent),
            "funding_oi": self.funding_oi_head(latent),
            "calendar": {name: head(latent) for name, head in self.calendar_heads.items()},
        }
