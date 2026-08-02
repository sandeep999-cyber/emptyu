import pytest
import torch
import numpy as np

from src.training.losses.masked_modeling import MaskedMarketModelingLoss


LOSS_CFG = {
    "price_indices": list(range(5)),
    "funding_oi_indices": [5, 6],
    "calendar": {
        "hour_sin": {"index": 5, "classes": 24, "offset": 0},
        "hour_cos": {"index": 5, "classes": 24, "offset": 0},
        "minute_sin": {"index": 7, "classes": 60, "offset": 0},
        "minute_cos": {"index": 7, "classes": 60, "offset": 0},
        "day_of_week_sin": {"index": 9, "classes": 7, "offset": 0},
        "day_of_week_cos": {"index": 9, "classes": 7, "offset": 0},
        "month_sin": {"index": 11, "classes": 12, "offset": 1},
        "month_cos": {"index": 11, "classes": 12, "offset": 1},
    },
    "group_weights": {"price": 1.0, "funding_oi": 1.0, "calendar": 1.0},
}


class TestMaskedMarketModelingLoss:
    @pytest.fixture
    def loss_fn(self):
        return MaskedMarketModelingLoss(LOSS_CFG, d_model=512, device=torch.device("cpu"))

    def test_forward(self, loss_fn):
        B, T = 2, 512
        reconstruction = {
            "price": torch.randn(B, T, 5),
            "funding_oi": torch.randn(B, T, 2),
            "calendar": {f: torch.randn(B, T, spec["classes"]) for f, spec in LOSS_CFG["calendar"].items()},
        }
        features_norm = torch.randn(B, T, 15)
        features_raw = torch.randint(0, 60, (B, T, 15)).float()
        feature_mask = torch.ones(B, T, 15, dtype=torch.bool)
        masked_positions = torch.rand(B, T) < 0.15

        losses = loss_fn(reconstruction, features_norm, features_raw, feature_mask, masked_positions)
        assert "total" in losses
        assert losses["total"].item() > 0
        assert "price" in losses
        assert "funding_oi" in losses

    def test_no_mask(self, loss_fn):
        B, T = 2, 64
        reconstruction = {
            "price": torch.randn(B, T, 5),
            "funding_oi": torch.randn(B, T, 2),
            "calendar": {f: torch.randn(B, T, spec["classes"]) for f, spec in LOSS_CFG["calendar"].items()},
        }
        features_norm = torch.randn(B, T, 15)
        features_raw = torch.randint(0, 60, (B, T, 15)).float()
        feature_mask = torch.ones(B, T, 15, dtype=torch.bool)
        masked_positions = torch.zeros(B, T, dtype=torch.bool)

        losses = loss_fn(reconstruction, features_norm, features_raw, feature_mask, masked_positions)
        assert losses["total"] == 0.0

    def test_gradient_flow(self, loss_fn):
        B, T = 2, 32
        pred = torch.randn(B, T, 5, requires_grad=True)
        reconstruction = {
            "price": pred,
            "funding_oi": torch.randn(B, T, 2),
            "calendar": {f: torch.randn(B, T, spec["classes"]) for f, spec in LOSS_CFG["calendar"].items()},
        }
        features_norm = torch.randn(B, T, 15)
        features_raw = torch.randint(0, 60, (B, T, 15)).float()
        feature_mask = torch.ones(B, T, 15, dtype=torch.bool)
        masked_positions = torch.ones(B, T, dtype=torch.bool)

        losses = loss_fn(reconstruction, features_norm, features_raw, feature_mask, masked_positions)
        losses["total"].backward()
        assert pred.grad is not None

    def test_reconstruct_calendar_false_skips_calendar(self):
        cfg = {**LOSS_CFG, "reconstruct_calendar": False}
        loss_fn = MaskedMarketModelingLoss(cfg, d_model=512, device=torch.device("cpu"))
        B, T = 2, 32
        reconstruction = {
            "price": torch.randn(B, T, 5),
            "funding_oi": torch.randn(B, T, 2),
            "calendar": {f: torch.randn(B, T, spec["classes"]) for f, spec in LOSS_CFG["calendar"].items()},
        }
        features_norm = torch.randn(B, T, 15)
        features_raw = torch.randint(0, 60, (B, T, 15)).float()
        feature_mask = torch.ones(B, T, 15, dtype=torch.bool)
        masked_positions = torch.rand(B, T) < 0.5

        losses = loss_fn(reconstruction, features_norm, features_raw, feature_mask, masked_positions)
        assert "calendar" not in losses
        assert any(f in losses for f in LOSS_CFG["calendar"]) is False
        # total must equal the weighted price + funding_oi terms only.
        expected = losses["price"] + losses["funding_oi"]
        assert np.isclose(losses["total"].item(), expected.item())

    def test_reconstruct_calendar_default_includes_calendar(self, loss_fn):
        B, T = 2, 32
        reconstruction = {
            "price": torch.randn(B, T, 5),
            "funding_oi": torch.randn(B, T, 2),
            "calendar": {f: torch.randn(B, T, spec["classes"]) for f, spec in LOSS_CFG["calendar"].items()},
        }
        features_norm = torch.randn(B, T, 15)
        features_raw = torch.randint(0, 60, (B, T, 15)).float()
        feature_mask = torch.ones(B, T, 15, dtype=torch.bool)
        masked_positions = torch.rand(B, T) < 0.5
        losses = loss_fn(reconstruction, features_norm, features_raw, feature_mask, masked_positions)
        assert "calendar" in losses
