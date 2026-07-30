import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.trainer import TeacherTrainer


MODEL_CFG = {
    "model": {
        "name": "teacher_transformer_v1",
        "feature_dim": 15,
        "context_length": 512,
        "cls_token": True,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "rope_theta": 10000.0,
    },
    "loss": {
        "masked_modeling": {
            "mask_ratio": 0.15,
            "price_indices": [0, 1, 2, 3, 4],
            "funding_oi_indices": [5, 6],
            "calendar": {
                "minute_of_day": {"index": 7, "classes": 1440, "offset": 0},
                "hour": {"index": 8, "classes": 24, "offset": 0},
                "day_of_week": {"index": 9, "classes": 7, "offset": 0},
                "day_of_month": {"index": 10, "classes": 31, "offset": 1},
                "month": {"index": 11, "classes": 12, "offset": 1},
                "quarter": {"index": 12, "classes": 4, "offset": 1},
                "year": {"index": 13, "classes": 16, "offset": 2020},
                "is_weekend": {"index": 14, "classes": 2, "offset": 0},
            },
            "group_weights": {"price": 1.0, "funding_oi": 1.0, "calendar": 1.0},
        }
    },
}

OPT_CFG = {
    "optimizer": {
        "adamw": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "grad_clip": 1.0,
    },
    "scheduler": {"warmup_frac": 0.05, "lr_floor": 1e-6},
}

TRAINER_CFG = {
    "epochs": 2,
    "batch_size": 4,
    "market": "futures",
    "train_window_stride": 16,
    "val_window_stride": 16,
    "device": "cpu",
    "seed": 42,
    "log_every": 50,
    "normalizer": {"mode": "zscore"},
}


@pytest.fixture
def mock_trainer():
    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch("src.training.trainer._load_manifest") as mock_mf,
            patch("src.training.trainer.lake") as mock_lake,
            patch("src.training.trainer.feature_builder") as mock_fb,
            patch("src.training.trainer.windowing_engine") as mock_we,
        ):
            mock_mf.return_value = {
                "splits": {
                    "train": {"symbols": ["BTCUSDT"]},
                    "validation": {"symbols": ["SOLUSDT"]},
                }
            }
            mock_df = MagicMock()
            mock_df.empty = False
            mock_lake.market_state.return_value = mock_df
            mock_fb.build_features.return_value = (
                torch.randn(1000, 15).numpy(),
                torch.ones(1000, 15, dtype=torch.bool).numpy(),
                [0] * 1000,
            )
            mock_we.create_windows.return_value = [
                {"features": torch.randn(512, 15).numpy(),
                 "feature_mask": torch.ones(512, 15, dtype=torch.bool).numpy(),
                 "timestamps": list(range(512)),
                 "mask": torch.ones(512, dtype=torch.bool).numpy(),
                 "metadata": {"symbol": "BTCUSDT", "snapshot_id": "2026-07-30", "market": "futures"}}
                for _ in range(10)
            ]

            trainer = TeacherTrainer(MODEL_CFG, OPT_CFG, TRAINER_CFG, Path(tmp))
            yield trainer


class TestTeacherTrainer:
    def test_init(self, mock_trainer):
        assert mock_trainer.model is not None
        assert mock_trainer.device == torch.device("cpu")

    def test_model_forward(self, mock_trainer):
        mock_trainer.model.eval()
        features = torch.randn(1, 512, 15)
        timestamps = torch.arange(512).unsqueeze(0)
        mask = torch.ones(1, 512, dtype=torch.bool)
        with torch.no_grad():
            latent, kpm, positions, t_data = mock_trainer.model(features, timestamps, mask)
        assert latent.shape == (1, 513, 128)

    def test_model_reconstruct(self, mock_trainer):
        mock_trainer.model.eval()
        latent = torch.randn(1, 512, 128)
        rec = mock_trainer.model.reconstruct(latent)
        assert "price" in rec
        assert "funding_oi" in rec
