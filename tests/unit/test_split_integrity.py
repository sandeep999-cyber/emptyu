"""Split-integrity tests: train/test/validation must be disjoint in symbol or time."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from src.data.market_dataset import MarketDataset
from src.training.trainer import TeacherTrainer, _build_windows_for_symbol, _date_to_ms

MANIFEST_PATH = Path("storage/training/training_manifest_v1.json")

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
    "epochs": 1,
    "batch_size": 4,
    "market": "futures",
    "train_window_stride": 1,
    "val_window_stride": 1,
    "device": "cpu",
    "seed": 42,
    "normalizer": {"mode": "zscore"},
}

MANIFEST_WITH_TIME_SPLIT = {
    "splits": {
        "train": {"symbols": ["BTCUSDT"]},
        "validation": {"symbols": ["SOLUSDT"]},
        "test": {"symbols": ["BTCUSDT"]},
        "time_split": {"train_end": "2024-11-30"},
    }
}


def _window(end_ms):
    return {
        "features": torch.randn(512, 15).numpy(),
        "feature_mask": torch.ones(512, 15, dtype=torch.bool).numpy(),
        "timestamps": torch.arange(end_ms - 512 * 60000, end_ms, 60000).numpy(),
        "mask": torch.ones(512, dtype=torch.bool).numpy(),
        "metadata": {
            "symbol": "BTCUSDT",
            "window_start_ms": end_ms - 512 * 60000,
            "window_end_ms": end_ms,
            "market": "futures",
        },
    }


class TestManifestSplits:
    def test_manifest_splits_disjoint(self):
        with open(MANIFEST_PATH) as f:
            splits = json.load(f)["training_manifest"]["splits"]
        train = set(splits["train"]["symbols"])
        val = set(splits["validation"]["symbols"])
        test = set(splits["test"]["symbols"])
        assert train
        assert val
        assert test
        assert not (train & val), "train and validation must not share symbols"
        assert not (test & val), "test and validation must not share symbols"

    def test_manifest_test_split_is_temporal_holdout(self):
        with open(MANIFEST_PATH) as f:
            splits = json.load(f)["training_manifest"]["splits"]
        time_split = splits.get("time_split") or {}
        assert "train_end" in time_split, "a train_end must be defined for the time split"
        train_end_ms = _date_to_ms(time_split["train_end"])
        # A real holdout must leave data after train_end (data runs through 2024-12-31).
        assert train_end_ms < _date_to_ms("2025-01-01")


class TestBuildWindowsForSymbol:
    def test_filters_windows_after_train_end(self):
        train_end_ms = _date_to_ms("2024-11-30")
        windows = [
            _window(train_end_ms - 60000),
            _window(train_end_ms),
            _window(train_end_ms + 60000),
        ]
        mock_df = MagicMock()
        mock_df.empty = False
        with (
            patch("src.training.trainer.lake") as mock_lake,
            patch("src.training.trainer.feature_builder") as mock_fb,
            patch("src.training.trainer.WindowingEngine") as mock_we,
        ):
            mock_lake.market_state.return_value = mock_df
            mock_fb.build_features.return_value = (
                torch.randn(3, 15).numpy(),
                torch.ones(3, 15, dtype=torch.bool).numpy(),
                [0, 1, 2],
            )
            mock_engine = MagicMock()
            mock_engine.create_windows.return_value = windows
            mock_we.return_value = mock_engine
            ds = _build_windows_for_symbol("BTCUSDT", "futures", stride=1, max_end_ms=train_end_ms)
        assert len(ds.windows) == 1
        assert ds.windows[0]["metadata"]["window_end_ms"] == train_end_ms - 60000


class TestEvalSplitTimeBounds:
    def test_eval_build_split_dataset_applies_time_split(self):
        from src.evaluation.embedding._common import build_split_dataset

        trainer_cfg = {
            "market": "futures",
            "train_window_stride": 8,
            "val_window_stride": 4,
            "seed": 42,
            "feature_style": "raw",
        }
        with (
            patch("src.evaluation.embedding._common._load_manifest", return_value=MANIFEST_WITH_TIME_SPLIT),
            patch("src.evaluation.embedding._common._build_windows_for_symbol") as mock_build,
        ):
            mock_build.return_value = MarketDataset([])
            build_split_dataset("train", trainer_cfg)
            build_split_dataset("test", trainer_cfg)
            build_split_dataset("validation", trainer_cfg)

        train_end_ms = _date_to_ms("2024-11-30")
        assert mock_build.call_count == 3  # BTCUSDT(train) + BTCUSDT(test) + SOLUSDT(validation)
        train_calls = [c for c in mock_build.call_args_list if c.kwargs.get("max_end_ms") == train_end_ms]
        test_calls = [c for c in mock_build.call_args_list if c.kwargs.get("min_start_ms") == train_end_ms]
        val_calls = [
            c for c in mock_build.call_args_list
            if c.kwargs.get("max_end_ms") is None and c.kwargs.get("min_start_ms") is None
        ]
        assert len(train_calls) == 1
        assert len(test_calls) == 1
        assert len(val_calls) == 1
        assert val_calls[0].args[0] == "SOLUSDT"
        # Seed/style are threaded so eval subsampling matches training.
        for c in mock_build.call_args_list:
            assert c.kwargs["seed"] == 42
            assert c.kwargs["feature_style"] == "raw"


class TestTrainerUsesTimeSplit:
    def test_trainer_respects_train_end(self):
        fake_windows = [
            {
                "features": torch.randn(512, 15).numpy(),
                "feature_mask": torch.ones(512, 15, dtype=torch.bool).numpy(),
                "timestamps": list(range(512)),
                "mask": torch.ones(512, dtype=torch.bool).numpy(),
                "metadata": {"symbol": "BTCUSDT", "window_end_ms": 0, "market": "futures"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.training.trainer._load_manifest", return_value=MANIFEST_WITH_TIME_SPLIT),
                patch("src.training.trainer._build_windows_for_symbol") as mock_build,
                patch("src.training.trainer.lake") as mock_lake,
                patch("src.training.trainer.feature_builder") as mock_fb,
            ):
                mock_build.return_value = MarketDataset(fake_windows)
                mock_df = MagicMock()
                mock_df.empty = False
                mock_lake.market_state.return_value = mock_df
                mock_fb.build_features.return_value = (
                    torch.randn(1000, 15).numpy(),
                    torch.ones(1000, 15, dtype=torch.bool).numpy(),
                    [0] * 1000,
                )
                trainer = TeacherTrainer(MODEL_CFG, OPT_CFG, TRAINER_CFG, Path(tmp))

            train_end_ms = _date_to_ms("2024-11-30")
            assert trainer.train_end_ms == train_end_ms
            assert trainer.test_symbols == ["BTCUSDT"]

            # Train split window building is capped at train_end; validation (unseen symbol) is not.
            train_call = next(c for c in mock_build.call_args_list if c.args[0] == "BTCUSDT")
            val_call = next(c for c in mock_build.call_args_list if c.args[0] == "SOLUSDT")
            assert train_call.kwargs["max_end_ms"] == train_end_ms
            assert val_call.kwargs.get("max_end_ms") is None

            # Normalizer fit must restrict the training period to avoid test-period leakage.
            end_ts_calls = [
                call.kwargs.get("end_ts")
                for call in mock_lake.market_state.call_args_list
                if "end_ts" in call.kwargs
            ]
            assert end_ts_calls and all(ts == train_end_ms for ts in end_ts_calls)
