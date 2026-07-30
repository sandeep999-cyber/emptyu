"""Unit tests for Feature Normalizer including robust mode."""

import tempfile
from pathlib import Path
import pytest
import torch
from src.training.normalizer import FeatureNormalizer


class TestZscoreNormalizer:
    def test_train_split_only(self):
        normalizer = FeatureNormalizer()
        feats = torch.randn(100, 15)
        fm = torch.ones(100, 15, dtype=torch.bool)
        with pytest.raises(ValueError, match="Normalizer can only be fit on train split symbols"):
            normalizer.fit(feats, fm, {"validation": {"symbols": ["SOLUSDT"]}})
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT", "ETHUSDT"]}})
        transformed = normalizer.transform(feats)
        assert transformed.shape == (100, 15)

    def test_save_load(self):
        normalizer = FeatureNormalizer()
        feats = torch.randn(100, 15)
        fm = torch.ones(100, 15, dtype=torch.bool)
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        original_mean = normalizer.mean.clone()
        original_std = normalizer.std.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer.json"
            normalizer.save(path)
            loaded = FeatureNormalizer.load(path)
            assert torch.allclose(loaded.mean, original_mean)
            assert torch.allclose(loaded.std, original_std)
            assert loaded.mode == "zscore"
        assert loaded.transform(feats).shape == feats.shape

    def test_mask_aware(self):
        normalizer = FeatureNormalizer()
        feats = torch.tensor([[10.0, 100.0], [0.0, 200.0], [20.0, 300.0]])
        fm = torch.tensor([[True, True], [False, True], [True, True]])
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        assert torch.isclose(normalizer.mean[0], torch.tensor(15.0))


class TestRobustNormalizer:
    def test_robust_fit_transform(self):
        normalizer = FeatureNormalizer(mode="robust")
        feats = torch.randn(100, 15)
        fm = torch.ones(100, 15, dtype=torch.bool)
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        assert normalizer.median is not None
        assert normalizer.iqr is not None
        transformed = normalizer.transform(feats)
        assert transformed.shape == (100, 15)

    def test_robust_save_load(self):
        normalizer = FeatureNormalizer(mode="robust")
        feats = torch.randn(100, 15)
        fm = torch.ones(100, 15, dtype=torch.bool)
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        orig_median = normalizer.median.clone()
        orig_iqr = normalizer.iqr.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer_robust.json"
            normalizer.save(path)
            loaded = FeatureNormalizer.load(path)
            assert loaded.mode == "robust"
            assert torch.allclose(loaded.median, orig_median)
            assert torch.allclose(loaded.iqr, orig_iqr)

    def test_robust_mask_aware(self):
        normalizer = FeatureNormalizer(mode="robust")
        feats = torch.tensor([[10.0], [0.0], [20.0], [30.0]])
        fm = torch.tensor([[True], [False], [True], [True]])
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        assert torch.isclose(normalizer.median[0], torch.tensor(20.0))

    def test_robust_rejects_invalid_split(self):
        normalizer = FeatureNormalizer(mode="robust")
        with pytest.raises(ValueError, match="train split"):
            normalizer.fit(torch.randn(10, 5), torch.ones(10, 5, dtype=torch.bool), {})

    def test_log_mode(self):
        normalizer = FeatureNormalizer(mode="log")
        feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fm = torch.ones(2, 2, dtype=torch.bool)
        normalizer.fit(feats, fm, {"train": {"symbols": ["BTCUSDT"]}})
        result = normalizer.transform(feats)
        assert result.shape == (2, 2)

    def test_unfit_raises(self):
        normalizer = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="must be fit"):
            normalizer.transform(torch.randn(10, 15))
