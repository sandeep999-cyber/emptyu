"""Unit tests for PyTorch MarketDataset."""

import numpy as np
import torch
from src.data.market_dataset import MarketDataset


def test_market_dataset():
    sample_win = {
        "features": np.zeros((512, 15), dtype=np.float32),
        "feature_mask": np.ones((512, 15), dtype=bool),
        "timestamps": np.arange(512, dtype=np.int64),
        "mask": np.ones((512,), dtype=bool),
        "metadata": {"symbol": "BTCUSDT"}
    }

    ds = MarketDataset([sample_win])
    assert len(ds) == 1

    sample = ds[0]
    assert isinstance(sample["features"], torch.Tensor)
    assert isinstance(sample["feature_mask"], torch.Tensor)
    assert isinstance(sample["timestamps"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)

    assert sample["features"].shape == (512, 15)
    assert sample["feature_mask"].dtype == torch.bool
    assert sample["mask"].dtype == torch.bool
    assert sample["metadata"]["symbol"] == "BTCUSDT"
