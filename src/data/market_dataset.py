"""PyTorch-native MarketDataset interface for foundation model training."""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    """PyTorch-native dataset exposing aligned, windowed market state tensors."""

    def __init__(self, windows: List[Dict[str, Any]]):
        """
        windows: List of window dictionaries produced by WindowingEngine.
        Each window dict contains:
            - "features": np.ndarray [seq_len, feature_dim]
            - "feature_mask": np.ndarray [seq_len, feature_dim]
            - "timestamps": np.ndarray [seq_len]
            - "mask": np.ndarray [seq_len]
            - "metadata": dict
        """
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        return {
            "features": torch.from_numpy(win["features"]).float(),
            "feature_mask": torch.from_numpy(win["feature_mask"]).bool(),
            "timestamps": torch.from_numpy(win["timestamps"]).long(),
            "mask": torch.from_numpy(win["mask"]).bool(),
            "metadata": win.get("metadata", {})
        }
