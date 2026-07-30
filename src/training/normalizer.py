"""Training-layer Feature Normalizer fitting statistics strictly on train split."""

from typing import Any, Dict, Optional
import json
from pathlib import Path
import numpy as np
import torch


class FeatureNormalizer:
    """Normalizes feature tensors using statistics fit exclusively on the train split."""

    def __init__(self, mode: str = "zscore"):
        self.mode = mode
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.median: Optional[torch.Tensor] = None
        self.iqr: Optional[torch.Tensor] = None

    def fit(
        self,
        train_features: torch.Tensor,
        feature_mask: torch.Tensor,
        manifest_splits: Dict[str, Any],
    ) -> None:
        if "train" not in manifest_splits or not manifest_splits["train"].get("symbols"):
            raise ValueError("Normalizer can only be fit on train split symbols!")

        feature_dim = train_features.shape[-1]

        if self.mode == "robust":
            med_vals = torch.zeros(feature_dim, dtype=train_features.dtype)
            iqr_vals = torch.ones(feature_dim, dtype=train_features.dtype)
            for d in range(feature_dim):
                observed = train_features[:, d][feature_mask[:, d]]
                if len(observed) > 1:
                    med_vals[d] = torch.median(observed)
                    q25 = torch.quantile(observed, 0.25)
                    q75 = torch.quantile(observed, 0.75)
                    iqr_vals[d] = q75 - q25
            self.median = med_vals
            self.iqr = torch.where(iqr_vals == 0, torch.tensor(1.0, dtype=iqr_vals.dtype), iqr_vals)
        else:
            mean_vals = torch.zeros(feature_dim, dtype=train_features.dtype)
            std_vals = torch.zeros(feature_dim, dtype=train_features.dtype)
            for d in range(feature_dim):
                observed = train_features[:, d][feature_mask[:, d]]
                if len(observed) > 0:
                    mean_vals[d] = torch.mean(observed)
                    std_vals[d] = torch.std(observed)
            self.mean = mean_vals
            self.std = torch.where(std_vals == 0, torch.tensor(1.0, dtype=std_vals.dtype), std_vals)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "robust":
            if self.median is None or self.iqr is None:
                raise RuntimeError("Normalizer must be fit before calling transform!")
            return (features - self.median) / self.iqr
        else:
            if self.mean is None or self.std is None:
                raise RuntimeError("Normalizer must be fit before calling transform!")
            if self.mode == "zscore":
                return (features - self.mean) / self.std
            elif self.mode == "log":
                return torch.log1p(torch.abs(features))
            return features

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"mode": self.mode}
        if self.mode == "robust":
            if self.median is None:
                raise RuntimeError("Normalizer must be fit before serializing state.")
            state["median"] = self.median.tolist()
            state["iqr"] = self.iqr.tolist()
        else:
            if self.mean is None:
                raise RuntimeError("Normalizer must be fit before serializing state.")
            state["mean"] = self.mean.tolist()
            state["std"] = self.std.tolist()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mode = state.get("mode", "zscore")
        if self.mode == "robust":
            self.median = torch.tensor(state["median"], dtype=torch.float32)
            self.iqr = torch.tensor(state["iqr"], dtype=torch.float32)
        else:
            self.mean = torch.tensor(state["mean"], dtype=torch.float32)
            self.std = torch.tensor(state["std"], dtype=torch.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        normalizer = cls()
        normalizer.load_state_dict(state)
        return normalizer
