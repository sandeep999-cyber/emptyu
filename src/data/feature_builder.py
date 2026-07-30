"""Feature Builder fusing active modalities into 15-dimensional feature vectors."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import config


class FeatureBuilder:
    """Fuses active modalities post-alignment into continuous feature vectors."""

    def __init__(self, schema_dict: Optional[Dict] = None):
        self.schema = schema_dict or config.market_state_schema
        self.version = self.schema.get("feature_builder_version", "v1")
        self.feature_order = self.schema.get("feature_order", [])
        self.feature_dimension = self.schema.get("feature_dimension", 15)
        if len(self.feature_order) != self.feature_dimension:
            raise ValueError(
                f"feature_order length ({len(self.feature_order)}) "
                f"must equal feature_dimension ({self.feature_dimension})"
            )

    def build_features(self, df_aligned: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert aligned DataFrame into feature matrix, feature_mask matrix, and timestamp array.

        Returns:
            features: np.ndarray [num_records, feature_dimension] (float32)
            feature_mask: np.ndarray [num_records, feature_dimension] (bool)
            timestamps: np.ndarray [num_records] (int64)
        """
        if df_aligned.empty:
            return (
                np.empty((0, self.feature_dimension), dtype=np.float32),
                np.empty((0, self.feature_dimension), dtype=bool),
                np.empty((0,), dtype=np.int64)
            )

        num_records = len(df_aligned)
        features = np.zeros((num_records, self.feature_dimension), dtype=np.float32)
        feature_mask = np.zeros((num_records, self.feature_dimension), dtype=bool)

        for i, col in enumerate(self.feature_order):
            if col in df_aligned.columns:
                vals = df_aligned[col].to_numpy(dtype=np.float32, copy=True)
                # NaNs represent missing/unobserved observations
                nan_mask = np.isnan(vals)
                feature_mask[:, i] = ~nan_mask
                vals[nan_mask] = 0.0  # Zero out NaNs for tensor numeric safety ONLY
                features[:, i] = vals
            else:
                features[:, i] = 0.0
                feature_mask[:, i] = False

        timestamps = df_aligned["timestamp"].to_numpy(dtype=np.int64)
        return features, feature_mask, timestamps


feature_builder = FeatureBuilder()
