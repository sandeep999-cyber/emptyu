"""Feature Builder fusing active modalities into 15-dimensional feature vectors."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import config


CALENDAR_COLUMNS = [
    "minute_of_day",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "is_weekend",
]


class FeatureBuilder:
    """Fuses active modalities post-alignment into continuous feature vectors.

    Two styles are supported:
      - "raw": price levels (OHLC) + volume + funding/OI + calendar.
      - "returns": stationary market-only features (log returns, ranges, log volume,
        log OI) + funding + calendar. Columns 7..14 (calendar) are kept as model
        *input* only; the reconstruction target excludes them (see loss config
        ``reconstruct_calendar: false``).
    """

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

    def build_features(
        self,
        df_aligned: pd.DataFrame,
        style: str = "raw",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a feature matrix in the requested style.

        Args:
            df_aligned: aligned market-state DataFrame.
            style: "raw" (price levels) or "returns" (stationary market-only).

        Returns:
            features: np.ndarray [num_records, feature_dimension] (float32)
            feature_mask: np.ndarray [num_records, feature_dimension] (bool)
            timestamps: np.ndarray [num_records] (int64)
        """
        if style == "returns":
            return self._build_return_features(df_aligned)
        return self._build_raw_features(df_aligned)

    def _build_raw_features(self, df_aligned: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert aligned DataFrame into feature matrix, feature_mask matrix, and timestamp array.
        """
        if df_aligned.empty:
            return self._empty()

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

        # Stale forward-filled funding/OI values are flagged by alignment; they
        # remain model *input* context but must not be reconstruction targets.
        for col in ("funding_rate", "open_interest"):
            stale_col = f"{col}_stale"
            if col in df_aligned.columns and stale_col in df_aligned.columns:
                idx = self.feature_order.index(col)
                stale = df_aligned[stale_col].to_numpy(dtype=bool)
                feature_mask[:, idx] = feature_mask[:, idx] & ~stale

        timestamps = df_aligned["timestamp"].to_numpy(dtype=np.int64)
        return features, feature_mask, timestamps

    def _build_return_features(self, df_aligned: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stationary market-only features in the same 15-dim layout.

        Columns:
            0 log_return        log(close[t] / close[t-1])
            1 hl_range          (high - low) / close
            2 oc_body           (close - open) / close
            3 log_volume        log1p(volume)
            4 volume_change     log1p(volume[t]) - log1p(volume[t-1])
            5 funding_rate      as-is
            6 open_interest     log1p(open_interest)
            7..14 calendar      as-is (input only)
        """
        if df_aligned.empty:
            return self._empty()

        num_records = len(df_aligned)
        features = np.zeros((num_records, self.feature_dimension), dtype=np.float32)
        feature_mask = np.zeros((num_records, self.feature_dimension), dtype=bool)
        timestamps = df_aligned["timestamp"].to_numpy(dtype=np.int64)

        def _col(name: str) -> np.ndarray:
            if name in df_aligned.columns:
                return df_aligned[name].to_numpy(dtype=np.float64, copy=True)
            return np.full(num_records, np.nan, dtype=np.float64)

        open_ = _col("open")
        high = _col("high")
        low = _col("low")
        close = _col("close")
        volume = _col("volume")
        funding = _col("funding_rate")
        oi = _col("open_interest")

        with np.errstate(divide="ignore", invalid="ignore"):
            prev_close = np.concatenate([[np.nan], close[:-1]])
            log_ret = np.log(close / prev_close)
            hl_range = (high - low) / close
            oc_body = (close - open_) / close
            log_vol = np.log1p(np.maximum(volume, 0.0))
            log_oi = np.log1p(np.maximum(oi, 0.0))
            prev_log_vol = np.concatenate([[np.nan], log_vol[:-1]])
            vol_change = log_vol - prev_log_vol

        market_cols = [log_ret, hl_range, oc_body, log_vol, vol_change, funding, log_oi]
        for i, col in enumerate(market_cols):
            obs = ~np.isnan(col)
            vals = np.where(obs, col, 0.0).astype(np.float32)
            features[:, i] = vals
            feature_mask[:, i] = obs

        # Stale forward-filled funding/OI (see _build_raw_features) must not be
        # reconstruction targets, though the value remains model input context.
        for col, idx in (("funding_rate", 5), ("open_interest", 6)):
            stale_col = f"{col}_stale"
            if stale_col in df_aligned.columns:
                stale = df_aligned[stale_col].to_numpy(dtype=bool)
                feature_mask[:, idx] = feature_mask[:, idx] & ~stale

        for j, name in enumerate(CALENDAR_COLUMNS):
            idx = 7 + j
            if name in df_aligned.columns:
                vals = df_aligned[name].to_numpy(dtype=np.float32)
                obs = ~np.isnan(vals)
                feature_mask[:, idx] = obs
                features[:, idx] = np.nan_to_num(vals)

        return features, feature_mask, timestamps

    def _empty(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, self.feature_dimension), dtype=np.float32),
            np.empty((0, self.feature_dimension), dtype=bool),
            np.empty((0,), dtype=np.int64),
        )


feature_builder = FeatureBuilder()
