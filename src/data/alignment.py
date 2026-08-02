"""Causal Alignment Engine executing policy declared in alignment_v1.yaml."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.config import config


# Column-name candidates for each known_at field in the contract
_KNOWN_AT_COLUMNS: Dict[str, list] = {
    "settlement_time": ["settlement_time", "timestamp", "calc_time"],
    "exchange_event_time": ["timestamp", "create_time", "calc_time", "time", "transact_time"],
}

# Value-column candidates per modality key used in the contract
_VALUE_COLUMNS: Dict[str, list] = {
    "funding_rate": ["funding_rate", "last_funding_rate", "fundingRate"],
    "open_interest": [
        "open_interest", "sum_open_interest", "sumOpenInterest",
        "sum_open_interest_value", "open_interest_value",
    ],
}

_FUTURE_MODALITIES = {"agg_trades", "depth", "liquidations"}


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class AlignmentEngine:
    """Declarative engine that executes policies from alignment_v1.yaml."""

    def __init__(self):
        self.contract = config.alignment

    def _modality_policy(self, modality: str) -> Dict[str, Any]:
        return self.contract.get(modality, {})

    def _missing_policy(self, modality: str) -> Optional[str]:
        return self.contract.get("missing", {}).get(modality, {}).get("policy")

    def _apply_known_at_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift kline timestamps to close_time (known_at semantics)."""
        if "close_time" in df.columns:
            df["timestamp"] = df["close_time"].astype("int64")
        else:
            ts_diffs = df["timestamp"].sort_values().diff().dropna()
            median_gap = int(ts_diffs.median()) if not ts_diffs.empty else 60000
            df["timestamp"] = df["timestamp"] + max(median_gap, 60000) - 1
        return df

    def _align_forward_fill(
        self, df_aligned: pd.DataFrame, df_modality: Optional[pd.DataFrame],
        value_key: str, policy: Dict[str, Any],
    ) -> pd.DataFrame:
        """ASOF-merge a sub-minute modality backward, then forward-fill.

        Forward-filled values that have gone stale (older than the modality's
        declared ``frequency``) are flagged in a ``<value_key>_stale`` boolean
        column so downstream consumers (feature_builder) can mark them
        unobserved in the feature mask. The carried value itself is retained
        as model input context.
        """
        if df_modality is None or df_modality.empty:
            df_aligned[value_key] = np.nan
            df_aligned[f"{value_key}_stale"] = False
            return df_aligned

        df_m = df_modality.copy()

        known_at = policy.get("known_at", "timestamp")
        ts_cols = _KNOWN_AT_COLUMNS.get(known_at, [known_at])
        ts_col = _find_column(df_m, *ts_cols)

        val_cols = _VALUE_COLUMNS.get(value_key, [value_key])
        val_col = _find_column(df_m, *val_cols)

        if ts_col is None or val_col is None:
            raise KeyError(
                f"df_{value_key} missing required columns. Found: {list(df_m.columns)}. "
                f"Needed timestamp column in {ts_cols} and value column in {val_cols}"
            )

        df_m = df_m[[ts_col, val_col]].rename(columns={ts_col: "timestamp", val_col: value_key})
        df_m["timestamp"] = pd.to_numeric(df_m["timestamp"], errors="coerce")
        df_m[value_key] = pd.to_numeric(df_m[value_key], errors="coerce")
        df_m.dropna(subset=["timestamp"], inplace=True)
        df_m.sort_values("timestamp", inplace=True)

        # Track the observation timestamp of the asof-matched source row so the
        # age of each forward-filled value can be computed.
        df_m["_src_ts"] = df_m["timestamp"]

        df_aligned = pd.merge_asof(df_aligned, df_m, on="timestamp", direction="backward")
        # Age since the real (non-carried) observation; NaN where no observation yet.
        age_ms = (df_aligned["timestamp"] - df_aligned["_src_ts"]).astype("float64")
        max_stale_ms = self._parse_frequency_ms(policy.get("frequency"))
        stale = age_ms.isna() | (age_ms > max_stale_ms)
        df_aligned[f"{value_key}_stale"] = stale.to_numpy(dtype=bool)
        # Forward-fill values so the latest known value is available as context.
        df_aligned[value_key] = df_aligned[value_key].ffill()
        df_aligned.drop(columns=["_src_ts"], inplace=True)
        return df_aligned

    @staticmethod
    def _parse_frequency_ms(frequency: Optional[str]) -> int:
        """Parse a frequency like '8h', '5m', '30s' into milliseconds."""
        if not frequency:
            return 0
        freq = str(frequency).strip().lower()
        try:
            amount = int(freq.rstrip("hms"))
        except ValueError:
            return 0
        if freq.endswith("h"):
            return amount * 3600 * 1000
        if freq.endswith("m"):
            return amount * 60 * 1000
        if freq.endswith("s"):
            return amount * 1000
        return 0

    def _align_derived_calendar(
        self, df_aligned: pd.DataFrame, df_calendar: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge a pre-built calendar or generate temporal fields inline."""
        if df_calendar is not None and not df_calendar.empty:
            df_cal = df_calendar.copy()
            df_cal.sort_values("timestamp", inplace=True)
            df_aligned = pd.merge_asof(df_aligned, df_cal, on="timestamp", direction="backward")
        else:
            ts = pd.to_datetime(df_aligned["timestamp"], unit="ms", utc=True)
            df_aligned["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
            df_aligned["hour"] = ts.dt.hour
            df_aligned["day_of_week"] = ts.dt.dayofweek
            df_aligned["day_of_month"] = ts.dt.day
            df_aligned["month"] = ts.dt.month
            df_aligned["quarter"] = ts.dt.quarter
            df_aligned["year"] = ts.dt.year
            df_aligned["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        return df_aligned

    def _check_future_modalities(self, **kwargs: Any) -> None:
        for key, val in kwargs.items():
            if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                modality = key.removeprefix("df_")
                raise NotImplementedError(
                    f"Modality '{modality}' is declared in alignment_v1.yaml but not yet active. "
                    f"Implementation lands in Phase 2/3."
                )

    def align_symbol_data(
        self,
        symbol: str,
        df_klines: pd.DataFrame,
        df_funding: Optional[pd.DataFrame] = None,
        df_open_interest: Optional[pd.DataFrame] = None,
        df_calendar: Optional[pd.DataFrame] = None,
        df_agg_trades: Optional[pd.DataFrame] = None,
        df_depth: Optional[pd.DataFrame] = None,
        df_liquidations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Align all modalities onto the klines timeline via contract-declared policies."""
        self._check_future_modalities(
            df_agg_trades=df_agg_trades, df_depth=df_depth, df_liquidations=df_liquidations,
        )

        if df_klines.empty:
            return pd.DataFrame()

        df_aligned = df_klines.copy()
        if "timestamp" not in df_aligned.columns:
            raise KeyError("df_klines missing 'timestamp' column")

        df_aligned.sort_values("timestamp", inplace=True)
        df_aligned.reset_index(drop=True, inplace=True)

        if self.contract.get("shift_to_known_at", False):
            df_aligned = self._apply_known_at_shift(df_aligned)

        funding_policy = self._modality_policy("funding")
        oi_policy = self._modality_policy("open_interest")
        cal_policy = self._modality_policy("calendar")

        if funding_policy.get("alignment") == "forward_fill":
            df_aligned = self._align_forward_fill(
                df_aligned, df_funding, "funding_rate", funding_policy,
            )
        else:
            df_aligned["funding_rate"] = np.nan

        if oi_policy.get("alignment") == "forward_fill":
            df_aligned = self._align_forward_fill(
                df_aligned, df_open_interest, "open_interest", oi_policy,
            )
        else:
            df_aligned["open_interest"] = np.nan

        if cal_policy.get("alignment") == "derived":
            df_aligned = self._align_derived_calendar(df_aligned, df_calendar)

        return df_aligned


alignment_engine = AlignmentEngine()
