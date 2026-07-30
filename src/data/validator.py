"""Data integrity auditor and declarative modality validator."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from src.config import config


class DataValidator:
    """Audits data integrity and enforces modality validation rules from validation.yaml."""

    def __init__(self, validation_config: Dict[str, Any] | None = None):
        cfg = validation_config or config.validation
        self.max_gap_seconds = cfg.get("max_timestamp_gap_seconds", 300)
        self.allow_duplicates = cfg.get("allow_duplicate_timestamps", False)
        self.modality_rules = cfg.get("modalities", {})

    def verify_sha256(self, file_path: Path, expected_hash: str) -> bool:
        if not file_path.exists():
            return False
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        clean_expected = expected_hash.replace("sha256:", "").strip()
        return actual_hash.lower() == clean_expected.lower()

    def _check_timestamp_rules(self, df: pd.DataFrame, label: str) -> List[str]:
        errors = []
        ts_col = "timestamp"
        if ts_col in df.columns:
            if not df[ts_col].is_monotonic_increasing:
                errors.append(f"{label} timestamps are not monotonically increasing")
            if not self.allow_duplicates and df[ts_col].duplicated().any():
                errors.append(f"{label} duplicate timestamps detected")
            gaps = df[ts_col].diff().dropna()
            if len(gaps) > 0:
                big_gaps = (gaps > self.max_gap_seconds * 1000).sum()
                if big_gaps > 0:
                    errors.append(f"{label} found {big_gaps} gap(s) exceeding {self.max_gap_seconds}s")
        return errors

    def validate_klines(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["Empty DataFrame"]

        errors.extend(self._check_timestamp_rules(df, "klines"))

        rules = self.modality_rules.get("klines", {})
        if rules.get("high_gte_low", True) and "high" in df.columns and "low" in df.columns:
            bad = (df["high"] < df["low"]).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows where high < low")

        if rules.get("close_positive", True) and "close" in df.columns:
            bad = (df["close"] <= 0).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows where close <= 0")

        return (len(errors) == 0), errors

    def validate_funding(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["Empty DataFrame"]

        errors.extend(self._check_timestamp_rules(df, "funding"))

        rules = self.modality_rules.get("funding", {})
        if not rules.get("allow_missing", True):
            ts_col = "timestamp" if "timestamp" in df.columns else "settlement_time"
            val_col = "funding_rate" if "funding_rate" in df.columns else "last_funding_rate"
            if val_col in df.columns:
                bad = df[val_col].isna().sum()
                if bad > 0:
                    errors.append(f"funding has {bad} missing values (allow_missing=false)")

        return (len(errors) == 0), errors

    def validate_agg_trades(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["Empty DataFrame"]

        errors.extend(self._check_timestamp_rules(df, "aggTrades"))

        rules = self.modality_rules.get("aggTrades", {})
        if rules.get("trade_count_gte_zero", True) and "trade_count" in df.columns:
            bad = (df["trade_count"] < 0).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows where trade_count < 0")

        if rules.get("volume_gte_zero", True) and "base_volume" in df.columns:
            bad = (df["base_volume"] < 0).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows where base_volume < 0")

        return (len(errors) == 0), errors

    def validate_depth(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["Empty DataFrame"]

        errors.extend(self._check_timestamp_rules(df, "depth"))

        rules = self.modality_rules.get("depth", {})
        if rules.get("bid_lte_ask", True):
            if "best_bid" in df.columns and "best_ask" in df.columns:
                bad = (df["best_bid"] > df["best_ask"]).sum()
                if bad > 0:
                    errors.append(f"Found {bad} rows where best_bid > best_ask")
            if rules.get("snapshot_complete", True):
                required = ["best_bid", "best_ask"]
                missing_cols = [c for c in required if c not in df.columns]
                if missing_cols:
                    errors.append(f"Depth snapshot missing columns: {missing_cols}")

        return (len(errors) == 0), errors

    def validate_liquidations(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["Empty DataFrame"]

        errors.extend(self._check_timestamp_rules(df, "liquidations"))

        rules = self.modality_rules.get("liquidations", {})
        if rules.get("valid_side", True) and "side" in df.columns:
            bad = (~df["side"].isin(["BUY", "SELL"])).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows with invalid side (not BUY/SELL)")

        if rules.get("quantity_gt_zero", True) and "quantity" in df.columns:
            bad = (df["quantity"] <= 0).sum()
            if bad > 0:
                errors.append(f"Found {bad} rows where quantity <= 0")

        return (len(errors) == 0), errors


validator = DataValidator()
