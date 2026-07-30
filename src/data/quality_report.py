"""Quality Report Generator creating quality_report.json (§9a)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from src.config import config
from src.data.db import db_manager


class QualityReportGenerator:
    """Generates quality_report.json for dataset snapshots."""

    def __init__(self, canonical_dir: Path | None = None):
        self.canonical_dir = canonical_dir or config.canonical_dir
        self.gap_threshold_ms = config.validation.get("max_timestamp_gap_seconds", 300) * 1000

    def generate_report(
        self,
        symbol: str,
        market: str = "futures",
        df_aligned: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "symbol": symbol,
            "market": market,
            "total_records": 0,
            "missing_values_per_modality": {},
            "forward_fill_percentage": {},
            "alignment_coverage": {},
            "duplicate_rows": 0,
            "gap_count": 0,
            "largest_gap_ms": 0,
            "symbols_with_incomplete_history": [],
            "resampling_statistics": {},
            "quality_score": 100.0,
        }

        if df_aligned is None or df_aligned.empty:
            report["quality_score"] = 0.0
            return report

        report["total_records"] = len(df_aligned)
        report["duplicate_rows"] = int(df_aligned.duplicated().sum())

        if "timestamp" in df_aligned.columns:
            ts_diff = df_aligned["timestamp"].sort_values().diff()
            report["gap_count"] = int((ts_diff > 60000).sum())
            report["largest_gap_ms"] = int(ts_diff.max()) if not ts_diff.empty else 0

        for col in ["open", "high", "low", "close", "volume", "funding_rate", "open_interest"]:
            if col in df_aligned.columns:
                total = len(df_aligned)
                missing = int(df_aligned[col].isna().sum())
                report["missing_values_per_modality"][col] = missing

                if total > 0:
                    report["forward_fill_percentage"][col] = round((1 - missing / total) * 100, 2)
                else:
                    report["forward_fill_percentage"][col] = 0.0

                if col in ("funding_rate", "open_interest"):
                    observed = total - missing
                    report["alignment_coverage"][col] = round(observed / total, 4) if total > 0 else 0.0

        files = db_manager.query_files(symbol=symbol, market=market, dataset_type="klines")
        for f in files:
            if f.get("status") == "RESAMPLED_INCOMPLETE":
                report["symbols_with_incomplete_history"].append({
                    "symbol": symbol,
                    "interval": f.get("interval"),
                    "year": f.get("year"),
                    "month": f.get("month"),
                })

        resample_files = db_manager.query_files(
            symbol=symbol, market=market, dataset_type="klines",
        )
        resampled = [f for f in resample_files if f.get("interval") != "1m"]
        for f in resampled:
            interval = f.get("interval", "unknown")
            report["resampling_statistics"][f"{interval}_{f.get('year')}_{f.get('month')}"] = {
                "rows": f.get("row_count", 0),
                "status": f.get("status", "UNKNOWN"),
            }

        gap_penalty = min(report["gap_count"] * 1.0, 30.0)
        dup_penalty = min(report["duplicate_rows"] * 5.0, 20.0)
        missing_penalty = min(sum(report["missing_values_per_modality"].values()) * 0.5, 20.0)
        report["quality_score"] = max(0.0, 100.0 - gap_penalty - dup_penalty - missing_penalty)

        return report

    def save_report(self, report: Dict[str, Any], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


quality_reporter = QualityReportGenerator()
