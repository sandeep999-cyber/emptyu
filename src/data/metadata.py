"""Metadata manager for dataset_version.json, statistics_v1.json, and schemas."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.config import config


class MetadataManager:
    """Manages versioned metadata files."""

    def create_dataset_version(
        self,
        version: str = "1.0.0",
        binance_snapshot: str = "2026-07-30",
        schema_version: int = 1,
        created: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate dataset_version.json dictionary. Pass created manually for reproducibility."""
        return {
            "version": version,
            "created": created or binance_snapshot,
            "binance_snapshot": binance_snapshot,
            "schema_version": schema_version
        }

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics (min, max, mean, std) for numeric columns."""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = df[col].dropna()
            if not vals.empty:
                stats[col] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if len(vals) > 1 else 0.0
                }
        return stats

    def save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


metadata_manager = MetadataManager()
