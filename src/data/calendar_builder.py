"""Calendar matrix generator (calendar_v1.parquet)."""

import hashlib
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.config import config
from src.data.db import db_manager


class CalendarBuilder:
    """Builds calendar_v1.parquet containing temporal vectors."""

    def __init__(
        self,
        canonical_dir: Optional[Path] = None,
        db_manager_override: Any = None,
    ):
        self.canonical_dir = canonical_dir or config.canonical_dir
        self.compression = config.storage.get("parquet_compression", "snappy")
        self._db = db_manager_override or db_manager

    def build_calendar_for_year(self, market: str, symbol: str, year: int) -> Path:
        """Generate 1-minute resolution calendar vector for an entire year."""
        start_dt = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end_dt = pd.Timestamp(f"{year}-12-31 23:59:00", tz="UTC")
        dt_index = pd.date_range(start=start_dt, end=end_dt, freq="1min")

        df = pd.DataFrame(index=dt_index)
        # Use .asi8 for safe tz-aware int64 conversion, then divide by 1e6 for ms
        df["timestamp"] = (dt_index.asi8 // 10**6).astype(np.int64)
        df["minute_of_day"] = dt_index.hour * 60 + dt_index.minute
        df["hour"] = dt_index.hour
        df["day_of_week"] = dt_index.dayofweek
        df["day_of_month"] = dt_index.day
        df["month"] = dt_index.month
        df["quarter"] = dt_index.quarter
        df["year"] = dt_index.year
        df["is_weekend"] = (dt_index.dayofweek >= 5).astype(np.int8)

        out_dir = self.canonical_dir / market / symbol / "metadata"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"calendar_{year}_v1.parquet"

        df_out = df.reset_index(drop=True)
        table = pa.Table.from_pandas(df_out, preserve_index=False)

        # Add deterministic provenance
        row_hash = hashlib.sha256(df_out.to_json().encode("utf-8")).hexdigest()[:16]
        provenance = {
            "provenance_created_by": "calendar_builder.py",
            "provenance_schema_version": "canonical_schema_v1",
            "provenance_source": "deterministic_time_calendar",
            "provenance_year": str(year),
            "provenance_row_hash": row_hash,
        }
        existing_meta = table.schema.metadata or {}
        merged_meta = {**{k.encode("utf-8"): str(v).encode("utf-8") for k, v in provenance.items()}, **existing_meta}
        table = table.replace_schema_metadata(merged_meta)

        pq.write_table(table, str(out_path), compression=self.compression)

        # Register in index
        start_ts = int(df_out["timestamp"].min())
        end_ts = int(df_out["timestamp"].max())
        file_id = f"{market}_{symbol}_calendar_{year}"
        file_size = out_path.stat().st_size
        file_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        self._db.register_file(
            file_id=file_id,
            symbol=symbol,
            market=market,
            dataset_type="calendar",
            interval="1m",
            year=year,
            month=None,
            start_ts=start_ts,
            end_ts=end_ts,
            row_count=len(df_out),
            file_size=file_size,
            sha256=file_sha256,
            schema_hash=hashlib.md5(str(table.schema).encode("utf-8")).hexdigest(),
            file_path=str(out_path),
            status="GENERATED",
        )

        return out_path


calendar_builder = CalendarBuilder()
