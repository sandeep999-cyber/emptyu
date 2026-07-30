"""CSV/ZIP to Canonical Snappy Parquet converter with embedded provenance and robust header detection."""

import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.config import config
from src.data.db import db_manager


# Schema for each dataset_type; used as column names when CSV has no header,
# and also to map real column names to canonical names when header is present.
BINANCE_CSV_SCHEMAS: Dict[str, Dict[str, str]] = {
    "klines": {
        "open_time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "close_time": "close_time",
        "quote_volume": "quote_volume",
        "count": "count",
        "taker_buy_volume": "taker_buy_volume",
        "taker_buy_quote_volume": "taker_buy_quote_volume",
        "ignore": "ignore",
    },
    "aggTrades": {
        "agg_trade_id": "agg_trade_id",
        "price": "price",
        "quantity": "quantity",
        "first_trade_id": "first_trade_id",
        "last_trade_id": "last_trade_id",
        "transact_time": "timestamp",
        "is_buyer_maker": "is_buyer_maker",
    },
    "trades": {
        "trade_id": "trade_id",
        "price":  "price",
        "qty": "qty",
        "quote_qty": "quote_qty",
        "time": "timestamp",
        "is_buyer_maker": "is_buyer_maker",
    },
    "funding": {
        "calc_time": "timestamp",
        "funding_interval_hours": "funding_interval_hours",
        "last_funding_rate": "funding_rate",
    },
    "fundingRate": {
        "calc_time": "timestamp",
        "funding_interval_hours": "funding_interval_hours",
        "last_funding_rate": "funding_rate",
    },
    "open_interest": {
        "create_time": "timestamp",
        "symbol": "base_symbol",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
    },
    "metrics": {
        "create_time": "timestamp",
        "symbol": "base_symbol",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
    },
}


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    """Detect unit (ms / us / string iso) and normalize to epoch ms."""
    sample = series.dropna()
    if sample.empty:
        return series

    # If already numeric (int/float)
    if pd.api.types.is_numeric_dtype(sample):
        vals = sample.astype(float)
        med = vals.median()
        # If median is ~13 digits → epoch ms; ~16 digits → epoch us
        if 1e15 < med < 1e17:
            return (vals / 1000).astype("int64")
        elif 1e17 < med < 1e19:
            return (vals / 1_000_000).astype("int64")
        return vals.astype("int64")

    # Try string-to-datetime parsing
    try:
        dt = pd.to_datetime(sample, utc=True)
        return dt.astype("int64") // 10**6
    except Exception:
        return series


def _schema_columns(dataset_type: str) -> Optional[List[str]]:
    """Return ordered column names for headerless parsing."""
    schema = BINANCE_CSV_SCHEMAS.get(dataset_type)
    if schema is None:
        return None
    # Order: preserve entry order from the schema dict
    return list(schema.keys())


def _remap_columns(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Rename raw Binance columns to canonical names per schema."""
    schema = BINANCE_CSV_SCHEMAS.get(dataset_type)
    if schema is None:
        return df
    rename_map = {k: v for k, v in schema.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


class ParquetConverter:
    """Converts raw Binance CSV/ZIP archives to canonical Snappy Parquet files."""

    def __init__(
        self,
        canonical_dir: Optional[Path] = None,
        db_manager_override: Any = None,
    ):
        self.canonical_dir = canonical_dir or config.canonical_dir
        self.compression = config.storage.get("parquet_compression", "snappy")
        self._db = db_manager_override or db_manager

    def compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def convert_zip_to_parquet(
        self,
        zip_path: Path,
        market: str,
        symbol: str,
        dataset_type: str,
        interval: str = "1m",
        year: int = 2024,
        month: int = 1,
        snapshot_date: str = "2026-07-30",
        download_date: str = "2026-07-30",
        alignment_version: str = "alignment_v1.yaml",
    ) -> Path:
        """Extract CSV from ZIP and convert to Snappy Parquet with embedded provenance."""
        if not zip_path.exists():
            raise FileNotFoundError(f"Archive not found: {zip_path}")

        # Deterministic hash of source (no timestamp dependency)
        raw_sha256 = self.compute_sha256(zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                first_line = csv_file.readline().decode("utf-8")
                csv_file.seek(0)

                # Robust header detection
                has_header = not (first_line[0].isdigit() or first_line.startswith("-") or first_line.startswith("+"))
                cols = _schema_columns(dataset_type)

                if not has_header and cols:
                    df = pd.read_csv(csv_file, names=cols, dtype_backend="numpy_nullable")
                else:
                    df = pd.read_csv(csv_file, dtype_backend="numpy_nullable")

        # Remap raw columns to canonical names
        df = _remap_columns(df, dataset_type)

        # If no 'timestamp' column yet, try to rename from known timestamp aliases
        if "timestamp" not in df.columns:
            for alias in ["open_time", "openTime", "transact_time", "time", "calc_time", "fundingTime", "settlement_time"]:
                if alias in df.columns:
                    df = df.rename(columns={alias: "timestamp"})
                    break

        # Normalize timestamp to epoch ms
        if "timestamp" in df.columns:
            before = len(df)
            df["timestamp"] = _normalize_timestamps(df["timestamp"])
            df.dropna(subset=["timestamp"], inplace=True)
            dropped = before - len(df)
            if dropped > 0:
                warnings.warn(
                    f"Dropped {dropped}/{before} rows with unparseable timestamps "
                    f"in {zip_path.name}"
                )
            df["timestamp"] = df["timestamp"].astype("int64")
            df.sort_values("timestamp", inplace=True)

        start_ts = int(df["timestamp"].min()) if "timestamp" in df.columns and not df.empty else 0
        end_ts = int(df["timestamp"].max()) if "timestamp" in df.columns and not df.empty else 0
        row_count = len(df)

        if dataset_type == "klines":
            out_dir = self.canonical_dir / market / symbol / dataset_type / interval
        else:
            out_dir = self.canonical_dir / market / symbol / dataset_type

        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{year}-{month:02d}.parquet"
        out_path = out_dir / filename

        table = pa.Table.from_pandas(df, preserve_index=False)

        # Deterministic provenance (no datetime.now() — uses source hash only)
        provenance = {
            "provenance_created_by": "parquet_converter.py",
            "provenance_source": "binance_vision_archive",
            "provenance_source_checksum": f"sha256:{raw_sha256}",
            "provenance_download_date": download_date,
            "provenance_converter_version": "v1",
            "provenance_alignment_version": alignment_version,
            "provenance_schema_version": "canonical_schema_v1",
            "provenance_snapshot": snapshot_date,
        }

        existing_meta = table.schema.metadata or {}
        merged_meta = {**{k.encode("utf-8"): str(v).encode("utf-8") for k, v in provenance.items()}, **existing_meta}
        table = table.replace_schema_metadata(merged_meta)

        pq.write_table(table, str(out_path), compression=self.compression)
        out_file_size = out_path.stat().st_size
        out_sha256 = self.compute_sha256(out_path)

        file_id = f"{market}_{symbol}_{dataset_type}_{interval}_{year}_{month}"
        self._db.register_file(
            file_id=file_id,
            symbol=symbol,
            market=market,
            dataset_type=dataset_type,
            interval=interval,
            year=year,
            month=month,
            start_ts=start_ts,
            end_ts=end_ts,
            row_count=row_count,
            file_size=out_file_size,
            sha256=out_sha256,
            schema_hash=hashlib.md5(str(table.schema).encode("utf-8")).hexdigest(),
            file_path=str(out_path),
            status="CONVERTED"
        )

        return out_path


converter = ParquetConverter()
