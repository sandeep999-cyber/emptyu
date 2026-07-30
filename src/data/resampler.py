"""Resampler engine converting 1m kline Parquets to 5m, 15m, 1h, 4h, 1d."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from src.config import config
from src.data.db import db_manager


class KlineResampler:
    """Resamples 1m klines into 5m, 15m, 1h, 4h, 1d canonical Parquets."""

    EXPECTED_1M_PER_PERIOD = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    INTERVAL_MS = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        canonical_dir: Optional[Path] = None,
        db_manager_override: Any = None,
    ):
        self.canonical_dir = canonical_dir or config.canonical_dir
        self.compression = config.storage.get("parquet_compression", "snappy")
        self.warn_incomplete = True
        self._db = db_manager_override or db_manager

    def compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def resample_file(
        self,
        input_1m_path: Path,
        market: str,
        symbol: str,
        target_interval: str,
        year: int,
        month: int,
        download_date: str = "2026-07-30",
        alignment_version: str = "alignment_v1.yaml",
    ) -> Path:
        """Resample a 1m Parquet file into target_interval using DuckDB SQL."""
        if target_interval not in self.INTERVAL_MS:
            raise ValueError(f"Unsupported target interval: {target_interval}")

        interval_ms = self.INTERVAL_MS[target_interval]
        expected_minutes = self.EXPECTED_1M_PER_PERIOD[target_interval]
        out_dir = self.canonical_dir / market / symbol / "klines" / target_interval
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year}-{month:02d}.parquet"

        conn = duckdb.connect(":memory:")
        try:
            # Aggregate 1m klines into target_interval using FLOOR integer bucket
            bucket_open = f"CAST(FLOOR(CAST(timestamp AS DOUBLE) / {interval_ms}) * {interval_ms} AS BIGINT)"
            query = f"""
                SELECT
                    {bucket_open} AS timestamp,
                    {bucket_open} + {interval_ms} - 1 AS close_time,
                    FIRST(open ORDER BY timestamp ASC) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close ORDER BY timestamp ASC) AS close,
                    SUM(volume) AS volume,
                    SUM(quote_volume) AS quote_volume,
                    SUM(count) AS count,
                    SUM(taker_buy_volume) AS taker_buy_volume,
                    SUM(taker_buy_quote_volume) AS taker_buy_quote_volume,
                    COUNT(*) AS constituent_count
                FROM read_parquet('{str(input_1m_path).replace("\\", "/")}')
                GROUP BY 1
                ORDER BY timestamp ASC
            """
            result = conn.execute(query)
            arrow_table = result.to_arrow_table()

            # Embed provenance and completeness metadata
            existing_meta = arrow_table.schema.metadata or {}
            source_hash = self.compute_sha256(input_1m_path)
            provenance = {
                "provenance_created_by": "resampler.py",
                "provenance_source_file": str(input_1m_path),
                "provenance_source_checksum": f"sha256:{source_hash}",
                "provenance_target_interval": target_interval,
                "provenance_download_date": download_date,
                "provenance_alignment_version": alignment_version,
                "provenance_schema_version": "canonical_schema_v1",
            }
            merged_meta = {**{k.encode("utf-8"): str(v).encode("utf-8") for k, v in provenance.items()}, **existing_meta}
            arrow_table = arrow_table.replace_schema_metadata(merged_meta)

            pq.write_table(arrow_table, str(out_path), compression=self.compression)

            start_ts = int(arrow_table["timestamp"][0].as_py()) if len(arrow_table) > 0 else 0
            end_ts = int(arrow_table["timestamp"][-1].as_py()) if len(arrow_table) > 0 else 0
            row_count = len(arrow_table)
            file_size = out_path.stat().st_size
            sha256_hash = self.compute_sha256(out_path)

            # Check for incomplete candles
            if row_count > 0 and "constituent_count" in arrow_table.column_names:
                cc = arrow_table.column("constituent_count").to_pylist()
                incomplete = [i for i, c in enumerate(cc) if c < expected_minutes]
                if incomplete and self.warn_incomplete:
                    import warnings
                    warnings.warn(
                        f"{len(incomplete)}/{row_count} candle(s) in {target_interval} {year}-{month:02d} "
                        f"for {symbol} have fewer than {expected_minutes} 1m constituents "
                        f"(indices: {incomplete[:10]}). Candles emitted but flagged."
                    )

            file_id = f"{market}_{symbol}_klines_{target_interval}_{year}_{month}"
            file_status = "RESAMPLED"
            if row_count > 0 and "constituent_count" in arrow_table.column_names:
                # Detect if any candle has too few constituents
                incomplete_count = sum(1 for c in cc if c < expected_minutes)
                if incomplete_count:
                    file_status = "RESAMPLED_INCOMPLETE"

            self._db.register_file(
                file_id=file_id,
                symbol=symbol,
                market=market,
                dataset_type="klines",
                interval=target_interval,
                year=year,
                month=month,
                start_ts=start_ts,
                end_ts=end_ts,
                row_count=row_count,
                file_size=file_size,
                sha256=sha256_hash,
                schema_hash=hashlib.md5(str(arrow_table.schema).encode("utf-8")).hexdigest(),
                file_path=str(out_path),
                status=file_status,
            )

            return out_path
        finally:
            conn.close()


resampler = KlineResampler()
