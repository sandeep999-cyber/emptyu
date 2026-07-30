"""Logical Data Lake Engine providing streaming market_state views."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd
from src.config import config
from src.data.alignment import alignment_engine


class DataLake:
    """Virtual Data Lake query engine with SQL filtering pushdown."""

    def __init__(self, canonical_dir: Optional[Path] = None):
        self.canonical_dir = canonical_dir or config.canonical_dir

    @staticmethod
    def _path_exists_sql(pattern: str, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if glob pattern matches any parquet files."""
        try:
            result = conn.sql(f"SELECT count(*) FROM glob('{pattern}')").fetchone()
            return result is not None and result[0] > 0
        except Exception:
            return False

    def _build_where(self, start_ts: Optional[int], end_ts: Optional[int]) -> str:
        clauses = []
        if start_ts is not None:
            clauses.append(f"timestamp >= {start_ts}")
        if end_ts is not None:
            clauses.append(f"timestamp <= {end_ts}")
        return f"WHERE {' AND '.join(clauses)}" if clauses else ""

    def _query_parquet(
        self,
        conn: duckdb.DuckDBPyConnection,
        pattern: str,
        where_clause: str = "",
        label: str = ""
    ) -> pd.DataFrame:
        """Read parquet files matching pattern, return empty on file-not-found, raise on corrupt."""
        try:
            query = f"SELECT * FROM read_parquet('{pattern}') {where_clause} ORDER BY timestamp ASC"
            return conn.sql(query).df()
        except (duckdb.CatalogException, duckdb.IOException):
            # Pattern matched no files — this is expected when modality is absent
            return pd.DataFrame()
        except Exception as e:
            # A real error: corrupted parquet, permission issue, etc — should propagate
            raise RuntimeError(f"Failed to read {label or pattern}: {type(e).__name__}: {e}") from e

    def market_state(
        self,
        symbol: str,
        market: str = "futures",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Query unified aligned market state view for symbol and timeframe range with SQL filter pushdown."""
        klines_pattern = str(self.canonical_dir / market / symbol / "klines" / interval / "*.parquet").replace("\\", "/")
        funding_pattern = str(self.canonical_dir / market / symbol / "funding" / "*.parquet").replace("\\", "/")
        oi_pattern = str(self.canonical_dir / market / symbol / "open_interest" / "*.parquet").replace("\\", "/")
        cal_pattern = str(self.canonical_dir / market / symbol / "metadata" / "*.parquet").replace("\\", "/")

        conn = duckdb.connect(":memory:")
        try:
            where_str = self._build_where(start_ts, end_ts)
            # Add a lookback window for funding and OI to ensure forward-fill validity
            lookback_start = max(0, start_ts - 8 * 60 * 60 * 1000) if start_ts is not None else None
            lookback_where = self._build_where(lookback_start, end_ts)

            df_klines = self._query_parquet(conn, klines_pattern, where_str, "klines")

            if df_klines.empty:
                return pd.DataFrame()

            df_funding = self._query_parquet(conn, funding_pattern, lookback_where, "funding")
            df_oi = self._query_parquet(conn, oi_pattern, lookback_where, "open_interest")
            df_cal = self._query_parquet(conn, cal_pattern, where_str, "calendar")

            df_state = alignment_engine.align_symbol_data(
                symbol=symbol,
                df_klines=df_klines,
                df_funding=df_funding,
                df_open_interest=df_oi,
                df_calendar=df_cal,
            )

            return df_state.reset_index(drop=True)
        finally:
            conn.close()


lake = DataLake()
