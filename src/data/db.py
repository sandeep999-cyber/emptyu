"""DuckDB Database Manager for file index and asset registry."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
from src.config import config


class DatabaseManager:
    """Manages index.duckdb for tracking files, row counts, checksums, and symbol metadata."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.db_path
        self._tables_initialized = False

    def _ensure_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        self._ensure_dir()
        conn = duckdb.connect(str(self.db_path))
        if not self._tables_initialized:
            self._init_tables(conn)
            self._tables_initialized = True
        return conn

    def _init_tables(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize database schema tables once."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index_v1 (
                file_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                market VARCHAR,
                dataset_type VARCHAR,
                interval VARCHAR,
                year INTEGER,
                month INTEGER,
                start_ts BIGINT,
                end_ts BIGINT,
                row_count BIGINT,
                file_size BIGINT,
                sha256 VARCHAR,
                schema_hash VARCHAR,
                file_path VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_symbol
            ON file_index_v1(symbol);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_market_type
            ON file_index_v1(market, dataset_type);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_year_month
            ON file_index_v1(year, month);
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_registry (
                symbol VARCHAR NOT NULL,
                market_type VARCHAR NOT NULL,
                base_asset VARCHAR,
                quote_asset VARCHAR,
                is_active BOOLEAN,
                listing_date VARCHAR,
                delisting_date VARCHAR,
                contract_type VARCHAR,
                tick_size DOUBLE,
                step_size DOUBLE,
                min_qty DOUBLE,
                contract_size DOUBLE,
                valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_to TIMESTAMP
            );
        """)

    def register_file(
        self,
        file_id: str,
        symbol: str,
        market: str,
        dataset_type: str,
        interval: str,
        year: int,
        month: Optional[int] = None,
        start_ts: int = 0,
        end_ts: int = 0,
        row_count: int = 0,
        file_size: int = 0,
        sha256: str = "",
        schema_hash: str = "",
        file_path: str = "",
        status: str = "CONVERTED"
    ) -> None:
        """Register or update a file in file_index_v1."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO file_index_v1 (
                    file_id, symbol, market, dataset_type, interval, year, month,
                    start_ts, end_ts, row_count, file_size, sha256, schema_hash,
                    file_path, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                file_id, symbol, market, dataset_type, interval, year, month,
                start_ts, end_ts, row_count, file_size, sha256, schema_hash,
                file_path, status
            ])
        finally:
            conn.close()

    def register_files_batch(
        self,
        records: List[Tuple]
    ) -> None:
        """Register multiple files in a single transaction. Each record must have exactly 15 elements."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            for i, r in enumerate(records):
                if len(r) != 15:
                    raise ValueError(
                        f"Record {i} has {len(r)} elements; expected 15 "
                        f"(file_id, symbol, market, dataset_type, interval, year, month, "
                        f"start_ts, end_ts, row_count, file_size, sha256, schema_hash, file_path, status)"
                    )
                conn.execute("""
                    INSERT OR REPLACE INTO file_index_v1 (
                        file_id, symbol, market, dataset_type, interval, year, month,
                        start_ts, end_ts, row_count, file_size, sha256, schema_hash,
                        file_path, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, list(r))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def register_asset(
        self,
        symbol: str,
        market_type: str,
        base_asset: str,
        quote_asset: str,
        is_active: bool = True,
        listing_date: Optional[str] = None,
        delisting_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        tick_size: float = 0.0,
        step_size: float = 0.0,
        min_qty: float = 0.0,
        contract_size: float = 1.0
    ) -> None:
        """Register asset in asset_registry, expiring previous versions (append-only)."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            # Expire previous active record for this (symbol, market_type)
            conn.execute("""
                UPDATE asset_registry
                SET valid_to = CURRENT_TIMESTAMP
                WHERE symbol = ? AND market_type = ? AND valid_to IS NULL
            """, [symbol, market_type])
            # Insert new version
            conn.execute("""
                INSERT INTO asset_registry (
                    symbol, market_type, base_asset, quote_asset, is_active,
                    listing_date, delisting_date, contract_type, tick_size,
                    step_size, min_qty, contract_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                symbol, market_type, base_asset, quote_asset, is_active,
                listing_date, delisting_date, contract_type, tick_size,
                step_size, min_qty, contract_size
            ])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def query_files(
        self,
        symbol: Optional[str] = None,
        market: Optional[str] = None,
        dataset_type: Optional[str] = None,
        interval: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query files from file_index_v1 matching parameters."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM file_index_v1 WHERE 1=1"
            params = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if market:
                query += " AND market = ?"
                params.append(market)
            if dataset_type:
                query += " AND dataset_type = ?"
                params.append(dataset_type)
            if interval:
                query += " AND interval = ?"
                params.append(interval)
            if year:
                query += " AND year = ?"
                params.append(year)
            query += " ORDER BY symbol, year, month"

            res = conn.execute(query, params).fetchall()
            cols = [desc[0] for desc in conn.description]
            return [dict(zip(cols, row)) for row in res]
        finally:
            conn.close()

    def delete_file(self, file_id: str) -> None:
        """Remove a file record from the index."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM file_index_v1 WHERE file_id = ?", [file_id])
        finally:
            conn.close()

    def cleanup_orphaned(self) -> int:
        """Remove index entries whose file_path no longer exists on disk. Returns count removed."""
        conn = self._get_connection()
        removed = 0
        try:
            rows = conn.execute("SELECT file_id, file_path FROM file_index_v1").fetchall()
            for file_id, file_path in rows:
                if file_path and not Path(file_path).exists():
                    conn.execute("DELETE FROM file_index_v1 WHERE file_id = ?", [file_id])
                    removed += 1
        finally:
            conn.close()
        return removed

    def query_assets(
        self,
        market_type: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Query current active assets from asset_registry (valid_to IS NULL)."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM asset_registry WHERE valid_to IS NULL"
            params = []
            if market_type:
                query += " AND market_type = ?"
                params.append(market_type)
            if is_active is not None:
                query += " AND is_active = ?"
                params.append(is_active)
            query += " ORDER BY symbol"

            res = conn.execute(query, params).fetchall()
            cols = [desc[0] for desc in conn.description]
            return [dict(zip(cols, row)) for row in res]
        finally:
            conn.close()


db_manager = DatabaseManager()
