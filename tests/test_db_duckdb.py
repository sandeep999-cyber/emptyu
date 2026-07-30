"""Unit tests for DuckDB Database Manager."""

import tempfile
from pathlib import Path
import pytest
from src.data.db import DatabaseManager


class TestDatabaseManager:
    def test_database_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_index.duckdb"
            db = DatabaseManager(db_path=db_path)

            db.register_asset(
                symbol="BTCUSDT", market_type="futures",
                base_asset="BTC", quote_asset="USDT", is_active=True,
            )
            assets = db.query_assets()
            assert len(assets) == 1
            assert assets[0]["symbol"] == "BTCUSDT"

            db.register_file(
                file_id="test_id_1", symbol="BTCUSDT", market="futures",
                dataset_type="klines", interval="1m", year=2024, month=1,
                start_ts=1700000000000, end_ts=1700086400000,
                row_count=1440, file_size=10240, sha256="hash123",
                schema_hash="schema123", file_path="/tmp/test.parquet",
            )
            files = db.query_files(symbol="BTCUSDT")
            assert len(files) == 1
            assert files[0]["row_count"] == 1440

    def test_asset_registry_composite_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(db_path=Path(tmpdir) / "test.duckdb")
            db.register_asset(symbol="BTCUSDT", market_type="futures", base_asset="BTC", quote_asset="USDT")
            db.register_asset(symbol="BTCUSDT", market_type="spot", base_asset="BTC", quote_asset="USDT")
            assets = db.query_assets()
            assert len(assets) == 2
            spot_assets = db.query_assets(market_type="spot")
            assert len(spot_assets) == 1

    def test_delete_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(db_path=Path(tmpdir) / "test.duckdb")
            db.register_file(
                file_id="test_delete", symbol="BTCUSDT", market="futures",
                dataset_type="klines", interval="1m", year=2024, month=1,
                start_ts=0, end_ts=1000, row_count=10, file_size=100,
                sha256="abc", schema_hash="def", file_path="/tmp/test.parquet",
            )
            assert len(db.query_files()) == 1
            db.delete_file("test_delete")
            assert len(db.query_files()) == 0

    def test_cleanup_orphaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "exists.parquet"
            file_path.write_bytes(b"test data")
            db = DatabaseManager(db_path=Path(tmpdir) / "test.duckdb")
            db.register_file(
                file_id="f1", symbol="BTCUSDT", market="futures",
                dataset_type="klines", interval="1m", year=2024, month=1,
                start_ts=0, end_ts=1000, row_count=10, file_size=100,
                sha256="abc", schema_hash="def", file_path=str(file_path),
            )
            db.register_file(
                file_id="f2", symbol="ETHUSDT", market="futures",
                dataset_type="klines", interval="1m", year=2024, month=1,
                start_ts=0, end_ts=1000, row_count=10, file_size=100,
                sha256="def", schema_hash="ghi",
                file_path=str(Path(tmpdir) / "nonexistent.parquet"),
            )
            removed = db.cleanup_orphaned()
            assert removed == 1
            assert len(db.query_files()) == 1

    def test_asset_version_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(db_path=Path(tmpdir) / "test.duckdb")
            db.register_asset("BTCUSDT", "futures", "BTC", "USDT", is_active=True)
            db.register_asset("BTCUSDT", "futures", "BTC", "USDT", is_active=False)
            assets = db.query_assets()
            assert len(assets) == 1
            assert assets[0]["is_active"] == False

    def test_register_files_batch_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(db_path=Path(tmpdir) / "test.duckdb")
            valid = ("f1", "BTCUSDT", "futures", "klines", "1m", 2024, 1,
                     0, 1000, 10, 100, "abc", "def", "/tmp/t.parquet", "CONVERTED")
            db.register_files_batch([valid])
            assert len(db.query_files()) == 1
            invalid = ("f2", "BTCUSDT")
            with pytest.raises(ValueError, match="Record 0 has 2 elements"):
                db.register_files_batch([invalid])
