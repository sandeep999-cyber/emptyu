"""Unit tests for Parquet Converter with provenance embedding."""

import hashlib
import tempfile
import zipfile
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from src.data.parquet_converter import ParquetConverter
from src.data.db import DatabaseManager


class TestParquetConverter:
    def test_convert_zip_to_parquet_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = DatabaseManager(db_path=tmp / "test.duckdb")

            csv_content = b"1700000000000,50000.0,50100.0,49900.0,50050.0,10.0,1700000059999,1000.0,100,5.0,500.0,0\n"
            csv_content += b"1700000060000,50050.0,50200.0,50000.0,50150.0,12.0,1700000119999,1200.0,120,6.0,600.0,0\n"

            zip_path = tmp / "BTCUSDT-1m-2024-01.zip"
            with zipfile.ZipFile(str(zip_path), "w") as zf:
                zf.writestr("BTCUSDT-1m-2024-01.csv", csv_content)

            conv = ParquetConverter(canonical_dir=tmp / "canonical", db_manager_override=db)
            out = conv.convert_zip_to_parquet(
                zip_path=zip_path, market="futures", symbol="BTCUSDT",
                dataset_type="klines", interval="1m", year=2024, month=1,
                download_date="2024-02-01", alignment_version="alignment_v1.yaml",
            )

            assert out.exists()
            table = pq.read_table(str(out))
            meta = table.schema.metadata or {}
            assert b"provenance_created_by" in meta
            assert b"provenance_download_date" in meta
            assert b"provenance_alignment_version" in meta
            assert b"provenance_converter_version" in meta
            assert b"provenance_snapshot" in meta
            assert meta[b"provenance_download_date"] == b"2024-02-01"
            assert meta[b"provenance_alignment_version"] == b"alignment_v1.yaml"

    def test_convert_registers_in_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = DatabaseManager(db_path=tmp / "test.duckdb")
            csv_content = b"1700000000000,100.0,105.0,99.0,102.0,10.0,1700000059999,1000.0,100,5.0,500.0,0\n"
            zip_path = tmp / "BTCUSDT-1m-2024-01.zip"
            with zipfile.ZipFile(str(zip_path), "w") as zf:
                zf.writestr("BTCUSDT-1m-2024-01.csv", csv_content)

            conv = ParquetConverter(canonical_dir=tmp / "canonical", db_manager_override=db)
            conv.convert_zip_to_parquet(
                zip_path=zip_path, market="futures", symbol="BTCUSDT",
                dataset_type="klines", interval="1m", year=2024, month=1,
            )
            files = db.query_files(symbol="BTCUSDT")
            assert len(files) == 1
            assert files[0]["row_count"] == 1
            assert files[0]["status"] == "CONVERTED"

    def test_convert_missing_file_raises(self):
        conv = ParquetConverter()
        with pytest.raises(FileNotFoundError):
            conv.convert_zip_to_parquet(
                zip_path=Path("/nonexistent.zip"), market="futures",
                symbol="BTCUSDT", dataset_type="klines",
            )
