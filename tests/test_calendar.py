"""Unit tests for Calendar Builder."""

import tempfile
from pathlib import Path
import pyarrow.parquet as pq
from src.data.calendar_builder import CalendarBuilder
from src.data.db import DatabaseManager


class TestCalendarBuilder:
    def test_build_calendar_for_year(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = DatabaseManager(db_path=tmp / "test.duckdb")
            cb = CalendarBuilder(canonical_dir=tmp / "canonical", db_manager_override=db)
            out = cb.build_calendar_for_year("futures", "BTCUSDT", 2024)
            assert out.exists()
            table = pq.read_table(str(out))
            df = table.to_pandas()
            assert "timestamp" in df.columns
            assert "minute_of_day" in df.columns
            assert "is_weekend" in df.columns
            assert len(df) == 527040  # 2024 is leap year: 366 * 1440

    def test_build_calendar_registers_in_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = DatabaseManager(db_path=tmp / "test.duckdb")
            cb = CalendarBuilder(canonical_dir=tmp / "canonical", db_manager_override=db)
            cb.build_calendar_for_year("futures", "BTCUSDT", 2024)
            files = db.query_files(symbol="BTCUSDT", dataset_type="calendar")
            assert len(files) == 1
            assert files[0]["interval"] == "1m"

    def test_calendar_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = DatabaseManager(db_path=tmp / "test.duckdb")
            cb = CalendarBuilder(canonical_dir=tmp / "canonical", db_manager_override=db)
            out = cb.build_calendar_for_year("futures", "BTCUSDT", 2024)
            table = pq.read_table(str(out))
            df = table.to_pandas()
            assert set(df.columns) >= {
                "timestamp", "minute_of_day", "hour", "day_of_week",
                "day_of_month", "month", "quarter", "year", "is_weekend",
            }
