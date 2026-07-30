"""Unit tests for Resampler verifying mathematical OHLCV accuracy."""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from src.data.resampler import KlineResampler
from src.data.db import DatabaseManager


def test_resampler_mathematical_accuracy():
    """Verify that 5m resample open=first, high=max, low=min, close=last, volume=sum."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db = DatabaseManager(db_path=tmp / "test.duckdb")
        input_dir = tmp / "input"
        input_dir.mkdir()
        input_file = input_dir / "2024-01.parquet"

        df_1m = pd.DataFrame({
            "timestamp": [0, 60000, 120000, 180000, 240000],
            "open": [100.0, 102.0, 105.0, 103.0, 101.0],
            "high": [103.0, 106.0, 108.0, 104.0, 102.0],
            "low": [99.0, 101.0, 104.0, 100.0, 98.0],
            "close": [102.0, 105.0, 103.0, 101.0, 107.0],
            "volume": [10.0, 20.0, 15.0, 25.0, 30.0],
            "quote_volume": [1000.0, 2000.0, 1500.0, 2500.0, 3000.0],
            "count": [10, 20, 15, 25, 30],
            "taker_buy_volume": [5.0, 10.0, 7.0, 12.0, 15.0],
            "taker_buy_quote_volume": [500.0, 1000.0, 700.0, 1200.0, 1500.0]
        })

        table_1m = pa.Table.from_pandas(df_1m)
        pq.write_table(table_1m, str(input_file))

        resampler = KlineResampler(canonical_dir=tmp / "canonical", db_manager_override=db)
        out_file = resampler.resample_file(
            input_1m_path=input_file,
            market="futures",
            symbol="BTCUSDT",
            target_interval="5m",
            year=2024,
            month=1
        )

        assert out_file.exists()
        res_df = pq.read_table(str(out_file)).to_pandas()

        assert len(res_df) == 1
        row = res_df.iloc[0]

        assert row["open"] == 100.0
        assert row["high"] == 108.0
        assert row["low"] == 98.0
        assert row["close"] == 107.0
        assert row["volume"] == 100.0

        # Verify constituent_count was tracked
        assert "constituent_count" in res_df.columns
        assert row["constituent_count"] == 5


def test_resampler_partial_candle_warning():
    """Verify incomplete candles are flagged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db = DatabaseManager(db_path=tmp / "test.duckdb")
        input_dir = tmp / "input"
        input_dir.mkdir()
        input_file = input_dir / "2024-01.parquet"

        df_1m = pd.DataFrame({
            "timestamp": [0, 60000],  # Only 2 minutes (not 5)
            "open": [100.0, 102.0],
            "high": [103.0, 106.0],
            "low": [99.0, 101.0],
            "close": [102.0, 105.0],
            "volume": [10.0, 20.0],
            "quote_volume": [1000.0, 2000.0],
            "count": [10, 20],
            "taker_buy_volume": [5.0, 10.0],
            "taker_buy_quote_volume": [500.0, 1000.0]
        })

        table_1m = pa.Table.from_pandas(df_1m)
        pq.write_table(table_1m, str(input_file))

        resampler = KlineResampler(canonical_dir=tmp / "canonical", db_manager_override=db)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out_file = resampler.resample_file(
                input_1m_path=input_file,
                market="futures",
                symbol="BTCUSDT",
                target_interval="5m",
                year=2024,
                month=1
            )
            assert len(w) >= 1
            assert any("constituent" in str(m.message).lower() for m in w)
