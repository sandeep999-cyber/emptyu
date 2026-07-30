"""Unit tests for Quality Report Generator."""

import pandas as pd
from src.data.quality_report import QualityReportGenerator


def test_quality_report_generator():
    generator = QualityReportGenerator()
    df_aligned = pd.DataFrame({
        "timestamp": [1000, 61000, 121000],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [102.0, 103.0, 104.0],
        "volume": [10.0, 12.0, 15.0],
        "funding_rate": [0.0001, 0.0001, 0.0001],
        "open_interest": [1000.0, 1000.0, 1005.0]
    })

    report = generator.generate_report("BTCUSDT", "futures", df_aligned)
    assert report["symbol"] == "BTCUSDT"
    assert report["total_records"] == 3
    assert report["duplicate_rows"] == 0
    assert report["gap_count"] == 0
    assert report["quality_score"] == 100.0
    assert "forward_fill_percentage" in report
    assert "missing_values_per_modality" in report
    assert "alignment_coverage" in report
    assert "symbols_with_incomplete_history" in report
    assert "resampling_statistics" in report
    assert report["alignment_coverage"]["funding_rate"] == 1.0
    assert report["alignment_coverage"]["open_interest"] == 1.0


def test_quality_report_empty():
    generator = QualityReportGenerator()
    report = generator.generate_report("ETHUSDT", "futures", pd.DataFrame())
    assert report["quality_score"] == 0.0
    assert report["symbol"] == "ETHUSDT"
