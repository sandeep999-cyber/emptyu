"""Unit tests for Feature Builder."""

import pandas as pd
import numpy as np
import pytest
from src.data.feature_builder import FeatureBuilder


def test_feature_builder():
    builder = FeatureBuilder()
    df_aligned = pd.DataFrame({
        "timestamp": [1700000000000, 1700000060000],
        "open": [50000.0, 50100.0],
        "high": [50200.0, 50300.0],
        "low": [49900.0, 50000.0],
        "close": [50100.0, 50200.0],
        "volume": [10.0, 12.0],
        "funding_rate": [0.0001, 0.0001],
        "open_interest": [1000.0, 1005.0],
        "minute_of_day": [0, 1],
        "hour": [0, 0],
        "day_of_week": [0, 0],
        "day_of_month": [1, 1],
        "month": [1, 1],
        "quarter": [1, 1],
        "year": [2024, 2024],
        "is_weekend": [0, 0]
    })

    feats, fm, ts = builder.build_features(df_aligned)
    assert feats.shape == (2, 15)
    assert fm.shape == (2, 15)
    assert len(ts) == 2
    assert np.all(fm == True)

    # Verify caller DataFrame not mutated
    df_check = pd.DataFrame({
        "timestamp": [0, 60000],
        "open": [np.nan, 1.0]
    })
    before_nan = df_check["open"].isna().sum()
    builder.build_features(df_check)
    after_nan = df_check["open"].isna().sum()
    assert before_nan == after_nan, "Feature builder should not mutate caller DataFrame"


def test_feature_builder_missing_columns():
    builder = FeatureBuilder()
    # Only price columns
    df = pd.DataFrame({
        "timestamp": [0, 60000],
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [102.0, 103.0],
        "volume": [10.0, 12.0],
    })
    feats, fm, ts = builder.build_features(df)
    assert feats.shape == (2, 15)
    # Price columns should be unmasked, others should be mask=False
    for i, col in enumerate(builder.feature_order):
        if col in df.columns:
            assert np.all(fm[:, i]), f"Column {col} should be fully masked (observed)"
        else:
            assert not np.any(fm[:, i]), f"Column {col} should not be masked (missing)"


def test_feature_builder_schema_dimension_mismatch():
    """Verify that mismatched feature_order / feature_dimension raises."""
    with pytest.raises(ValueError, match="feature_order length"):
        FeatureBuilder(schema_dict={
            "feature_order": ["open", "high"],
            "feature_dimension": 5,
        })
