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


def _aligned_df():
    return pd.DataFrame({
        "timestamp": [1700000000000 + 60000 * i for i in range(4)],
        "open": [50000.0, 50100.0, 50200.0, 50300.0],
        "high": [50200.0, 50300.0, 50400.0, 50500.0],
        "low": [49900.0, 50000.0, 50100.0, 50200.0],
        "close": [50100.0, 50200.0, 50300.0, 50400.0],
        "volume": [10.0, 12.0, 15.0, 14.0],
        "funding_rate": [0.0001, 0.0001, 0.0002, 0.0001],
        "open_interest": [1000.0, 1005.0, 1010.0, 1015.0],
        "minute_of_day": [0, 1, 2, 3],
        "hour": [0, 0, 0, 0],
        "day_of_week": [0, 0, 0, 0],
        "day_of_month": [1, 1, 1, 1],
        "month": [1, 1, 1, 1],
        "quarter": [1, 1, 1, 1],
        "year": [2024, 2024, 2024, 2024],
        "is_weekend": [0, 0, 0, 0],
    })


def test_returns_features_shape_and_timestamps():
    builder = FeatureBuilder()
    feats, fm, ts = builder.build_features(_aligned_df(), style="returns")
    assert feats.shape == (4, 15)
    assert fm.shape == (4, 15)
    assert ts.shape == (4,)
    # Calendar columns preserved in input (7..14).
    assert np.all(fm[:, 7:])
    assert feats.shape[1] == 15


def test_returns_log_return_correct_and_first_row_masked():
    builder = FeatureBuilder()
    df = _aligned_df()
    feats, fm, _ = builder.build_features(df, style="returns")
    # log_return[i] = log(close[i]/close[i-1])
    assert np.isclose(feats[1, 0], np.log(50200.0 / 50100.0))
    assert np.isclose(feats[2, 0], np.log(50300.0 / 50200.0))
    # First row has no previous close -> masked out (value zero).
    assert not fm[0, 0]
    assert feats[0, 0] == 0.0
    assert np.all(fm[1:, 0])


def test_returns_range_body_volume_oi():
    builder = FeatureBuilder()
    df = _aligned_df()
    feats, fm, _ = builder.build_features(df, style="returns")
    # hl_range = (high-low)/close
    assert np.isclose(feats[0, 1], (50200.0 - 49900.0) / 50100.0)
    # oc_body = (close-open)/close
    assert np.isclose(feats[0, 2], (50100.0 - 50000.0) / 50100.0)
    # log_volume = log1p(volume)
    assert np.isclose(feats[0, 3], np.log1p(10.0))
    # open_interest = log1p(open_interest)
    assert np.isclose(feats[0, 6], np.log1p(1000.0))
    # funding rate preserved as-is
    assert np.isclose(feats[0, 5], 0.0001)


def test_returns_volume_change_first_row_masked():
    builder = FeatureBuilder()
    feats, fm, _ = builder.build_features(_aligned_df(), style="returns")
    assert not fm[0, 4]
    assert feats[0, 4] == 0.0
    # volume_change[1] = log1p(12) - log1p(10)
    assert np.isclose(feats[1, 4], np.log1p(12.0) - np.log1p(10.0))


def test_returns_raw_still_default():
    builder = FeatureBuilder()
    feats, fm, _ = builder.build_features(_aligned_df())
    # Default style keeps raw price levels.
    assert feats[0, 0] == 50000.0
    assert np.all(fm[0, :])


def test_stale_funding_and_oi_masked_in_raw_style():
    builder = FeatureBuilder()
    df = _aligned_df()
    df["funding_rate_stale"] = [True, False, False, True]
    df["open_interest_stale"] = [False, True, False, True]
    feats, fm, _ = builder.build_features(df, style="raw")
    funding_idx = builder.feature_order.index("funding_rate")
    oi_idx = builder.feature_order.index("open_interest")
    assert list(fm[:, funding_idx]) == [False, True, True, False]
    assert list(fm[:, oi_idx]) == [True, False, True, False]


def test_stale_funding_and_oi_masked_in_returns_style():
    builder = FeatureBuilder()
    df = _aligned_df()
    df["funding_rate_stale"] = [True, False, False, True]
    df["open_interest_stale"] = [False, True, False, True]
    feats, fm, _ = builder.build_features(df, style="returns")
    assert list(fm[:, 5]) == [False, True, True, False]
    assert list(fm[:, 6]) == [True, False, True, False]
