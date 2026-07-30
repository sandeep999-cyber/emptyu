"""Causality and no-leakage property tests for Alignment Engine across all 4 Phase 1 modalities."""

import pandas as pd
import numpy as np
import pytest
from src.data.alignment import AlignmentEngine


def _make_klines(n=1440, start=1700000000000, interval_ms=60000):
    ts = np.arange(start, start + n * interval_ms, interval_ms)
    return pd.DataFrame({
        "timestamp": ts, "open": 50000.0, "high": 50100.0,
        "low": 49900.0, "close": 50050.0, "volume": 10.0,
    })


def test_causal_alignment_no_future_leakage_funding():
    engine = AlignmentEngine()
    df_klines = _make_klines()
    ts = df_klines["timestamp"].values
    settlement_ts = ts[480]

    base = pd.DataFrame({"settlement_time": [ts[0], settlement_ts], "funding_rate": [0.0001, 0.0005]})
    perturbed = pd.DataFrame({"settlement_time": [ts[0], settlement_ts], "funding_rate": [0.0001, 0.9999]})

    res_base = engine.align_symbol_data("BTCUSDT", df_klines, df_funding=base)
    res_pert = engine.align_symbol_data("BTCUSDT", df_klines, df_funding=perturbed)

    before_base = res_base[res_base["timestamp"] < settlement_ts]["funding_rate"].values
    before_pert = res_pert[res_pert["timestamp"] < settlement_ts]["funding_rate"].values
    np.testing.assert_array_equal(before_base, before_pert)

    after_pert = res_pert[res_pert["timestamp"] >= settlement_ts]["funding_rate"].values
    assert np.all(after_pert == 0.9999)


def test_causal_alignment_no_future_leakage_open_interest():
    engine = AlignmentEngine()
    df_klines = _make_klines()
    ts = df_klines["timestamp"].values
    oit_ts = ts[600]

    base = pd.DataFrame({"timestamp": [ts[0], oit_ts], "open_interest": [100.0, 200.0]})
    perturbed = pd.DataFrame({"timestamp": [ts[0], oit_ts], "open_interest": [100.0, 99999.0]})

    res_base = engine.align_symbol_data("BTCUSDT", df_klines, df_open_interest=base)
    res_pert = engine.align_symbol_data("BTCUSDT", df_klines, df_open_interest=perturbed)

    before_base = res_base[res_base["timestamp"] < oit_ts]["open_interest"].values
    before_pert = res_pert[res_pert["timestamp"] < oit_ts]["open_interest"].values
    np.testing.assert_array_equal(before_base, before_pert)

    after_pert = res_pert[res_pert["timestamp"] >= oit_ts]["open_interest"].values
    assert np.all(after_pert == 99999.0)


def test_causal_alignment_no_future_leakage_calendar():
    engine = AlignmentEngine()
    df_klines = _make_klines()
    ts = df_klines["timestamp"].values

    future_cal = pd.DataFrame({
        "timestamp": [ts[0], ts[1440] if len(ts) > 1440 else ts[-1] + 3600000],
        "minute_of_day": [0, 0],
    })
    perturbed_cal = pd.DataFrame({
        "timestamp": [ts[0], ts[1440] if len(ts) > 1440 else ts[-1] + 3600000],
        "minute_of_day": [0, 999],
    })

    res_base = engine.align_symbol_data("BTCUSDT", df_klines, df_calendar=future_cal)
    res_pert = engine.align_symbol_data("BTCUSDT", df_klines, df_calendar=perturbed_cal)

    np.testing.assert_array_equal(
        res_base["minute_of_day"].values,
        res_pert["minute_of_day"].values,
    )


def test_causal_alignment_no_future_leakage_klines():
    engine = AlignmentEngine()
    df_klines = _make_klines()
    ts = df_klines["timestamp"].values

    base_close = df_klines["close"].copy()
    df_klines_pert = df_klines.copy()
    df_klines_pert.loc[df_klines_pert.index[600], "close"] = 99999.0

    res_base = engine.align_symbol_data("BTCUSDT", df_klines)
    res_pert = engine.align_symbol_data("BTCUSDT", df_klines_pert)

    before_base = res_base[res_base["timestamp"] < ts[600]]["close"].values
    before_pert = res_pert[res_pert["timestamp"] < ts[600]]["close"].values
    np.testing.assert_array_equal(before_base, before_pert)


def test_future_modality_raises_not_implemented():
    engine = AlignmentEngine()
    df_klines = _make_klines(n=10)
    dummy = pd.DataFrame({"timestamp": [0], "price": [1.0]})

    with pytest.raises(NotImplementedError, match="agg_trades"):
        engine.align_symbol_data("BTCUSDT", df_klines, df_agg_trades=dummy)

    with pytest.raises(NotImplementedError, match="depth"):
        engine.align_symbol_data("BTCUSDT", df_klines, df_depth=dummy)

    with pytest.raises(NotImplementedError, match="liquidations"):
        engine.align_symbol_data("BTCUSDT", df_klines, df_liquidations=dummy)


def test_empty_klines_returns_empty():
    engine = AlignmentEngine()
    result = engine.align_symbol_data("BTCUSDT", pd.DataFrame())
    assert result.empty


def test_calendar_fallback_generation():
    engine = AlignmentEngine()
    df_klines = _make_klines(n=5)
    result = engine.align_symbol_data("BTCUSDT", df_klines)
    assert "minute_of_day" in result.columns
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert len(result) == 5
