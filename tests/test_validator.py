"""Unit tests for Data Validator covering all Phase 1 modality rules."""

import pandas as pd
import pytest
from src.data.validator import DataValidator


class TestKlinesValidator:
    def test_valid_klines(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "high": [105.0, 106.0, 107.0], "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 104.0],
        })
        is_valid, errors = validator.validate_klines(df)
        assert is_valid

    def test_invalid_high_lt_low(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000], "high": [90.0, 106.0],
            "low": [99.0, 100.0], "close": [102.0, 103.0],
        })
        is_valid, errors = validator.validate_klines(df)
        assert not is_valid
        assert any("high < low" in e for e in errors)

    def test_close_positive(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000], "high": [105.0, 106.0],
            "low": [99.0, 100.0], "close": [0.0, 103.0],
        })
        is_valid, errors = validator.validate_klines(df)
        assert not is_valid
        assert any("close <= 0" in e for e in errors)

    def test_rejects_empty(self):
        validator = DataValidator()
        is_valid, errors = validator.validate_klines(pd.DataFrame())
        assert not is_valid
        assert "Empty" in errors[0]

    def test_gap_detection(self):
        validator = DataValidator({"max_timestamp_gap_seconds": 1})
        df = pd.DataFrame({
            "timestamp": [1000, 2000, 200000], "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0], "close": [102.0, 103.0, 104.0],
        })
        is_valid, errors = validator.validate_klines(df)
        assert not is_valid


class TestFundingValidator:
    def test_valid_funding(self):
        validator = DataValidator()
        df = pd.DataFrame({"timestamp": [1000, 2000, 3000], "funding_rate": [0.0001, 0.0002, 0.0003]})
        is_valid, errors = validator.validate_funding(df)
        assert is_valid

    def test_funding_missing_values_rejected(self):
        validator = DataValidator({"modalities": {"funding": {"allow_missing": False}}})
        df = pd.DataFrame({"timestamp": [1000, 2000], "funding_rate": [0.0001, None]})
        is_valid, errors = validator.validate_funding(df)
        assert not is_valid
        assert any("missing" in e.lower() for e in errors)

    def test_funding_empty(self):
        validator = DataValidator()
        is_valid, errors = validator.validate_funding(pd.DataFrame())
        assert not is_valid


class TestAggTradesValidator:
    def test_valid_agg_trades(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000], "trade_count": [10, 20],
            "base_volume": [1.0, 2.0],
        })
        is_valid, errors = validator.validate_agg_trades(df)
        assert is_valid

    def test_negative_trade_count(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000], "trade_count": [-1, 20],
            "base_volume": [1.0, 2.0],
        })
        is_valid, errors = validator.validate_agg_trades(df)
        assert not is_valid

    def test_negative_volume(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000, 2000], "trade_count": [10, 20],
            "base_volume": [-1.0, 2.0],
        })
        is_valid, errors = validator.validate_agg_trades(df)
        assert not is_valid


class TestDepthValidator:
    def test_valid_depth(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000], "best_bid": [99.0], "best_ask": [100.0],
        })
        is_valid, errors = validator.validate_depth(df)
        assert is_valid

    def test_bid_gt_ask(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000], "best_bid": [101.0], "best_ask": [100.0],
        })
        is_valid, errors = validator.validate_depth(df)
        assert not is_valid


class TestLiquidationsValidator:
    def test_valid_liquidations(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000], "side": ["BUY"], "quantity": [1.0],
        })
        is_valid, errors = validator.validate_liquidations(df)
        assert is_valid

    def test_invalid_side(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000], "side": ["LONG"], "quantity": [1.0],
        })
        is_valid, errors = validator.validate_liquidations(df)
        assert not is_valid

    def test_zero_quantity(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1000], "side": ["BUY"], "quantity": [0.0],
        })
        is_valid, errors = validator.validate_liquidations(df)
        assert not is_valid


class TestSHA256Verification:
    def test_verify_sha256(self, tmp_path):
        validator = DataValidator()
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        import hashlib
        h = hashlib.sha256(b"hello world").hexdigest()
        assert validator.verify_sha256(f, h)
        assert not validator.verify_sha256(f, "wrong_hash")
