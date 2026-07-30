import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from src.evaluation.embedding.linear_probe import evaluate_probe


class TestEvaluateProbe:
    @pytest.fixture
    def dummy_emb_data(self):
        rng = np.random.default_rng(42)
        n_train, n_test = 200, 50
        return {
            "train": {
                "embedding": rng.normal(0, 1, (n_train, 64)),
                "volatility": rng.integers(0, 2, size=n_train).tolist(),
                "range_expansion": rng.integers(0, 2, size=n_train).tolist(),
                "liquidity": rng.integers(0, 3, size=n_train).tolist(),
            },
            "test_cross": {
                "embedding": rng.normal(0, 1, (n_test, 64)),
                "volatility": rng.integers(0, 2, size=n_test).tolist(),
                "range_expansion": rng.integers(0, 2, size=n_test).tolist(),
                "liquidity": rng.integers(0, 3, size=n_test).tolist(),
            },
            "test_insample": {
                "embedding": rng.normal(0, 1, (n_test, 64)),
                "volatility": rng.integers(0, 2, size=n_test).tolist(),
                "range_expansion": rng.integers(0, 2, size=n_test).tolist(),
                "liquidity": rng.integers(0, 3, size=n_test).tolist(),
            },
        }

    def test_evaluate_probe_returns_dict(self, dummy_emb_data):
        train = dummy_emb_data["train"]
        test_cross = dummy_emb_data["test_cross"]
        test_in = dummy_emb_data["test_insample"]
        results = evaluate_probe(train, test_cross, test_in)
        assert "probes" in results
        assert "baselines" in results
        assert "n_train" in results

    def test_all_targets_present(self, dummy_emb_data):
        train = dummy_emb_data["train"]
        test_cross = dummy_emb_data["test_cross"]
        test_in = dummy_emb_data["test_insample"]
        results = evaluate_probe(train, test_cross, test_in)
        probe = results["probes"]
        for target in ["volatility", "range_expansion", "liquidity"]:
            assert target in probe
            assert "cross_symbol_bacc" in probe[target]
            assert "in_sample_bacc" in probe[target]
            assert "majority_baseline" in probe[target]

    def test_bacc_in_range(self, dummy_emb_data):
        train = dummy_emb_data["train"]
        test_cross = dummy_emb_data["test_cross"]
        test_in = dummy_emb_data["test_insample"]
        results = evaluate_probe(train, test_cross, test_in)
        for target in ["volatility", "range_expansion", "liquidity"]:
            for metric in ["cross_symbol_bacc", "in_sample_bacc", "majority_baseline"]:
                val = results["probes"][target][metric]
                assert 0.0 <= val <= 1.0, f"{target}.{metric} = {val}"

    def test_not_all_chance(self, dummy_emb_data):
        train = dummy_emb_data["train"]
        test_cross = dummy_emb_data["test_cross"]
        test_in = dummy_emb_data["test_insample"]
        results = evaluate_probe(train, test_cross, test_in)
        # With random data, scores should be near 0.5 (not 1.0)
        for target in ["volatility", "range_expansion", "liquidity"]:
            assert results["probes"][target]["cross_symbol_bacc"] < 0.9
