import pytest
import torch

from src.training.sampler import EpochMarketSampler


class TestEpochMarketSampler:
    def test_epoch_changes_order(self):
        s1 = EpochMarketSampler(100, shuffle=True, seed=42)
        s1.set_epoch(1)
        s2 = EpochMarketSampler(100, shuffle=True, seed=42)
        s2.set_epoch(2)
        assert list(s1) != list(s2)

    def test_deterministic_per_epoch(self):
        a = EpochMarketSampler(100, shuffle=True, seed=42)
        b = EpochMarketSampler(100, shuffle=True, seed=42)
        a.set_epoch(3)
        b.set_epoch(3)
        assert list(a) == list(b)

    def test_epoch_zero_differs_from_epoch_one(self):
        """Regression: epoch stuck at 0 -> identical permutation every epoch."""
        s0 = EpochMarketSampler(100, shuffle=True, seed=42)
        s1 = EpochMarketSampler(100, shuffle=True, seed=42)
        s0.set_epoch(0)
        s1.set_epoch(1)
        assert list(s0) != list(s1)

    def test_len(self):
        s = EpochMarketSampler(50, shuffle=True, seed=42)
        assert len(s) == 50

    def test_no_shuffle(self):
        s = EpochMarketSampler(50, shuffle=False, seed=42)
        assert list(s) == list(range(50))
