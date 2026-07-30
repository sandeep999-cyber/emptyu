"""Unit tests for Modality Registry with schema validation."""

import pytest
from src.data.modality_registry import ModalityRegistry


class TestModalityRegistry:
    def test_is_enabled(self):
        reg = ModalityRegistry({"price": {"enabled": True}, "funding": {"enabled": False}})
        assert reg.is_enabled("price")
        assert not reg.is_enabled("funding")

    def test_get_active_modalities(self):
        reg = ModalityRegistry({
            "klines": {"enabled": True}, "funding": {"enabled": False},
            "open_interest": {"enabled": True}, "calendar": {"enabled": True},
        })
        active = reg.get_active_modalities()
        assert "klines" in active
        assert "open_interest" in active
        assert "calendar" in active
        assert "funding" not in active

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            ModalityRegistry({"klines": {"enabled": True}, "unknown_mod": {"enabled": True}})

    def test_rejects_missing_enabled_field(self):
        with pytest.raises(ValueError, match="must have an 'enabled' field"):
            ModalityRegistry({"klines": {"foo": "bar"}})

    def test_rejects_non_bool_enabled(self):
        with pytest.raises(TypeError, match="enabled must be bool"):
            ModalityRegistry({"klines": {"enabled": "yes"}})

    def test_all_known_modalities_accepted(self):
        reg = ModalityRegistry({
            "klines": {"enabled": True}, "funding": {"enabled": True},
            "open_interest": {"enabled": True}, "calendar": {"enabled": True},
            "agg_trades": {"enabled": False}, "depth": {"enabled": False},
            "liquidations": {"enabled": False}, "price": {"enabled": True},
        })
        assert len(reg.get_active_modalities()) == 5

    def test_empty_registry(self):
        reg = ModalityRegistry({})
        assert reg.get_active_modalities() == []
        assert not reg.is_enabled("klines")
