"""Modality Registry Manager with schema validation."""

from typing import Any, Dict, List
from src.config import config

_KNOWN_MODALITIES = frozenset({
    "klines", "price", "funding", "open_interest", "calendar",
    "agg_trades", "depth", "liquidations",
})


class ModalityRegistry:
    """Manages active modalities configured in modalities_v1.yaml with validation."""

    def __init__(self, config_dict: Dict[str, Any] | None = None):
        self.modalities = config_dict if config_dict is not None else config.modalities
        self._validate()

    def _validate(self) -> None:
        for key, val in self.modalities.items():
            if key not in _KNOWN_MODALITIES:
                raise ValueError(
                    f"Unknown modality '{key}' in modalities config. "
                    f"Known modalities: {sorted(_KNOWN_MODALITIES)}"
                )
            if not isinstance(val, dict) or "enabled" not in val:
                raise ValueError(f"Modality '{key}' must have an 'enabled' field.")
            if not isinstance(val["enabled"], bool):
                raise TypeError(f"Modality '{key}' enabled must be bool, got {type(val['enabled']).__name__}")

    def is_enabled(self, modality_name: str) -> bool:
        mod = self.modalities.get(modality_name, {})
        return mod.get("enabled", False)

    def get_active_modalities(self) -> List[str]:
        return [k for k, v in self.modalities.items() if v.get("enabled", False)]


modality_registry = ModalityRegistry()
