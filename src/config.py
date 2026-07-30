"""Central configuration module loading YAML and JSON configs."""

import os
import json
from pathlib import Path
from typing import Any, Dict
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"


def load_yaml_config(filename: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the configs directory."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_config(filename: str) -> Dict[str, Any]:
    """Load a JSON configuration file from the configs directory."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Config:
    """System configuration container."""
    def __init__(self):
        self.download = load_yaml_config("download.yaml").get("download", {})
        self.storage = load_yaml_config("storage.yaml").get("storage", {})
        self.validation = load_yaml_config("validation.yaml").get("validation", {})
        self.dataset = load_yaml_config("dataset.yaml").get("dataset", {})
        self.modalities = load_yaml_config("modalities_v1.yaml")
        self.alignment = load_yaml_config("alignment_v1.yaml")
        self.market_state_schema = load_json_config("market_state_schema_v1.json")
        self.windowing = load_yaml_config("windowing_v1.yaml")

    @property
    def raw_dir(self) -> Path:
        return BASE_DIR / self.storage.get("raw_dir", "storage/raw")

    @property
    def canonical_dir(self) -> Path:
        return BASE_DIR / self.storage.get("canonical_dir", "storage/canonical")

    @property
    def lake_dir(self) -> Path:
        return BASE_DIR / self.storage.get("lake_dir", "storage/lake")

    @property
    def training_dir(self) -> Path:
        return BASE_DIR / self.storage.get("training_dir", "storage/training")

    @property
    def db_path(self) -> Path:
        return BASE_DIR / self.storage.get("db_path", "storage/training/index.duckdb")

    @property
    def experiment_db_path(self) -> Path:
        return BASE_DIR / self.storage.get("experiment_db_path", "storage/training/experiment_registry.duckdb")

    @property
    def logs_dir(self) -> Path:
        return BASE_DIR / self.storage.get("logs_dir", "logs")


config = Config()
