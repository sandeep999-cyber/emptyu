"""Unit tests for Metadata Manager."""

import pandas as pd
from src.data.metadata import MetadataManager


class TestMetadata:
    def test_create_dataset_version(self):
        mm = MetadataManager()
        ver = mm.create_dataset_version("2.0.0", "2026-07-30", 2)
        assert ver["version"] == "2.0.0"
        assert ver["schema_version"] == 2

    def test_create_dataset_version_defaults(self):
        mm = MetadataManager()
        ver = mm.create_dataset_version()
        assert ver["version"] == "1.0.0"
        assert ver["schema_version"] == 1

    def test_compute_statistics(self):
        mm = MetadataManager()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        stats = mm.compute_statistics(df)
        assert stats["a"]["mean"] == 2.0
        assert stats["a"]["std"] == 1.0
        assert stats["b"]["min"] == 10.0

    def test_compute_statistics_with_nan(self):
        mm = MetadataManager()
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, 30.0]})
        stats = mm.compute_statistics(df)
        assert stats["a"]["mean"] == 2.0
