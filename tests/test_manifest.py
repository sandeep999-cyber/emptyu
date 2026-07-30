"""Unit tests for Manifest Builder."""

import tempfile
from pathlib import Path
import json
from src.data.manifest_builder import ManifestBuilder


class TestManifestBuilder:
    def test_manifest_builder(self):
        builder = ManifestBuilder()
        manifest = builder.build_manifest(
            snapshot_date="2026-07-30",
            train_symbols=["BTCUSDT", "ETHUSDT"],
            val_symbols=["SOLUSDT"], test_symbols=["DOGEUSDT"],
        )
        data = manifest["training_manifest"]
        assert data["dataset"]["snapshot"] == "2026-07-30"
        assert "BTCUSDT" in data["splits"]["train"]["symbols"]
        assert "SOLUSDT" in data["splits"]["validation"]["symbols"]
        assert "DOGEUSDT" in data["splits"]["test"]["symbols"]
        assert data["random_seed"]["python"] == 42
        assert "file_ledger" in data
        assert isinstance(data["file_ledger"], dict)

    def test_version_locks_complete(self):
        builder = ManifestBuilder()
        manifest = builder.build_manifest()
        ds = manifest["training_manifest"]["dataset"]
        assert ds["alignment_version"] == "alignment_v1.yaml"
        assert ds["feature_builder_version"] == "v1"
        assert ds["windowing_version"] == "windowing_v1.yaml"
        assert ds["canonical_schema_version"] == "v1"
        assert ds["market_state_schema_version"] == "v1"
        assert ds["modality_registry"] == "modalities_v1.yaml"

    def test_save_manifest(self):
        builder = ManifestBuilder()
        manifest = builder.build_manifest()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            builder.save_manifest(manifest, path)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "training_manifest" in data

    def test_fingerprint_in_manifest(self):
        builder = ManifestBuilder()
        manifest = builder.build_manifest(file_hashes={"f1": "h1"})
        fp = builder.compute_fingerprint(manifest)
        assert fp["file_count"] == 1
        assert "fingerprint" in fp
