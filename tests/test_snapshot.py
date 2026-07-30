"""Unit tests for Snapshot Manager and dataset fingerprint."""

import tempfile
from pathlib import Path
import json
import pytest
from src.data.snapshot_manager import SnapshotManager
from src.data.manifest_builder import ManifestBuilder


class TestSnapshotManager:
    def test_snapshot_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(training_dir=Path(tmpdir))
            snap_path = manager.create_snapshot(
                snapshot_date="2026-07-30",
                manifest_data={"version": "1.0.0"},
                checksums_data={"sha256": "abc12345"},
                stats_data={"count": 100},
            )
            assert snap_path.exists()
            assert (snap_path / "manifest.json").exists()
            assert (snap_path / "checksums.json").exists()
            assert (snap_path / "stats.json").exists()
            assert (snap_path / "content_hash").exists()
            with open(snap_path / "manifest.json", "r") as f:
                assert json.load(f)["version"] == "1.0.0"

    def test_snapshot_immutability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(training_dir=Path(tmpdir))
            manager.create_snapshot(snapshot_date="2026-07-30", manifest_data={"v": 1})
            with pytest.raises(FileExistsError, match="already exists"):
                manager.create_snapshot(snapshot_date="2026-07-30", manifest_data={"v": 2})

    def test_content_hash_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(training_dir=Path(tmpdir))
            snap = manager.create_snapshot("2026-01-01", {"v": 1}, {"a": 1}, {"b": 2})
            h1 = (snap / "content_hash").read_text()
            manager2 = SnapshotManager(training_dir=Path(tmpdir) / "other")
            snap2 = manager2.create_snapshot("2026-01-01", {"v": 1}, {"a": 1}, {"b": 2})
            h2 = (snap2 / "content_hash").read_text()
            assert h1 == h2


class TestDatasetFingerprint:
    def test_fingerprint_computation(self):
        builder = ManifestBuilder()
        manifest = builder.build_manifest(
            snapshot_date="2026-07-30",
            file_hashes={"file1": "hash1", "file2": "hash2"},
        )
        fp = builder.compute_fingerprint(manifest)
        assert "fingerprint" in fp
        assert len(fp["fingerprint"]) == 64  # SHA256 hex
        assert fp["file_count"] == 2

    def test_fingerprint_deterministic(self):
        builder = ManifestBuilder()
        m1 = builder.build_manifest(snapshot_date="2026-07-30", file_hashes={"a": "1"})
        m2 = builder.build_manifest(snapshot_date="2026-07-30", file_hashes={"a": "1"})
        fp1 = builder.compute_fingerprint(m1)
        fp2 = builder.compute_fingerprint(m2)
        assert fp1["fingerprint"] == fp2["fingerprint"]

    def test_fingerprint_differs_with_different_files(self):
        builder = ManifestBuilder()
        m1 = builder.build_manifest(snapshot_date="2026-07-30", file_hashes={"a": "1"})
        m2 = builder.build_manifest(snapshot_date="2026-07-30", file_hashes={"a": "2"})
        fp1 = builder.compute_fingerprint(m1)
        fp2 = builder.compute_fingerprint(m2)
        assert fp1["fingerprint"] != fp2["fingerprint"]

    def test_save_fingerprint(self):
        builder = ManifestBuilder()
        fp = {"fingerprint": "abc123", "file_count": 5, "dataset_versions": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fingerprint.json"
            builder.save_fingerprint(fp, path)
            assert path.exists()
            with open(path) as f:
                assert json.load(f)["fingerprint"] == "abc123"
