"""Unit tests for Experiment Registry."""

import tempfile
from pathlib import Path
import duckdb
import pytest
from src.training.experiment_registry import ExperimentRegistry


class TestExperimentRegistry:
    def test_log_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "experiments.duckdb"
            er = ExperimentRegistry(db_path=db_path)
            er.log_experiment(
                experiment_id="exp_001", snapshot="2026-07-30",
                alignment_version="v1", feature_builder_version="v1",
                windowing_version="v1", modality_registry="v1",
                objective="ssl", encoder="transformer", loss=0.5,
            )
            results = er.query_experiments()
            assert len(results) == 1
            assert results[0]["experiment_id"] == "exp_001"

    def test_duplicate_experiment_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "experiments.duckdb"
            er = ExperimentRegistry(db_path=db_path)
            er.log_experiment(
                experiment_id="exp_001", snapshot="2026-07-30",
                alignment_version="v1", feature_builder_version="v1",
                windowing_version="v1", modality_registry="v1",
                objective="ssl", encoder="transformer", loss=0.5,
            )
            with pytest.raises(duckdb.ConstraintException):
                er.log_experiment(
                    experiment_id="exp_001", snapshot="2026-07-30",
                    alignment_version="v1", feature_builder_version="v1",
                    windowing_version="v1", modality_registry="v1",
                    objective="ssl", encoder="transformer", loss=0.5,
                )

    def test_query_all_fields_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "experiments.duckdb"
            er = ExperimentRegistry(db_path=db_path)
            er.log_experiment(
                experiment_id="exp_002", snapshot="2026-07-30",
                alignment_version="alignment_v1.yaml", feature_builder_version="v1",
                windowing_version="windowing_v1.yaml", modality_registry="modalities_v1.yaml",
                objective="ssl", encoder="transformer", loss=0.123,
                seed=42, git_commit="abc123", hardware="GPU",
                software="PyTorch 2.1", metrics='{"loss": 0.123}',
            )
            results = er.query_experiments()
            assert len(results) == 1
            r = results[0]
            assert r["seed"] == 42
            assert r["hardware"] == "GPU"
            assert r["software"] == "PyTorch 2.1"
