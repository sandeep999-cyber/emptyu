"""Experiment Registry Manager logging run metadata to experiment_registry.duckdb."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
from src.config import config


class ExperimentRegistry:
    """Manages experiment_registry.duckdb for tracking training runs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.experiment_db_path
        self._table_initialized = False

    def _ensure_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        self._ensure_dir()
        conn = duckdb.connect(str(self.db_path))
        if not self._table_initialized:
            self._init_table(conn)
            self._table_initialized = True
        return conn

    def _init_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_registry (
                experiment_id VARCHAR PRIMARY KEY,
                snapshot VARCHAR,
                alignment_version VARCHAR,
                feature_builder_version VARCHAR,
                windowing_version VARCHAR,
                modality_registry VARCHAR,
                objective VARCHAR,
                encoder VARCHAR,
                loss DOUBLE,
                seed INTEGER,
                git_commit VARCHAR,
                hardware VARCHAR,
                software VARCHAR,
                metrics VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def log_experiment(
        self,
        experiment_id: str,
        snapshot: str,
        alignment_version: str,
        feature_builder_version: str,
        windowing_version: str,
        modality_registry: str,
        objective: str,
        encoder: str,
        loss: float,
        seed: int = 42,
        git_commit: str = "main",
        hardware: str = "CPU/GPU",
        software: str = "PyTorch",
        metrics: str = "{}"
    ) -> None:
        """Log a training experiment record. Raises if experiment_id already exists."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO experiment_registry (
                    experiment_id, snapshot, alignment_version, feature_builder_version,
                    windowing_version, modality_registry, objective, encoder, loss,
                    seed, git_commit, hardware, software, metrics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                experiment_id, snapshot, alignment_version, feature_builder_version,
                windowing_version, modality_registry, objective, encoder, loss,
                seed, git_commit, hardware, software, metrics
            ])
        finally:
            conn.close()

    def query_experiments(self) -> List[Dict[str, Any]]:
        """Query all logged experiment records."""
        conn = self._get_connection()
        try:
            res = conn.execute("SELECT * FROM experiment_registry ORDER BY created_at DESC").fetchall()
            cols = [desc[0] for desc in conn.description]
            return [dict(zip(cols, row)) for row in res]
        finally:
            conn.close()


experiment_registry = ExperimentRegistry()
