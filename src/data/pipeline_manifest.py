"""Pipeline Lineage Manifest Generator tracking stage inputs, outputs, checksums, and durations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import config


class PipelineManifest:
    """Tracks end-to-end lineage across build stages."""

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or (config.training_dir / "pipeline_manifest_v1.json")
        self.stages: Dict[str, List[Dict[str, Any]]] = {
            "download": [],
            "convert": [],
            "resample": [],
            "alignment": [],
            "feature_builder": [],
            "windowing": []
        }

    def record_stage(
        self,
        stage_name: str,
        inputs: List[str],
        outputs: List[str],
        checksums: List[str],
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record an execution entry for a pipeline stage. Pass timestamp for reproducibility."""
        if stage_name not in self.stages:
            self.stages[stage_name] = []

        entry = {
            "timestamp": timestamp or "unknown",
            "inputs": inputs,
            "outputs": outputs,
            "checksums": checksums,
            "duration_seconds": round(duration_seconds, 4),
            "metadata": metadata or {}
        }
        self.stages[stage_name].append(entry)

    def save(self) -> None:
        """Save lineage manifest to JSON."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.stages, f, indent=2)


pipeline_manifest = PipelineManifest()
