"""Snapshot manager creating immutable reproducible dataset snapshots."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.config import config
from src.data.db import db_manager


def _compute_snapshot_content_hash(manifest: dict, checksums: dict, stats: dict) -> str:
    """Compute a deterministic hash of all snapshot contents."""
    payload = {
        "manifest": manifest,
        "checksums": checksums,
        "stats": stats,
    }
    raw = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SnapshotManager:
    """Manages immutable snapshots in storage/training/snapshots/<date>/."""

    def __init__(self, training_dir: Optional[Path] = None):
        self.training_dir = training_dir or config.training_dir
        self.snapshots_dir = self.training_dir / "snapshots"

    def snapshot_exists(self, snapshot_path: Path) -> bool:
        """Check if a snapshot already exists for this date."""
        return snapshot_path.exists() and (snapshot_path / "content_hash").exists()

    def create_snapshot(
        self,
        snapshot_date: str = "2026-07-30",
        manifest_data: Optional[Dict[str, Any]] = None,
        checksums_data: Optional[Dict[str, Any]] = None,
        stats_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create a new immutable snapshot directory.

        Raises FileExistsError if a snapshot with this date already exists.
        """
        snapshot_path = self.snapshots_dir / snapshot_date

        if self.snapshot_exists(snapshot_path):
            raise FileExistsError(
                f"Snapshot for date {snapshot_date} already exists at {snapshot_path}. "
                "Use a new date or remove the existing snapshot to recreate it."
            )

        # Build real checksums from the index
        if checksums_data is None or "status" in checksums_data:
            files = db_manager.query_files()
            checksums_data = {
                "file_count": len(files),
                "files": {f["file_id"]: f["sha256"] for f in files},
            }

        snapshot_path.mkdir(parents=True, exist_ok=True)

        if manifest_data:
            with open(snapshot_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

        if checksums_data:
            with open(snapshot_path / "checksums.json", "w", encoding="utf-8") as f:
                json.dump(checksums_data, f, indent=2)

        if stats_data:
            with open(snapshot_path / "stats.json", "w", encoding="utf-8") as f:
                json.dump(stats_data, f, indent=2)

        # Write content hash for immutability verification
        content_hash = _compute_snapshot_content_hash(
            manifest_data or {},
            checksums_data or {},
            stats_data or {},
        )
        with open(snapshot_path / "content_hash", "w", encoding="utf-8") as f:
            f.write(content_hash)

        return snapshot_path


snapshot_manager = SnapshotManager()
