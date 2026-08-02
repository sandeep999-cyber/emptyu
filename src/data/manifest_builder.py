"""Manifest Builder for training_manifest_v1.json and dataset_fingerprint.json."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import config
from src.data.db import db_manager


class ManifestBuilder:
    """Generates version-locked training manifests with embedded file hash ledger."""

    def build_manifest(
        self,
        snapshot_date: str = "2026-07-30",
        train_symbols: Optional[List[str]] = None,
        val_symbols: Optional[List[str]] = None,
        test_symbols: Optional[List[str]] = None,
        train_end_date: str = "2024-11-30",
        seed: int = 42,
        file_hashes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build training_manifest_v1.json dictionary locking all artifact versions and file hashes."""
        train_symbols = train_symbols or ["BTCUSDT", "ETHUSDT"]
        val_symbols = val_symbols or ["SOLUSDT"]
        test_symbols = test_symbols or ["BTCUSDT", "ETHUSDT"]

        manifest = {
            "training_manifest": {
                "dataset": {
                    "snapshot": snapshot_date,
                    "alignment_version": "alignment_v1.yaml",
                    "feature_builder_version": "v1",
                    "windowing_version": "windowing_v1.yaml",
                    "canonical_schema_version": "v1",
                    "market_state_schema_version": "v1",
                    "modality_registry": "modalities_v1.yaml",
                },
                "splits": {
                    "train": {
                        "symbols": train_symbols,
                    },
                    "validation": {
                        "symbols": val_symbols,
                    },
                    "test": {
                        "symbols": test_symbols,
                    },
                    "time_split": {
                        "train_end": train_end_date,
                    },
                },
                "random_seed": {
                    "python": seed,
                    "numpy": seed,
                    "torch": seed,
                },
                "file_ledger": file_hashes or {},
            }
        }
        return manifest

    def build_manifest_from_index(
        self,
        snapshot_date: str = "2026-07-30",
        train_symbols: Optional[List[str]] = None,
        val_symbols: Optional[List[str]] = None,
        test_symbols: Optional[List[str]] = None,
        train_end_date: str = "2024-11-30",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Build manifest with file hashes fetched from the database index (filtered by snapshot symbols)."""
        all_symbols = (train_symbols or []) + (val_symbols or []) + (test_symbols or [])
        files = []
        for sym in set(all_symbols):
            files.extend(db_manager.query_files(symbol=sym))

        file_hashes = {}
        for f in files:
            file_hashes[f["file_id"]] = f["sha256"]

        return self.build_manifest(
            snapshot_date=snapshot_date,
            train_symbols=train_symbols,
            val_symbols=val_symbols,
            test_symbols=test_symbols,
            train_end_date=train_end_date,
            seed=seed,
            file_hashes=file_hashes,
        )

    def save_manifest(self, manifest: Dict[str, Any], path: Path) -> None:
        """Write manifest to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def compute_fingerprint(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a deterministic dataset fingerprint from the manifest."""
        ledger = manifest.get("training_manifest", {}).get("file_ledger", {})
        sorted_ledger = json.dumps(dict(sorted(ledger.items())), sort_keys=True)
        dataset = manifest.get("training_manifest", {}).get("dataset", {})
        sorted_versions = json.dumps(dict(sorted(dataset.items())), sort_keys=True)
        payload = f"{sorted_ledger}|{sorted_versions}"
        fingerprint_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return {
            "fingerprint": fingerprint_hash,
            "file_count": len(ledger),
            "dataset_versions": dataset,
        }

    def save_fingerprint(self, fingerprint: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)


manifest_builder = ManifestBuilder()
