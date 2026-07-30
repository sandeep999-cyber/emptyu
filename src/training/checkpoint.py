"""Versioned checkpoint saver/loader with manifest and resume support."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import torch


class CheckpointManager:
    """Manages versioned checkpoints under models/foundation/teacher_v1/<run_id>/."""

    def __init__(self, run_dir: Path, configs: Dict[str, Any]):
        self.run_dir = run_dir
        self.configs = configs
        self.history: list = []

    def _ensure_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _resolve_git_commit(self) -> str:
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    def save(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        normalizer_state: Dict[str, Any],
        mask_generator_state: Any,
        train_loss: Optional[float],
        val_loss: Optional[float],
        is_best: bool = False,
    ):
        self._ensure_dir()

        state = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "normalizer_state": normalizer_state,
            "mask_generator_state": mask_generator_state,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        path = self.run_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(state, path)
        ckpt_hash = self._compute_hash(path)

        # Update manifest
        self.history.append({
            "epoch": epoch,
            "step": step,
            "path": str(path.name),
            "sha256": ckpt_hash,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        self._write_manifest()

        if is_best:
            best_path = self.run_dir / "best.pt"
            shutil.copy2(path, best_path)
            latest = {"best": str(path.name), "latest": str(path.name)}
        else:
            latest = {"latest": str(path.name)}
            if (self.run_dir / "best.pt").exists():
                latest["best"] = "best.pt"
        self._write_latest(latest)

    def _write_manifest(self):
        manifest = {
            "run_id": self.run_dir.name,
            "configs": self.configs,
            "git_commit": self._resolve_git_commit(),
            "checkpoints": self.history,
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _write_latest(self, latest: dict):
        (self.run_dir / "latest.json").write_text(json.dumps(latest, indent=2))

    def load_latest_checkpoint(self, model, optimizer=None, scheduler=None):
        latest_path = self.run_dir / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest.json found in {self.run_dir}")
        latest = json.loads(latest_path.read_text())
        ckpt_name = latest.get("latest") or latest.get("best")
        ckpt_path = self.run_dir / ckpt_name
        return self._load(ckpt_path, model, optimizer, scheduler)

    def _load(self, path: Path, model, optimizer, scheduler):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
        if optimizer and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        return state["epoch"], state["step"], state.get("normalizer_state"), state.get("mask_generator_state"), state.get("train_loss"), state.get("val_loss")
