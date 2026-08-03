"""Versioned checkpoint saver/loader with manifest and resume support.

``async_writes`` (default False) offloads the slow filesystem write of each
checkpoint to a background thread. This is important on Google Colab where the
run directory is symlinked to Drive: a synchronous ~360MB write per checkpoint
can stall training for tens of seconds and trigger Drive-API throttling. The
state is serialized in the caller's thread (a consistent snapshot), so the
background thread only writes bytes and small JSON, with atomic temp+rename so
``latest.json``/``manifest.json`` never reference a partially-written file.
"""

import hashlib
import io
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import torch


class CheckpointManager:
    """Manages versioned checkpoints under models/foundation/teacher_v1/<run_id>/."""

    def __init__(self, run_dir: Path, configs: Dict[str, Any], async_writes: bool = False):
        self.run_dir = run_dir
        self.configs = configs
        self.async_writes = async_writes
        self.history: list = []
        self._queue: Optional[queue.Queue] = None
        self._writer: Optional[threading.Thread] = None
        self._last_error: Optional[BaseException] = None
        self._load_existing_history()

    def _load_existing_history(self):
        """Load prior manifest history when resuming a run, so new saves append."""
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self.history = manifest.get("checkpoints", [])
            except Exception:
                self.history = []

    def _ensure_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_git_commit(self) -> str:
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------ writer
    def _ensure_writer(self):
        if self._queue is None:
            self._queue = queue.Queue()
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

    def _writer_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                break
            try:
                self._write_all(*job)
            except BaseException as e:  # never let Drive I/O kill the worker
                self._last_error = e
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 600.0):
        """Block until all enqueued checkpoint writes are durable."""
        if self._queue is not None:
            self._queue.join()
            if self._last_error is not None:
                err = self._last_error
                self._last_error = None
                raise err

    def close(self):
        """Flush pending writes and stop the writer thread."""
        if self._queue is not None:
            try:
                self.flush()
            finally:
                self._queue.put(None)
                self._writer.join(timeout=300.0)
                self._queue = None
                self._writer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _atomic_write(path: Path, data: bytes):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _write_all(self, path, data, best_path, manifest, latest):
        self._atomic_write(path, data)
        if best_path is not None:
            self._atomic_write(best_path, data)
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.run_dir / "latest.json").write_text(json.dumps(latest, indent=2))

    # ------------------------------------------------------------------ save
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
        metrics: Optional[Dict[str, Any]] = None,
        val_mask_generator_state: Any = None,
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
            "val_mask_generator_state": val_mask_generator_state,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics,
        }
        # Serialize in the caller's thread: a consistent snapshot of the tensors
        # (no mutation while the background thread writes bytes).
        buf = io.BytesIO()
        torch.save(state, buf)
        data = buf.getvalue()
        ckpt_hash = hashlib.sha256(data).hexdigest()[:16]

        path = self.run_dir / f"checkpoint_epoch{epoch}.pt"
        history_entry = {
            "epoch": epoch,
            "step": step,
            "path": str(path.name),
            "sha256": ckpt_hash,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if metrics:
            history_entry["metrics"] = metrics
        self.history.append(history_entry)

        if is_best:
            best_path = self.run_dir / "best.pt"
            latest = {"best": str(path.name), "latest": str(path.name)}
        else:
            best_path = None
            latest = {"latest": str(path.name)}
            if (self.run_dir / "best.pt").exists():
                latest["best"] = "best.pt"

        manifest = {
            "run_id": self.run_dir.name,
            "configs": self.configs,
            "git_commit": self._resolve_git_commit(),
            "checkpoints": self.history,
        }

        if self.async_writes:
            self._ensure_writer()
            self._queue.put((path, data, best_path, manifest, latest))
            # Best / final checkpoints are flushed synchronously so the
            # deliverable model is always durable before training continues.
            if is_best:
                self.flush()
        else:
            self._write_all(path, data, best_path, manifest, latest)

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
        return (
            state["epoch"],
            state["step"],
            state.get("normalizer_state"),
            state.get("mask_generator_state"),
            state.get("train_loss"),
            state.get("val_loss"),
            state.get("val_mask_generator_state"),
        )
