"""Shared utilities for embedding evaluation scripts.

Loads a trained TeacherEncoder + FeatureNormalizer from a real
CheckpointManager run directory, rebuilds datasets through the same
frozen pipeline the trainer uses, and extracts pooled embeddings
via the real extract_embeddings API.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from src.models.teacher.encoder import TeacherEncoder
from src.models.teacher.embeddings import extract_embeddings
from src.training.dataloader import create_dataloader
from src.training.normalizer import FeatureNormalizer
from src.training.trainer import _build_windows_for_symbol, _load_manifest, _date_to_ms
from src.data.market_dataset import MarketDataset


def load_checkpoint_config(run_dir: Path) -> Dict[str, Any]:
    """Load model/optimizer/trainer configs recorded in the run manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    configs = manifest.get("configs", {})
    if not configs:
        raise ValueError(f"manifest.json in {run_dir} has no configs section")
    return configs


def _resolve_checkpoint_path(run_dir: Path, which: str = "best") -> Path:
    latest_path = run_dir / "latest.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        name = latest.get(which) or latest.get("latest")
        if name:
            return run_dir / name
    candidates = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return candidates[-1]


def _resolve_run_dir(run_dir: Path, base: Optional[Path] = None) -> Path:
    """Resolve a checkpoint run dir; if the given dir is missing or has no
    manifest, fall back to the most recent run under the runs base dir
    (default models/foundation/teacher_v1).

    Handles an empty ``--checkpoint`` (e.g. when $CHECKPOINT_DIR was not
    exported by the notebook) gracefully instead of erroring on ".".
    """
    if run_dir.exists() and (run_dir / "manifest.json").exists():
        return run_dir
    base = base or Path("models/foundation/teacher_v1")
    if base.exists():
        candidates = sorted(base.iterdir())
        if candidates:
            resolved = candidates[-1]
            print(f"[eval] No manifest at {run_dir}; using latest run dir {resolved}")
            return resolved
    return run_dir


def load_model_and_normalizer(
    run_dir: Path,
    device: torch.device,
    which: str = "best",
) -> Tuple[TeacherEncoder, FeatureNormalizer, Dict[str, Any]]:
    """Rebuild model + normalizer from a CheckpointManager run directory."""
    run_dir = _resolve_run_dir(run_dir)
    configs = load_checkpoint_config(run_dir)
    model_cfg = configs["model_config"]
    full_model_cfg = {**model_cfg["model"], "loss": model_cfg.get("loss", {})}

    model = TeacherEncoder(full_model_cfg).to(device)
    ckpt_path = _resolve_checkpoint_path(run_dir, which)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    normalizer = FeatureNormalizer(mode=state["normalizer_state"]["mode"])
    normalizer.load_state_dict(state["normalizer_state"])
    return model, normalizer, configs


def build_split_dataset(
    split: str,
    trainer_cfg: Dict[str, Any],
    max_windows: Optional[int] = None,
) -> MarketDataset:
    """Rebuild a dataset for a manifest split through the frozen pipeline.

    Time-split awareness: the "train" split is capped at
    ``time_split.train_end`` and the "test" split starts at or after that
    boundary, so evaluation matches the trainer's causal split exactly.
    Validation uses an unseen symbol and is unbounded in time.
    """
    manifest = _load_manifest()
    symbols = manifest["splits"][split]["symbols"]
    market = trainer_cfg.get("market", "futures")
    stride = (
        trainer_cfg.get("train_window_stride", 16)
        if split == "train"
        else trainer_cfg.get("val_window_stride", 16)
    )
    time_split = manifest["splits"].get("time_split") or {}
    train_end = time_split.get("train_end")
    style = trainer_cfg.get("feature_style", "raw")
    seed = trainer_cfg.get("seed", 42)

    max_end_ms = None
    min_start_ms = None
    if split == "train" and train_end:
        max_end_ms = _date_to_ms(train_end)
    elif split == "test" and train_end:
        min_start_ms = _date_to_ms(train_end)

    windows = []
    for sym in symbols:
        ds = _build_windows_for_symbol(
            sym, market, stride, max_windows,
            max_end_ms=max_end_ms, min_start_ms=min_start_ms,
            feature_style=style, seed=seed,
        )
        windows.extend(ds.windows)
    return MarketDataset(windows)


def extract_split_embeddings(
    model: TeacherEncoder,
    normalizer: FeatureNormalizer,
    split: str,
    pooling: str,
    trainer_cfg: Dict[str, Any],
    device: torch.device,
    max_windows: Optional[int] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Extract normalized pooled embeddings for a full split.

    Normalization matches training: features are transformed with the
    checkpoint's fitted normalizer before entering the model.
    """
    dataset = build_split_dataset(split, trainer_cfg, max_windows)

    class _NormalizedDataset(MarketDataset):
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = super().__getitem__(idx)
            feats = item["features"]
            item["features_raw"] = feats.clone()
            flat = feats.reshape(-1, feats.shape[-1])
            item["features"] = normalizer.transform(flat).reshape(feats.shape)
            return item

    norm_dataset = _NormalizedDataset(dataset.windows)
    loader = create_dataloader(
        norm_dataset, batch_size=batch_size, shuffle=False,
        seed=trainer_cfg.get("seed", 42),
    )
    return extract_embeddings(model, loader, pooling, device)
