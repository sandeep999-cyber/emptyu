"""Teacher training loop."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime
import json
import time
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
import psutil

from src.config import config as cfg
from src.data.lake import lake
from src.data.feature_builder import feature_builder
from src.data.windowing import WindowingEngine
from src.data.market_dataset import MarketDataset
from src.training.dataloader import create_dataloader
from src.training.sampler import EpochMarketSampler
from src.training.normalizer import FeatureNormalizer
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.checkpoint import CheckpointManager
from src.training.losses.masked_modeling import MaskGenerator, MaskedMarketModelingLoss
from src.models.teacher.encoder import TeacherEncoder
from src.logger import get_stage_logger


MANIFEST_PATH = cfg.training_dir / "training_manifest_v1.json"


def _load_manifest() -> dict:
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)["training_manifest"]


def _date_to_ms(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _build_windows_for_symbol(
    symbol: str,
    market: str,
    stride: int,
    max_windows: Optional[int] = None,
    max_end_ms: Optional[int] = None,
    min_start_ms: Optional[int] = None,
    feature_style: str = "raw",
    seed: int = 42,
) -> MarketDataset:
    df_state = lake.market_state(symbol, market=market)
    if df_state.empty:
        return MarketDataset([])
    feats, fm, ts = feature_builder.build_features(df_state, style=feature_style)
    metadata = {
        "symbol": symbol,
        "snapshot_id": cfg.dataset.get("snapshot_date", "2026-07-30"),
        "market": market,
    }
    # Build directly at the requested stride (avoid materializing all stride-1
    # windows in RAM first, which is ~500k windows for a full-year symbol).
    engine = WindowingEngine(
        {
            "sequence_length": cfg.windowing.get("sequence_length", 512),
            "stride": max(1, stride),
            "drop_incomplete_windows": cfg.windowing.get("drop_incomplete_windows", True),
            "max_gap_ms": cfg.windowing.get("max_gap_ms", 300000),
        }
    )
    selected = engine.create_windows(feats, fm, ts, metadata=metadata)
    if max_end_ms is not None:
        selected = [w for w in selected if w["metadata"]["window_end_ms"] < max_end_ms]
    if min_start_ms is not None:
        selected = [w for w in selected if w["metadata"]["window_start_ms"] >= min_start_ms]
    if max_windows is not None and len(selected) > max_windows:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(selected), size=max_windows, replace=False).tolist())
        selected = [selected[i] for i in idx]
    return MarketDataset(selected)


class TeacherTrainer:
    def __init__(self, model_cfg: Dict, opt_cfg: Dict, trainer_cfg: Dict, run_dir: Path, resume_dir: Optional[Path] = None):
        self.model_cfg = model_cfg
        self.opt_cfg = opt_cfg
        self.trainer_cfg = trainer_cfg
        self.run_dir = run_dir
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.start_epoch = 0
        self.start_step = 0
        self.best_val_loss = float("inf")
        self.log = get_stage_logger("train_teacher")
        self.device = self._resolve_device()
        self.log.info(f"Device: {self.device}")

        # Load frozen manifest splits
        manifest = _load_manifest()
        splits = manifest["splits"]
        self.train_symbols = splits["train"]["symbols"]
        self.val_symbols = splits["validation"]["symbols"]
        self.test_symbols = splits.get("test", {}).get("symbols", [])
        time_split = splits.get("time_split") or {}
        train_end = time_split.get("train_end")
        self.train_end_ms = _date_to_ms(train_end) if train_end else None
        self.log.info(f"Train symbols: {self.train_symbols}, Val symbols: {self.val_symbols}")
        self.log.info(f"Test symbols: {self.test_symbols}, train_end: {train_end}")

        self.feature_style = self.trainer_cfg.get("feature_style", "raw")
        self.log.info(f"Feature style: {self.feature_style}")

        self.train_dataset, self.val_dataset = self._build_datasets()
        self.log.info(f"Train windows: {len(self.train_dataset)}")
        self.log.info(f"Val windows: {len(self.val_dataset)}")
        if len(self.train_dataset) == 0:
            raise RuntimeError("No training windows available; cannot train (check manifest splits and data).")
        if len(self.val_dataset) == 0:
            raise RuntimeError("No validation windows available; cannot train (check manifest splits and data).")

        self.normalizer = self._fit_normalizer()
        self.log.info(f"Normalizer fitted (mode={self.normalizer.mode})")

        full_model_cfg = {**model_cfg["model"], "loss": model_cfg.get("loss", {})}
        self.model = TeacherEncoder(full_model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, opt_cfg)
        total_steps = max(1, len(self.train_dataset) // max(1, trainer_cfg["batch_size"]) * trainer_cfg["epochs"])
        self.scheduler = build_scheduler(self.optimizer, opt_cfg, total_steps)
        masked_cfg = model_cfg.get("loss", {}).get("masked_modeling", {})
        self.mask_generator = MaskGenerator(
            mask_ratio=masked_cfg.get("mask_ratio", 0.15),
            seed=trainer_cfg.get("seed", 42),
            mask_mode=masked_cfg.get("mask_mode", "random"),
            span_len=masked_cfg.get("span_len", 16),
            device=self.device,
        )
        # Deterministic validation masks: separate generator with a fixed seed so
        # every epoch validates against the identical mask pattern.
        self.val_mask_generator = MaskGenerator(
            mask_ratio=masked_cfg.get("mask_ratio", 0.15),
            seed=(trainer_cfg.get("seed", 42) + 10**6) & 0xFFFFFFFF,
            mask_mode=masked_cfg.get("mask_mode", "random"),
            span_len=masked_cfg.get("span_len", 16),
            device=self.device,
        )
        self.loss_fn = MaskedMarketModelingLoss(
            loss_cfg=masked_cfg,
            d_model=model_cfg["model"]["d_model"],
            device=self.device,
        )
        self.grad_clip = opt_cfg["optimizer"]["grad_clip"]
        self.num_workers = self.trainer_cfg.get("num_workers", 0)
        self.pin_memory = self.device.type == "cuda"
        self.persistent_workers = self.num_workers > 0
        self.amp_dtype = self._resolve_amp_dtype()
        self.use_amp = self.amp_dtype is not None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # bf16 is only tensor-core-accelerated on Ampere+ (sm_80); on older GPUs
        # (e.g. Colab T4 / Turing sm_75) bf16 runs emulated, so fall back to fp16.
        if self.use_amp:
            self.log.info(f"AMP dtype: {self.amp_dtype}")
        # Micro-batching with gradient accumulation: `batch_size` stays the
        # effective batch (optimizer step + scheduler step + LR curve), while
        # `micro_batch_size` bounds per-step VRAM. Default = no accumulation.
        eff_batch = int(trainer_cfg["batch_size"])
        self.micro_batch_size = min(eff_batch, int(trainer_cfg.get("micro_batch_size", eff_batch)))
        self.grad_accum = max(1, eff_batch // max(1, self.micro_batch_size))
        self.log.info(
            f"Micro-batch={self.micro_batch_size} grad_accum={self.grad_accum} "
            f"(effective batch={self.micro_batch_size * self.grad_accum})"
        )
        self.checkpoint_mgr = CheckpointManager(run_dir, {
            "model_config": model_cfg,
            "optimizer_config": opt_cfg,
            "trainer_config": trainer_cfg,
        })
        self._restore_from_resume()
        self.log.info(f"TeacherTrainer initialized (num_workers={self.num_workers}).")

    def _resolve_amp_dtype(self):
        """Pick an autocast dtype suited to the GPU: bf16 on Ampere+, fp16 on
        sm_53+, None (no AMP) on older GPUs that lack fp16 compute."""
        if not (bool(self.trainer_cfg.get("mixed_precision", False)) and self.device.type == "cuda"):
            return None
        try:
            major, minor = torch.cuda.get_device_capability(self.device)
        except Exception:
            major, minor = 0, 0
        if major >= 8:
            return torch.bfloat16
        if (major, minor) >= (5, 3):
            return torch.float16
        return None

    def _restore_from_resume(self):
        """Restore model/optimizer/scheduler/normalizer state when resuming a run."""
        if not self.resume_dir or not (self.resume_dir / "latest.json").exists():
            if self.resume_dir:
                self.log.warning(f"Resume dir {self.resume_dir} has no latest.json; starting fresh.")
            return
        resume_mgr = CheckpointManager(self.resume_dir, {})
        try:
            (
                epoch,
                step,
                normalizer_state,
                mask_generator_state,
                _,
                _,
                val_mask_generator_state,
            ) = resume_mgr.load_latest_checkpoint(
                self.model, self.optimizer, self.scheduler
            )
        except Exception as e:
            self.log.warning(f"Failed to load resume checkpoint from {self.resume_dir}: {e}; starting fresh.")
            return
        if normalizer_state:
            self.normalizer.load_state_dict(normalizer_state)
        if mask_generator_state is not None:
            self.mask_generator.set_state(mask_generator_state)
        if val_mask_generator_state is not None:
            self.val_mask_generator.set_state(val_mask_generator_state)
        self.start_epoch = int(epoch)
        self.start_step = int(step)
        if self.resume_dir != self.run_dir:
            self.checkpoint_mgr._load_existing_history()
        self.log.info(f"Resumed from {self.resume_dir} at epoch {self.start_epoch}, step {self.start_step}.")

    def _resolve_device(self) -> torch.device:
        device_str = self.trainer_cfg.get("device", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_datasets(self) -> Tuple[MarketDataset, MarketDataset]:
        market = self.trainer_cfg.get("market", "futures")
        train_stride = self.trainer_cfg.get("train_window_stride", 16)
        val_stride = self.trainer_cfg.get("val_window_stride", 16)
        max_train = self.trainer_cfg.get("max_train_windows")
        max_val = self.trainer_cfg.get("max_val_windows")

        train_windows = []
        for sym in self.train_symbols:
            ds = _build_windows_for_symbol(
                sym, market, train_stride, max_train,
                max_end_ms=self.train_end_ms, feature_style=self.feature_style,
                seed=self.trainer_cfg.get("seed", 42),
            )
            train_windows.extend(ds.windows)

        val_windows = []
        for sym in self.val_symbols:
            ds = _build_windows_for_symbol(
                sym, market, val_stride, max_val,
                feature_style=self.feature_style,
                seed=self.trainer_cfg.get("seed", 42),
            )
            val_windows.extend(ds.windows)

        return MarketDataset(train_windows), MarketDataset(val_windows)

    def _fit_normalizer(self) -> FeatureNormalizer:
        all_feats = []
        all_masks = []
        for sym in self.train_symbols:
            market = self.trainer_cfg.get("market", "futures")
            df = lake.market_state(sym, market=market, end_ts=self.train_end_ms)
            if df.empty:
                continue
            feats, fm, _ = feature_builder.build_features(df, style=self.feature_style)
            all_feats.append(feats)
            all_masks.append(fm)
        if not all_feats:
            raise RuntimeError("No training data available for normalizer fitting.")
        train_features = np.concatenate(all_feats, axis=0)
        train_mask = np.concatenate(all_masks, axis=0)
        normalizer = FeatureNormalizer(mode=self.trainer_cfg.get("normalizer", {}).get("mode", "zscore"))
        normalizer.fit(
            torch.from_numpy(train_features),
            torch.from_numpy(train_mask),
            {"train": {"symbols": self.train_symbols}},
        )
        return normalizer

    def _normalize_batch(self, features: torch.Tensor, feature_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = features.shape
        flat = features.reshape(-1, D)
        flat_norm = self.normalizer.transform(flat)
        features_norm = flat_norm.reshape(B, T, D)
        return features_norm.to(self.device), features.to(self.device)

    def train(self):
        self.log.info("Starting training.")
        self.model.train()
        batch_size = self.micro_batch_size
        sampler = EpochMarketSampler(
            len(self.train_dataset), shuffle=True, seed=self.trainer_cfg.get("seed", 42)
        )
        dataloader = create_dataloader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers,
            seed=self.trainer_cfg.get("seed", 42),
            sampler=sampler,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        val_loader = create_dataloader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers,
            seed=self.trainer_cfg.get("seed", 42),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        global_step = self.start_step
        best_val_loss = self.best_val_loss
        last_val_loss = None
        tokens_per_epoch = len(self.train_dataset) * self.model_cfg["model"]["context_length"]
        val_every = max(1, int(self.trainer_cfg.get("val_every", 1)))
        checkpoint_every = max(1, int(self.trainer_cfg.get("checkpoint_every", 1)))

        for epoch in range(self.start_epoch + 1, self.trainer_cfg["epochs"] + 1):
            sampler.set_epoch(epoch)
            epoch_start = time.time()
            self.model.train()
            epoch_losses = []
            epoch_group_sums: Dict[str, float] = {}
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_n = 0

            for i, batch in enumerate(dataloader):
                features = batch["features"]
                feature_mask = batch["feature_mask"]
                timestamps = batch["timestamps"]
                mask = batch["mask"]

                features_norm, features_raw = self._normalize_batch(features, feature_mask)

                data_mask = mask.to(self.device)
                masked_positions = self.mask_generator(data_mask)

                corrupted = features_norm.clone()
                corrupted = corrupted.masked_fill(masked_positions.unsqueeze(-1), 0.0)

                autocast = (
                    torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                    if self.use_amp
                    else nullcontext()
                )
                with autocast:
                    latent, kpm, positions, t_data = self.model(corrupted, timestamps.to(self.device), data_mask)
                    data_latent = latent[:, 1:, :]
                    reconstruction = self.model.reconstruct(data_latent)
                if self.use_amp:
                    reconstruction = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in reconstruction.items()}

                losses = self.loss_fn(
                    reconstruction,
                    features_norm,
                    features_raw,
                    feature_mask.to(self.device),
                    masked_positions,
                )
                loss = losses["total"]

                self.scaler.scale(loss).backward()
                accum_loss += float(loss.item())
                accum_n += 1
                for k, v in losses.items():
                    if k == "total":
                        continue
                    epoch_group_sums[k] = epoch_group_sums.get(k, 0.0) + v.item() * features.shape[0]

                # Optimizer step at each accumulation boundary (and at epoch end
                # for partial windows). Scheduler steps once per optimizer step,
                # so the effective batch and LR curve match `batch_size`.
                is_accum_last = (i + 1) % self.grad_accum == 0 or (i + 1) == len(dataloader)
                if not is_accum_last:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                step_loss = accum_loss / max(1, accum_n)
                accum_loss = 0.0
                accum_n = 0
                self.optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(step_loss)
                global_step += 1

                if global_step % self.trainer_cfg.get("log_every", 50) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    rss_mb = psutil.Process().memory_info().rss / 1e6
                    self.log.info(
                        f"E{epoch} S{global_step} loss={step_loss:.4f} lr={lr:.2e} "
                        f"gn={grad_norm:.2f} rss={rss_mb:.0f}MB"
                    )

            epoch_time = time.time() - epoch_start
            avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            train_group = {k: v / max(1, len(epoch_losses)) for k, v in epoch_group_sums.items()}
            should_validate = epoch % val_every == 0 or epoch == self.trainer_cfg["epochs"]
            if should_validate:
                val_loss, val_group = self._validate(val_loader)
                last_val_loss = val_loss
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
            else:
                val_loss = last_val_loss
                val_group = {}
                is_best = False

            should_checkpoint = (
                should_validate and (epoch % checkpoint_every == 0)
            ) or epoch == self.trainer_cfg["epochs"] or is_best
            if should_checkpoint:
                self.checkpoint_mgr.save(
                    epoch=epoch,
                    step=global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    normalizer_state=self.normalizer.state_dict(),
                    mask_generator_state=self.mask_generator.get_state(),
                    val_mask_generator_state=self.val_mask_generator.get_state(),
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    is_best=is_best,
                    metrics={"train": {"total": avg_train_loss, **train_group}, "val": {"total": val_loss, **val_group}},
                )
            val_display = f"{val_loss:.4f}" if val_loss is not None else "skipped"
            self.log.info(
                f"Epoch {epoch}/{self.trainer_cfg['epochs']} "
                f"train_loss={avg_train_loss:.4f} val_loss={val_display} "
                f"time={epoch_time:.1f}s best={is_best}"
            )
            self.log.info(f"  train groups: {train_group}")
            self.log.info(f"  val groups: {val_group}")

        self.log.info("Training complete.")

    def _validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        group_sums: Dict[str, float] = {}
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"]
                feature_mask = batch["feature_mask"]
                timestamps = batch["timestamps"]
                mask = batch["mask"]

                features_norm, features_raw = self._normalize_batch(features, feature_mask)
                data_mask = mask.to(self.device)
                masked_positions = self.val_mask_generator(data_mask)
                corrupted = features_norm.clone()
                corrupted = corrupted.masked_fill(masked_positions.unsqueeze(-1), 0.0)

                latent, kpm, positions, t_data = self.model(corrupted, timestamps.to(self.device), data_mask)
                data_latent = latent[:, 1:, :]
                reconstruction = self.model.reconstruct(data_latent)

                losses = self.loss_fn(
                    reconstruction,
                    features_norm,
                    features_raw,
                    feature_mask.to(self.device),
                    masked_positions,
                )
                total_loss += losses["total"].item() * features.shape[0]
                for k, v in losses.items():
                    if k == "total":
                        continue
                    group_sums[k] = group_sums.get(k, 0.0) + v.item() * features.shape[0]
                count += features.shape[0]

        self.model.train()
        if count == 0:
            return 0.0, {}
        return total_loss / count, {k: v / count for k, v in group_sums.items()}
