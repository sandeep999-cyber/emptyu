"""Teacher training loop."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import psutil
import platform

from src.config import config as cfg
from src.data.lake import lake
from src.data.feature_builder import feature_builder
from src.data.windowing import windowing_engine
from src.data.market_dataset import MarketDataset
from src.training.dataloader import create_dataloader
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


def _build_windows_for_symbol(symbol: str, market: str, stride: int, max_windows: Optional[int] = None) -> MarketDataset:
    df_state = lake.market_state(symbol, market=market)
    if df_state.empty:
        return MarketDataset([])
    feats, fm, ts = feature_builder.build_features(df_state)
    metadata = {
        "symbol": symbol,
        "snapshot_id": cfg.dataset.get("snapshot_date", "2026-07-30"),
        "market": market,
    }
    all_windows = windowing_engine.create_windows(feats, fm, ts, metadata=metadata)
    selected = all_windows[::stride] if stride > 1 else all_windows
    if max_windows is not None and len(selected) > max_windows:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(selected), size=max_windows, replace=False).tolist())
        selected = [selected[i] for i in idx]
    return MarketDataset(selected)


class TeacherTrainer:
    def __init__(self, model_cfg: Dict, opt_cfg: Dict, trainer_cfg: Dict, run_dir: Path):
        self.model_cfg = model_cfg
        self.opt_cfg = opt_cfg
        self.trainer_cfg = trainer_cfg
        self.run_dir = run_dir
        self.log = get_stage_logger("train_teacher")
        self.device = self._resolve_device()
        self.log.info(f"Device: {self.device}")

        # Load frozen manifest splits
        manifest = _load_manifest()
        splits = manifest["splits"]
        self.train_symbols = splits["train"]["symbols"]
        self.val_symbols = splits["validation"]["symbols"]
        self.log.info(f"Train symbols: {self.train_symbols}, Val symbols: {self.val_symbols}")

        self.train_dataset, self.val_dataset = self._build_datasets()
        self.log.info(f"Train windows: {len(self.train_dataset)}")
        self.log.info(f"Val windows: {len(self.val_dataset)}")

        self.normalizer = self._fit_normalizer()
        self.log.info(f"Normalizer fitted (mode={self.normalizer.mode})")

        full_model_cfg = {**model_cfg["model"], "loss": model_cfg.get("loss", {})}
        self.model = TeacherEncoder(full_model_cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, opt_cfg)
        total_steps = max(1, len(self.train_dataset) // max(1, trainer_cfg["batch_size"]) * trainer_cfg["epochs"])
        self.scheduler = build_scheduler(self.optimizer, opt_cfg, total_steps)
        self.mask_generator = MaskGenerator(
            mask_ratio=model_cfg.get("loss", {}).get("masked_modeling", {}).get("mask_ratio", 0.15),
            seed=trainer_cfg.get("seed", 42),
        )
        self.loss_fn = MaskedMarketModelingLoss(
            loss_cfg=model_cfg.get("loss", {}).get("masked_modeling", {}),
            d_model=model_cfg["model"]["d_model"],
            device=self.device,
        )
        self.grad_clip = opt_cfg["optimizer"]["grad_clip"]
        self.num_workers = self.trainer_cfg.get("num_workers", 0)
        self.checkpoint_mgr = CheckpointManager(run_dir, {
            "model_config": model_cfg,
            "optimizer_config": opt_cfg,
            "trainer_config": trainer_cfg,
        })
        self.log.info(f"TeacherTrainer initialized (num_workers={self.num_workers}).")

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
            ds = _build_windows_for_symbol(sym, market, train_stride, max_train)
            train_windows.extend(ds.windows)

        val_windows = []
        for sym in self.val_symbols:
            ds = _build_windows_for_symbol(sym, market, val_stride, max_val)
            val_windows.extend(ds.windows)

        return MarketDataset(train_windows), MarketDataset(val_windows)

    def _fit_normalizer(self) -> FeatureNormalizer:
        all_feats = []
        all_masks = []
        for sym in self.train_symbols:
            market = self.trainer_cfg.get("market", "futures")
            df = lake.market_state(sym, market=market)
            if df.empty:
                continue
            feats, fm, _ = feature_builder.build_features(df)
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
        batch_size = self.trainer_cfg["batch_size"]
        dataloader = create_dataloader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers,
            seed=self.trainer_cfg.get("seed", 42),
        )
        val_loader = create_dataloader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers,
            seed=self.trainer_cfg.get("seed", 42),
        )

        global_step = 0
        best_val_loss = float("inf")
        tokens_per_epoch = len(self.train_dataset) * self.model_cfg["model"]["context_length"]

        for epoch in range(1, self.trainer_cfg["epochs"] + 1):
            epoch_start = time.time()
            self.model.train()
            epoch_losses = []

            for batch in dataloader:
                features = batch["features"]
                feature_mask = batch["feature_mask"]
                timestamps = batch["timestamps"]
                mask = batch["mask"]

                features_norm, features_raw = self._normalize_batch(features, feature_mask)

                data_mask = mask.to(self.device)
                masked_positions = self.mask_generator(data_mask)

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
                loss = losses["total"]

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_losses.append(loss.item())
                global_step += 1

                if global_step % self.trainer_cfg.get("log_every", 50) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    rss_mb = psutil.Process().memory_info().rss / 1e6
                    self.log.info(
                        f"E{epoch} S{global_step} loss={loss.item():.4f} lr={lr:.2e} "
                        f"gn={grad_norm:.2f} rss={rss_mb:.0f}MB"
                    )

            epoch_time = time.time() - epoch_start
            avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            val_loss = self._validate(val_loader)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint_mgr.save(
                epoch=epoch,
                step=global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                normalizer_state=self.normalizer.state_dict(),
                mask_generator_state=self.mask_generator.get_state(),
                train_loss=avg_train_loss,
                val_loss=val_loss,
                is_best=is_best,
            )
            self.log.info(
                f"Epoch {epoch}/{self.trainer_cfg['epochs']} "
                f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                f"time={epoch_time:.1f}s best={is_best}"
            )

        self.log.info("Training complete.")

    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"]
                feature_mask = batch["feature_mask"]
                timestamps = batch["timestamps"]
                mask = batch["mask"]

                features_norm, features_raw = self._normalize_batch(features, feature_mask)
                data_mask = mask.to(self.device)
                masked_positions = self.mask_generator(data_mask)
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
                count += features.shape[0]

        self.model.train()
        return total_loss / count if count > 0 else 0.0
