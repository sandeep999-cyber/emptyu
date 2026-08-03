import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.trainer import TeacherTrainer
from src.training.sampler import EpochMarketSampler


MODEL_CFG = {
    "model": {
        "name": "teacher_transformer_v1",
        "feature_dim": 15,
        "context_length": 512,
        "cls_token": True,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "rope_theta": 10000.0,
    },
    "loss": {
        "masked_modeling": {
            "mask_ratio": 0.15,
            "price_indices": [0, 1, 2, 3, 4],
            "funding_oi_indices": [5, 6],
            "calendar": {
                "minute_of_day": {"index": 7, "classes": 1440, "offset": 0},
                "hour": {"index": 8, "classes": 24, "offset": 0},
                "day_of_week": {"index": 9, "classes": 7, "offset": 0},
                "day_of_month": {"index": 10, "classes": 31, "offset": 1},
                "month": {"index": 11, "classes": 12, "offset": 1},
                "quarter": {"index": 12, "classes": 4, "offset": 1},
                "year": {"index": 13, "classes": 16, "offset": 2020},
                "is_weekend": {"index": 14, "classes": 2, "offset": 0},
            },
            "group_weights": {"price": 1.0, "funding_oi": 1.0, "calendar": 1.0},
        }
    },
}

OPT_CFG = {
    "optimizer": {
        "adamw": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "grad_clip": 1.0,
    },
    "scheduler": {"warmup_frac": 0.05, "lr_floor": 1e-6},
}

TRAINER_CFG = {
    "epochs": 2,
    "batch_size": 4,
    "market": "futures",
    "train_window_stride": 16,
    "val_window_stride": 16,
    "device": "cpu",
    "seed": 42,
    "log_every": 50,
    "normalizer": {"mode": "zscore"},
}


@pytest.fixture
def mock_trainer():
    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch("src.training.trainer._load_manifest") as mock_mf,
            patch("src.training.trainer.lake") as mock_lake,
            patch("src.training.trainer.feature_builder") as mock_fb,
            patch("src.training.trainer.WindowingEngine") as mock_we,
        ):
            mock_mf.return_value = {
                "splits": {
                    "train": {"symbols": ["BTCUSDT"]},
                    "validation": {"symbols": ["SOLUSDT"]},
                }
            }
            mock_df = MagicMock()
            mock_df.empty = False
            mock_lake.market_state.return_value = mock_df
            mock_fb.build_features.return_value = (
                torch.randn(1000, 15).numpy(),
                torch.ones(1000, 15, dtype=torch.bool).numpy(),
                [0] * 1000,
            )
            mock_engine = MagicMock()
            mock_engine.create_windows.return_value = [
                {"features": torch.randn(512, 15).numpy(),
                 "feature_mask": torch.ones(512, 15, dtype=torch.bool).numpy(),
                 "timestamps": list(range(512)),
                 "mask": torch.ones(512, dtype=torch.bool).numpy(),
                 "metadata": {"symbol": "BTCUSDT", "snapshot_id": "2026-07-30", "market": "futures"}}
                for _ in range(10)
            ]
            mock_we.return_value = mock_engine

            trainer = TeacherTrainer(MODEL_CFG, OPT_CFG, TRAINER_CFG, Path(tmp))
            yield trainer


class TestTeacherTrainer:
    def test_init(self, mock_trainer):
        assert mock_trainer.model is not None
        assert mock_trainer.device == torch.device("cpu")

    def test_amp_dtype_disabled(self, mock_trainer):
        assert mock_trainer.use_amp is False
        assert mock_trainer.amp_dtype is None

    def test_amp_dtype_bf16_on_ampere_plus(self, mock_trainer):
        mock_trainer.device = torch.device("cuda")
        mock_trainer.trainer_cfg["mixed_precision"] = True
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            assert mock_trainer._resolve_amp_dtype() == torch.bfloat16

    def test_amp_dtype_fp16_on_pre_ampere(self, mock_trainer):
        mock_trainer.device = torch.device("cuda")
        mock_trainer.trainer_cfg["mixed_precision"] = True
        with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            assert mock_trainer._resolve_amp_dtype() == torch.float16

    def test_amp_dtype_none_on_maxwell(self, mock_trainer):
        mock_trainer.device = torch.device("cuda")
        mock_trainer.trainer_cfg["mixed_precision"] = True
        with patch("torch.cuda.get_device_capability", return_value=(5, 0)):
            assert mock_trainer._resolve_amp_dtype() is None

    def test_model_forward(self, mock_trainer):
        mock_trainer.model.eval()
        features = torch.randn(1, 512, 15)
        timestamps = torch.arange(512).unsqueeze(0)
        mask = torch.ones(1, 512, dtype=torch.bool)
        with torch.no_grad():
            latent, kpm, positions, t_data = mock_trainer.model(features, timestamps, mask)
        assert latent.shape == (1, 513, 128)

    def test_model_reconstruct(self, mock_trainer):
        mock_trainer.model.eval()
        latent = torch.randn(1, 512, 128)
        rec = mock_trainer.model.reconstruct(latent)
        assert "price" in rec
        assert "funding_oi" in rec

    def test_rope_cache_prebuilt_before_compile(self, mock_trainer):
        """Regression: the RoPE cache must be built BEFORE torch.compile so CUDA-graph
        replay (compile_mode: reduce-overhead) never overwrites the cos/sin buffers."""
        rope = mock_trainer.model.rope
        assert rope._cache_ready is True
        mp = mock_trainer.model.max_position
        assert mp is not None and mp > 0
        assert rope._cache_cos is not None
        assert rope._cache_cos.shape[0] == mp + 1

    def test_rope_cache_noop_forward_does_not_resize(self, mock_trainer):
        """Forward must not resize the pre-built cache (guard short-circuits)."""
        rope = mock_trainer.model.rope
        before_len = rope._cache_max_len
        mock_trainer.model.eval()
        features = torch.randn(1, 512, 15)
        timestamps = torch.arange(512).unsqueeze(0)
        mask = torch.ones(1, 512, dtype=torch.bool)
        with torch.no_grad():
            mock_trainer.model(features, timestamps, mask)
        assert rope._cache_max_len == before_len

    def test_cudagraph_mark_step_begin_called_per_invocation(self, mock_trainer):
        """Regression: reduce-overhead CUDA-graph replay overwrites pooled output
        buffers, so cudagraph_mark_step_begin() must be called before each compiled
        model invocation (train + val) or backward reads stale memory."""
        mock_trainer.model_is_compiled = True
        batch = {
            "features": torch.randn(4, 512, 15),
            "feature_mask": torch.ones(4, 512, 15, dtype=torch.bool),
            "timestamps": torch.arange(512).unsqueeze(0).expand(4, -1),
            "mask": torch.ones(4, 512, dtype=torch.bool),
        }
        fake_loader = MagicMock()
        fake_loader.__iter__.side_effect = lambda: iter([batch])  # fresh iterator per epoch

        with (
            patch("src.training.trainer.create_dataloader", return_value=fake_loader),
            patch("src.training.trainer.torch.compiler.cudagraph_mark_step_begin") as mark,
            patch.object(mock_trainer, "_validate", return_value=(1.0, {})),
            patch.object(mock_trainer.checkpoint_mgr, "save"),
        ):
            mock_trainer.train()

        # 2 epochs x 1 train batch = 2 compiled invocations (val loop is patched out).
        assert mark.call_count >= 2, mark.call_count

    def test_reduce_overhead_compile_mode_refused(self):
        """Regression: CUDA-graph reduce-overhead crashes this model's autograd on
        Colab (overwritten buffer error); the trainer must fall back to the safe
        default inductor mode (None) instead of propagating it."""
        from src.training.trainer import _resolve_compile_mode
        assert _resolve_compile_mode("reduce-overhead") is None
        assert _resolve_compile_mode(None) is None
        assert _resolve_compile_mode("max-autotune") == "max-autotune"
        assert _resolve_compile_mode("default") == "default"

    def test_trainer_advances_sampler_epoch(self, mock_trainer):
        """Regression: sampler epoch was stuck at 0 (same order every epoch)."""
        class SpySampler(EpochMarketSampler):
            instances = []

            def __init__(self, n, shuffle=True, seed=42):
                super().__init__(n, shuffle, seed)
                self.epochs = []
                SpySampler.instances.append(self)

            def set_epoch(self, epoch):
                self.epochs.append(epoch)
                super().set_epoch(epoch)

        SpySampler.instances = []
        fake_loader = MagicMock()
        fake_loader.__iter__.return_value = iter([])

        with (
            patch("src.training.trainer.EpochMarketSampler", SpySampler),
            patch("src.training.trainer.create_dataloader", return_value=fake_loader),
            patch.object(mock_trainer, "_validate", return_value=(1.0, {})),
            patch.object(mock_trainer.checkpoint_mgr, "save"),
        ):
            mock_trainer.train()

        train_samplers = [s for s in SpySampler.instances if s.data_source_len == len(mock_trainer.train_dataset)]
        assert train_samplers, "expected a train sampler to be created"
        assert train_samplers[0].epochs == [1, 2]


class TestResume:
    def _make_checkpoint(self, run_dir: Path, epoch: int, step: int, val_loss: float):
        """Build a real checkpoint + manifest under run_dir for resume tests."""
        run_dir.mkdir(parents=True, exist_ok=True)
        from src.training.checkpoint import CheckpointManager
        from src.models.teacher.encoder import TeacherEncoder
        mgr = CheckpointManager(run_dir, {"model_config": MODEL_CFG, "optimizer_config": OPT_CFG, "trainer_config": TRAINER_CFG})
        full_model_cfg = {**MODEL_CFG["model"], "loss": MODEL_CFG.get("loss", {})}
        model = TeacherEncoder(full_model_cfg)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        gen_state = torch.Generator().manual_seed(42).get_state()
        mgr.save(
            epoch=epoch,
            step=step,
            model=model,
            optimizer=opt,
            scheduler=None,
            normalizer_state={"mode": "zscore", "mean": [1.0, 2.0, 3.0] + [0.0] * 12, "std": [1.0] * 15},
            mask_generator_state=gen_state,
            train_loss=0.5,
            val_loss=val_loss,
            is_best=True,
            metrics={"train": {"total": 0.5}, "val": {"total": val_loss}},
        )
        return mgr

    def test_resume_restores_state_and_appends_history(self, mock_trainer):
        with tempfile.TemporaryDirectory() as tmp:
            resume_dir = Path(tmp) / "run"
            self._make_checkpoint(resume_dir, epoch=1, step=120, val_loss=0.7)

            with (
                patch("src.training.trainer._load_manifest") as mock_mf,
                patch("src.training.trainer.lake") as mock_lake,
                patch("src.training.trainer.feature_builder") as mock_fb,
                patch("src.training.trainer.WindowingEngine") as mock_we,
            ):
                mock_mf.return_value = {
                    "splits": {
                        "train": {"symbols": ["BTCUSDT"]},
                        "validation": {"symbols": ["SOLUSDT"]},
                    }
                }
                mock_df = MagicMock()
                mock_df.empty = False
                mock_lake.market_state.return_value = mock_df
                mock_fb.build_features.return_value = (
                    torch.randn(1000, 15).numpy(),
                    torch.ones(1000, 15, dtype=torch.bool).numpy(),
                    [0] * 1000,
                )
                mock_engine = MagicMock()
                mock_engine.create_windows.return_value = [
                    {"features": torch.randn(512, 15).numpy(),
                     "feature_mask": torch.ones(512, 15, dtype=torch.bool).numpy(),
                     "timestamps": list(range(512)),
                     "mask": torch.ones(512, dtype=torch.bool).numpy(),
                     "metadata": {"symbol": "BTCUSDT", "snapshot_id": "2026-07-30", "market": "futures"}}
                    for _ in range(10)
                ]
                mock_we.return_value = mock_engine

                trainer = TeacherTrainer(
                    MODEL_CFG, OPT_CFG, TRAINER_CFG, Path(tmp) / "run", resume_dir=resume_dir
                )

            assert trainer.start_epoch == 1
            assert trainer.start_step == 120
            # History from the resume dir should be loaded into the new manager
            assert len(trainer.checkpoint_mgr.history) == 1
            assert trainer.checkpoint_mgr.history[0]["epoch"] == 1
            # Normalizer restored from checkpoint state
            assert trainer.normalizer.state_dict()["mean"][0] == 1.0

    def test_resume_continues_epochs_after_resume_point(self, mock_trainer):
        """train() must skip completed epochs when resuming."""
        with tempfile.TemporaryDirectory() as tmp:
            resume_dir = Path(tmp) / "run"
            self._make_checkpoint(resume_dir, epoch=1, step=60, val_loss=0.9)

            with patch("src.training.trainer._load_manifest") as mock_mf:
                mock_mf.return_value = {
                    "splits": {
                        "train": {"symbols": ["BTCUSDT"]},
                        "validation": {"symbols": ["SOLUSDT"]},
                    }
                }
                # Reuse an already-built trainer (mock_trainer has _build_datasets done),
                # only patch state so it picks up resume info.
                trainer = mock_trainer
                trainer.resume_dir = resume_dir
                trainer.start_epoch = 1
                trainer.start_step = 60
                trainer.best_val_loss = 0.9

                fake_loader = MagicMock()
                fake_loader.__iter__.return_value = iter([])
                with (
                    patch("src.training.trainer.create_dataloader", return_value=fake_loader) as m_dl,
                    patch.object(trainer, "_validate", return_value=(0.8, {})) as m_val,
                ):
                    trainer.train()

            # With start_epoch=1 and epochs=2, only epoch 2 should run.
            assert m_val.call_count == 1


class TestResumeConfigAuthoritative:
    def test_load_resume_configs_reads_manifest(self):
        import json
        from src.training.train_teacher import load_resume_configs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "configs": {
                            "model_config": {"model": {"d_model": 999}},
                            "optimizer_config": {"optimizer": {"adamw": {"lr": 1e-3}}},
                            "trainer_config": {"trainer": {"seed": 7}},
                        }
                    }
                )
            )
            cfg = load_resume_configs(run_dir)
            assert cfg["model_config"]["model"]["d_model"] == 999
            assert cfg["trainer_config"]["trainer"]["seed"] == 7

    def test_load_resume_configs_missing_raises(self):
        from src.training.train_teacher import load_resume_configs

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                load_resume_configs(Path(tmp) / "nope")

    def test_main_uses_resume_manifest_configs_not_cli_yaml(self):
        """Regression: resume must use the run's recorded configs, not CLI defaults."""
        import json
        import sys
        from unittest.mock import patch

        from src.training import train_teacher

        with tempfile.TemporaryDirectory() as tmp:
            resume_dir = Path(tmp) / "run"
            resume_dir.mkdir()
            (resume_dir / "latest.json").write_text(json.dumps({"latest": "checkpoint_epoch1.pt"}))
            (resume_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "configs": {
                            "model_config": {"model": {"d_model": 512, "feature_style_hint": "returns"}},
                            "optimizer_config": {"optimizer": {"adamw": {"lr": 5e-4}}},
                            "trainer_config": {"seed": 7, "batch_size": 64, "feature_style": "returns"},
                        }
                    }
                )
            )
            captured = {}

            class FakeTrainer:
                def __init__(self, **kw):
                    captured.update(kw)

                def train(self):
                    pass

            with (
                patch.object(sys, "argv", ["train_teacher", "--resume", str(resume_dir)]),
                patch.object(train_teacher, "TeacherTrainer", FakeTrainer),
                patch.object(
                    train_teacher,
                    "load_yaml",
                    return_value={
                        "model": {"d_model": 1},
                        "trainer": {"seed": 1, "batch_size": 8},
                        "optimizer": {"adamw": {"lr": 1e-9}},
                    },
                ),
                patch.object(train_teacher, "experiment_registry"),
                patch.object(train_teacher, "seed_everything"),
            ):
                train_teacher.main()

            assert captured["model_cfg"]["model"]["d_model"] == 512
            assert captured["trainer_cfg"]["seed"] == 7
            assert captured["trainer_cfg"]["feature_style"] == "returns"
            assert captured["trainer_cfg"]["batch_size"] == 64
            assert captured["run_dir"] == resume_dir

    def test_resume_accepts_wrapped_trainer_config(self):
        """Backward compat: manifests with a nested 'trainer' key still resume."""
        import json
        import sys
        from unittest.mock import patch

        from src.training import train_teacher

        with tempfile.TemporaryDirectory() as tmp:
            resume_dir = Path(tmp) / "run"
            resume_dir.mkdir()
            (resume_dir / "latest.json").write_text(json.dumps({"latest": "checkpoint_epoch1.pt"}))
            (resume_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "configs": {
                            "model_config": {"model": {"d_model": 512}},
                            "optimizer_config": {"optimizer": {"adamw": {"lr": 5e-4}}},
                            "trainer_config": {"trainer": {"seed": 9, "batch_size": 32}},
                        }
                    }
                )
            )
            captured = {}

            class FakeTrainer:
                def __init__(self, **kw):
                    captured.update(kw)

                def train(self):
                    pass

            with (
                patch.object(sys, "argv", ["train_teacher", "--resume", str(resume_dir)]),
                patch.object(train_teacher, "TeacherTrainer", FakeTrainer),
                patch.object(train_teacher, "load_yaml", return_value={"model": {}, "trainer": {}, "optimizer": {}}),
                patch.object(train_teacher, "experiment_registry"),
                patch.object(train_teacher, "seed_everything"),
            ):
                train_teacher.main()

            assert captured["trainer_cfg"]["seed"] == 9
            assert captured["trainer_cfg"]["batch_size"] == 32


class TestSelfContainedSmoke:
    """Runs the full train() loop end-to-end with synthetic windows (no real data)."""

    def _train_one_epoch(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = dict(TRAINER_CFG)
            cfg["epochs"] = 1
            cfg["max_train_windows"] = 8
            cfg["max_val_windows"] = 4
            with (
                patch("src.training.trainer._load_manifest") as mock_mf,
                patch("src.training.trainer.lake") as mock_lake,
                patch("src.training.trainer.feature_builder") as mock_fb,
                patch("src.training.trainer.WindowingEngine") as mock_we,
            ):
                mock_mf.return_value = {
                    "splits": {
                        "train": {"symbols": ["BTCUSDT"]},
                        "validation": {"symbols": ["SOLUSDT"]},
                    }
                }
                mock_df = MagicMock()
                mock_df.empty = False
                mock_lake.market_state.return_value = mock_df
                mock_fb.build_features.return_value = (
                    np.random.default_rng(0).standard_normal((512, 15)).astype(np.float32),
                    np.ones((512, 15), dtype=bool),
                    np.arange(512, dtype=np.int64),
                )
                mock_engine = MagicMock()
                mock_engine.create_windows.return_value = [
                    {"features": np.random.default_rng(i).standard_normal((128, 15)).astype(np.float32),
                     "feature_mask": np.ones((128, 15), dtype=bool),
                     "timestamps": np.arange(128, dtype=np.int64),
                     "mask": np.ones(128, dtype=bool),
                     "metadata": {"symbol": s, "window_end_ms": 1700000000000 + i * 60000, "market": "futures"}}
                    for i, s in enumerate(["BTCUSDT"] * 8 + ["SOLUSDT"] * 4)
                ]
                mock_we.return_value = mock_engine

                model_cfg = dict(MODEL_CFG)
                model_cfg["model"] = dict(model_cfg["model"])
                model_cfg["model"]["context_length"] = 128
                # Tiny model for speed.
                model_cfg["model"]["d_model"] = 32
                model_cfg["model"]["n_layers"] = 1
                model_cfg["model"]["n_heads"] = 2
                model_cfg["model"]["d_ff"] = 64

                trainer = TeacherTrainer(model_cfg, OPT_CFG, cfg, run_dir)
                trainer.train()

            manifest = json.loads((run_dir / "manifest.json").read_text())
            assert len(manifest["checkpoints"]) == 1
            assert (run_dir / "checkpoint_epoch1.pt").exists()
            assert (run_dir / "latest.json").exists()
            ckpt = manifest["checkpoints"][0]
            assert ckpt["epoch"] == 1
            assert "metrics" in ckpt
            assert ckpt["metrics"]["val"]["total"] >= 0

    def test_smoke_train_end_to_end(self):
        self._train_one_epoch()
