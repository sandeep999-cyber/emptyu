import pytest
import torch
import json
import tempfile
from pathlib import Path

from src.training.checkpoint import CheckpointManager
from src.models.teacher.encoder import TeacherEncoder


MODEL_CFG = {
    "context_length": 128,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.0,
    "feature_dim": 15,
    "rope_theta": 10000.0,
    "loss": {
        "masked_modeling": {
            "mask_ratio": 0.15,
            "price_indices": [0, 1, 2, 3, 4],
            "funding_oi_indices": [5, 6],
            "calendar": {},
        }
    },
}


@pytest.fixture
def run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestCheckpointManager:
    def test_save_and_load(self, run_dir):
        model = TeacherEncoder(MODEL_CFG)
        mgr = CheckpointManager(run_dir, configs={"model": MODEL_CFG})
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        mgr.save(
            epoch=0, step=10, model=model, optimizer=optim, scheduler=None,
            normalizer_state={}, mask_generator_state=torch.Generator().get_state(),
            train_loss=1.0, val_loss=0.5, is_best=True,
        )
        assert (run_dir / "checkpoint_epoch0.pt").exists()
        assert (run_dir / "best.pt").exists()

    def test_manifest_created(self, run_dir):
        mgr = CheckpointManager(run_dir, configs={"model": MODEL_CFG})
        model = TeacherEncoder(MODEL_CFG)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        mgr.save(
            epoch=0, step=10, model=model, optimizer=optim, scheduler=None,
            normalizer_state={}, mask_generator_state=torch.Generator().get_state(),
            train_loss=1.0, val_loss=0.5,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert len(manifest["checkpoints"]) == 1

    def test_multiple_checkpoints(self, run_dir):
        mgr = CheckpointManager(run_dir, configs={})
        model = TeacherEncoder(MODEL_CFG)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        for ep in range(3):
            mgr.save(
                epoch=ep, step=ep * 10, model=model, optimizer=optim, scheduler=None,
                normalizer_state={}, mask_generator_state=torch.Generator().get_state(),
                train_loss=1.0 / (ep + 1), val_loss=0.5 / (ep + 1),
            )
        assert len(json.loads((run_dir / "manifest.json").read_text())["checkpoints"]) == 3

    def test_latest_json(self, run_dir):
        mgr = CheckpointManager(run_dir, configs={})
        model = TeacherEncoder(MODEL_CFG)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        mgr.save(
            epoch=5, step=100, model=model, optimizer=optim, scheduler=None,
            normalizer_state={}, mask_generator_state=torch.Generator().get_state(),
            train_loss=1.0, val_loss=0.5,
        )
        latest = json.loads((run_dir / "latest.json").read_text())
        assert latest["latest"] == "checkpoint_epoch5.pt"

    def test_async_save_flush_and_load(self, run_dir):
        mgr = CheckpointManager(run_dir, configs={}, async_writes=True)
        model = TeacherEncoder(MODEL_CFG)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        mgr.save(
            epoch=7, step=70, model=model, optimizer=optim, scheduler=None,
            normalizer_state={}, mask_generator_state=torch.Generator().get_state(),
            train_loss=0.7, val_loss=0.4,
        )
        mgr.flush()
        assert (run_dir / "checkpoint_epoch7.pt").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert len(manifest["checkpoints"]) == 1
        latest = json.loads((run_dir / "latest.json").read_text())
        assert latest["latest"] == "checkpoint_epoch7.pt"

        loader = CheckpointManager(run_dir, configs={})
        (epoch, step, *_rest) = loader.load_latest_checkpoint(model, optim)
        assert epoch == 7 and step == 70
        mgr.close()
