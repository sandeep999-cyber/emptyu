"""CLI entry point for Teacher Foundation Model training."""

import argparse
import json
import os
from pathlib import Path
import yaml
import torch

from src.training.trainer import TeacherTrainer
from src.training.checkpoint import CheckpointManager
from src.training.experiment_registry import experiment_registry
from src.training.seeding import seed_everything
import time


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_resume_configs(resume_dir: Path) -> dict:
    """Load the configs recorded in a run's manifest.json (source of truth)."""
    manifest_path = resume_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {resume_dir}")
    manifest = json.loads(manifest_path.read_text())
    configs = manifest.get("configs")
    if not configs:
        raise ValueError(f"manifest.json in {resume_dir} has no configs section")
    return configs


def main():
    parser = argparse.ArgumentParser(description="Train Teacher Foundation Model")
    parser.add_argument("--model-config", default="configs/model_v1.yaml")
    parser.add_argument("--optimizer-config", default="configs/optimizer_v1.yaml")
    parser.add_argument("--trainer-config", default="configs/trainer_v1.yaml")
    parser.add_argument("--smoke", action="store_true", help="Override for CPU smoke test (tiny model, few windows)")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing run dir to resume")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    opt_cfg = load_yaml(args.optimizer_config)
    trainer_cfg = load_yaml(args.trainer_config)

    if args.smoke:
        model_cfg["model"]["n_layers"] = 2
        model_cfg["model"]["n_heads"] = 4
        model_cfg["model"]["d_model"] = 128
        model_cfg["model"]["d_ff"] = 512
        trainer_cfg["trainer"]["batch_size"] = 8
        trainer_cfg["trainer"]["epochs"] = 1
        trainer_cfg["trainer"]["max_train_windows"] = 256
        trainer_cfg["trainer"]["max_val_windows"] = 64
        trainer_cfg["trainer"]["device"] = "cpu"
        trainer_cfg["trainer"]["train_window_stride"] = 16
        trainer_cfg["trainer"]["val_window_stride"] = 16
        trainer_cfg["trainer"]["log_every"] = 10
        trainer_cfg["trainer"]["num_workers"] = max(1, (os.cpu_count() or 1) // 2)
        print(f"[Smoke] Overriding config for smoke test (2L/128d, 256 windows, 1 epoch, CPU, {trainer_cfg['trainer']['num_workers']} workers)")

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{'smoke' if args.smoke else 'full'}"
    run_dir = Path("models/foundation/teacher_v1") / run_id

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir is not None:
        if not (resume_dir / "latest.json").exists():
            parser.error(f"--resume path {resume_dir} does not contain latest.json")
        try:
            resume_configs = load_resume_configs(resume_dir)
        except (FileNotFoundError, ValueError) as e:
            parser.error(f"--resume path {resume_dir}: {e}")
        # The resumed run's recorded configs are authoritative; the CLI configs
        # must not silently override them (regression: `x or y` never fell back).
        model_cfg = resume_configs["model_config"]
        opt_cfg = resume_configs["optimizer_config"]
        trainer_cfg = resume_configs["trainer_config"]
        print(f"[Resume] Restoring configs from {resume_dir}")
        run_dir = resume_dir

    seed = trainer_cfg["trainer"].get("seed", 42)
    seed_everything(seed)
    print(f"[Seeding] Applied global seeds (seed={seed})")

    trainer = TeacherTrainer(
        model_cfg=model_cfg,
        opt_cfg=opt_cfg,
        trainer_cfg=trainer_cfg["trainer"],
        run_dir=run_dir,
        resume_dir=resume_dir,
    )
    trainer.train()

    # Log to experiment registry
    exp_id = run_id
    try:
        with open(run_dir / "manifest.json") as f:
            manifest = json.load(f)
        val_losses = [
            c.get("val_loss")
            for c in manifest.get("checkpoints", [])
            if c.get("val_loss") is not None
        ]
        best_loss = min(val_losses) if val_losses else 0.0
        experiment_registry.log_experiment(
            experiment_id=exp_id,
            snapshot=trainer.trainer_cfg.get("snapshot_date", "2026-07-30"),
            alignment_version="alignment_v1.yaml",
            feature_builder_version="v1",
            windowing_version="windowing_v1.yaml",
            modality_registry="modalities_v1.yaml",
            objective="masked_market_modeling",
            encoder="teacher_transformer_v1",
            loss=best_loss,
            seed=trainer_cfg["trainer"].get("seed", 42),
            metrics=json.dumps({"run_dir": str(run_dir), "smoke": args.smoke}),
        )
        print(f"[ExperimentRegistry] Logged {exp_id} with final val_loss={best_loss:.4f}")
    except Exception as e:
        print(f"[Warning] Failed to log experiment: {e}")

    print(f"Training complete. Checkpoint dir: {run_dir}")


if __name__ == "__main__":
    main()
