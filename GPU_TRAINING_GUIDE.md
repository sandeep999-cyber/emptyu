# GPU Training Guide — CUDA Host

Full-model training targets a CUDA GPU. This guide covers host setup, moving
the frozen dataset, running, resuming, and recovering from failures.

## 0. TL;DR

```bash
# Host prep (one-time)
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt          # torch>=2.1 CUDA build

# Data on this machine (frozen fingerprint must match)
python main.py validate                   # asserts dataset fingerprint
python -m src.training.train_teacher \
  --model-config configs/model_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_v1.yaml

# If interrupted
python -m src.training.train_teacher --resume models/foundation/teacher_v1/<run_id>
```

## 1. Prerequisites

- **PyTorch built for CUDA.** Verify:

  ```bash
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
  ```

  If `torch.cuda.is_available()` is `False`, reinstall the CUDA wheel
  (`pip install torch --index-url https://download.pytorch.org/whl/cu121` or
  matching your driver). The default PyPI wheel is CPU-only.
- **Disk space.** Dataset ≈ few hundred MB of parquet; checkpoints per run ≈
  a few hundred MB for the full model (`d_model 512`).
- **RAM.** Window building materializes only `N/stride` windows (stride 16),
  so RAM stays modest; `num_workers` still multiplies loader memory.

## 2. Moving data onto the GPU host

The trainer reads from `storage/canonical/` via the DuckDB index. You have two
options:

1. **Full canonical tree** (source of truth): copy the whole
   `storage/` folder (or the snapshot under `storage/training/snapshots/`) and
   run `python main.py snapshot --date <date>` to rebuild the canonical tree +
   index (see `DATA_FLOW.md` §9).
2. **Reduced subset** (fast start on the Colab-style path): copy only the
   futures 1m subset (BTCUSDT/ETHUSDT/SOLUSDT klines+funding+OI) into
   `storage/canonical/futures/`, then run `python main.py validate`. The
   fingerprint check will differ — see `REPRODUCIBILITY.md` §8 for the change
   gate (a reduced subset is a *different* dataset identity).

After moving data, always run `python main.py validate` and confirm
`storage/training/index.duckdb` covers the copied symbols (`DATA_FLOW.md` §8).

## 3. Configuration for a GPU run

Defaults in `configs/trainer_v1.yaml` already target GPU:

- `device: auto` → `cuda` when available.
- `mixed_precision: true` → bf16 autocast + `GradScaler` on CUDA.
- `num_workers` → set to a sensible value for your machine (common choice:
  CPU core count); `pin_memory` and `persistent_workers` are enabled
  automatically on CUDA.
- **Micro-batching:** `batch_size: 64` is the *effective* batch; the model is
  fed `micro_batch_size: 16` per step and grads accumulate 4×, so the LR
  schedule and optimizer semantics match a true batch-64 run while VRAM usage
  drops ~4×. This fits a ~14.5 GiB GPU (e.g. Colab T4) at `context_length 512`
  / `d_model 512`. On a smaller GPU, lower `micro_batch_size` further; on a
  large GPU you can raise it or set it equal to `batch_size` (no accumulation).

Expectations (full model, stride 16): ~65 K windows/epoch ≈ 33 M tokens,
~1,030 optimizer steps/epoch at effective batch 64; order of minutes per epoch
on a modern GPU; 10 epochs well under a few hours. CPU full training is not
recommended.

## 4. Running

```bash
python -m src.training.train_teacher \
  --model-config configs/model_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_v1.yaml
```

Returns/span-mask variant (stationary features):

```bash
python -m src.training.train_teacher \
  --model-config configs/model_returns_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_returns_v1.yaml
```

Monitor: `logs/train_teacher.log` for `Device: cuda`, per-step loss/lr/grad
norm, per-epoch train/val loss, and the final registry log.

## 5. Resuming after interruption

The trainer saves a checkpoint every `checkpoint_every` epoch, so an interrupt
never loses more than one epoch:

```bash
python -m src.training.train_teacher --resume models/foundation/teacher_v1/<run_id>
```

- Resume uses the run's `manifest.json` configs (CLI configs ignored) and
  `latest.json` to pick the checkpoint; restores model/optimizer/scheduler/
  normalizer/mask-generator state and continues at the recorded
  `(epoch, step)`.
- If `latest.json` is missing in the run dir, the CLI **errors out** — a valid
  resume dir must contain `manifest.json` + `latest.json` + at least one `.pt`
  (`CHECKPOINT_FORMAT.md` §3). If a checkpoint `.pt` fails to load, the trainer
  warns and starts fresh.

## 6. Failure recovery playbook

| Failure | Detection | Recovery |
|---|---|---|
| Interrupt / power loss | no `val loss` lines recently | `--resume` same run dir |
| CUDA OOM | `torch.cuda.OutOfMemoryError` | lower `micro_batch_size` (keeps effective batch + LR schedule) or `max_train_windows`; resume from last checkpoint |
| CUDA driver/version mismatch | `torch.cuda.is_available() == False` | reinstall matching CUDA wheel |
| Corrupt checkpoint | `Failed to load resume checkpoint` warning | delete/rename the corrupt `.pt`, update `latest.json` to a good one, resume |
| Fingerprint mismatch | `validate` fails assertion | reconcile data (restore snapshot) — do NOT silently train on different data |
| Disk full during save | save traceback, partial `.pt` | free space; restart save; sha256 in manifest protects against using partial files |

`CHECKPOINT_FORMAT.md` §4 note: `manifest.json`'s per-checkpoint `sha256` lets
you confirm a `.pt` is complete before trusting it.

## 7. Multi-node / reproducibility caveat

The pipeline is single-node, single-GPU by design. Bit-exact reproducibility
across GPU models is not guaranteed (see `REPRODUCIBILITY.md` §4); record the
GPU model + torch version in your experiment notes to explain any numeric
divergence.

## 8. Related

- `TRAINING_GUIDE.md` — full training semantics (smoke/full, windowing,
  normalizer, checkpoint lifecycle).
- `COLAB_GUIDE.md` — managed GPU (Colab) variant of this workflow.
- `CHECKPOINT_FORMAT.md`, `REPRODUCIBILITY.md` — formats + anchors.
