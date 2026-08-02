# Troubleshooting — Catalog of Failure Modes

Common failures, their likely causes, and fixes, organized by area. Each entry
ties back to the module docs (`MODULE_REFERENCE.md`) or a guide.

## 1. Setup / dependencies

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src...` | running from wrong directory or not installed | run from repo root; `pip install -e .` if you use installs |
| `pip install` fails building pyarrow/duckdb | no prebuilt wheel for your Python | use Python 3.10-3.12; `pip install -r requirements.txt` |
| `torch.cuda.is_available()` is False | CPU-only torch wheel installed | reinstall CUDA wheel (see `GPU_TRAINING_GUIDE.md` §1) |
| Colab: torch downgraded after `pip install -r` | `requirements.txt` pins a torch version | Cell 3b fail-fast; reinstall CUDA torch |
| `No module named sklearn` | scikit-learn missing | `pip install scikit-learn>=1.3` |

## 2. Data / pipeline

| Symptom | Cause | Fix |
|---|---|---|
| `validate` fingerprint mismatch | canonical data differs from `dataset_fingerprint.json` | restore from snapshot (`main.py snapshot --date <date>`); do NOT train on changed data (`REPRODUCIBILITY.md` §8) |
| `No data for symbol ...` | `storage/canonical/futures/<sym>/` missing | copy full dataset or subset (`GPU_TRAINING_GUIDE.md` §2) |
| `RuntimeError: No training windows available` | no windows pass the time split (train windows must end ≤ `train_end`) | check manifest, data coverage |
| `No validation windows available` | SOL data absent | download/rebuild SOL |
| Empty results in baseline harness | `len(df) < seq_len` for a symbol | enlarge period or lower `--seq-len` |
| DuckDB `database is locked` | concurrent writers on shared file (Drive) | Colab: copy to local SSD (Cell 5), sync back at end (`COLAB_GUIDE.md` §2) |
| `file_index_v1` missing table | index not built for this storage | `python main.py index` (`DATA_FLOW.md` §8) |

## 3. Training

| Symptom | Cause | Fix |
|---|---|---|
| CUDA OOM | batch 64 × 513×512 too large for VRAM | lower `batch_size`, or `max_train_windows`; resume from last checkpoint |
| Loss explodes / NaN | extreme raw features + unstable lr | check normalizer fitted (it's auto); reduce lr; verify fingerprint |
| Val loss far above train | held-out symbol distribution shift | expected at pilot scale; see `MODEL_CARD.md` §6 |
| Resume errors: `does not contain latest.json` | resume dir lacks `latest.json` | a valid run dir needs `manifest.json` + `latest.json` + `.pt` files; recreate `latest.json` pointing at a good checkpoint (`CHECKPOINT_FORMAT.md` §3) |
| `Failed to load resume checkpoint` warning (then fresh start) | corrupt `.pt` | verify sha256 in manifest; delete corrupt file; point `latest.json` at a good one; resume |
| Results not reproducible | changed data/config/seed | re-check fingerprint, configs, git commit, seed 42 |
| Very slow on CPU | full config on CPU | use smoke config or a GPU (`TRAINING_GUIDE.md` §3/§4) |
| `num_workers` stall/hang (Colab) | too many workers for 2-core VM | keep trainer `num_workers` default (2) |

## 4. Evaluation

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: No manifest.json in <dir>` | passed a checkpoint .pt path instead of the run dir | pass `models/foundation/teacher_v1/<run_id>` (the run directory) |
| `Too few samples for k clusters` | `--max-windows` too small vs `--clusters` | raise `--max-windows` or lower `--clusters` |
| All-NaN metrics for a task | single-class labels after binarization | expected with tiny data; check `n` in JSON |
| `--checkpoint` seq-len mismatch warning | CLI `--seq-len` ≠ model `context_length` | harness auto-overrides to model value (`EVALUATION_GUIDE.md` §4) |
| `--feature-style` mismatch warning | forced style differs from checkpoint | omit `--feature-style` (defaults to checkpoint) |
| Eval JSONs not appearing | out dir is git-ignored / wrong CWD | outputs under `evaluation/embedding/`, `evaluation/baselines/` (git-ignored by design) |

## 5. Checkpoints / run dirs

| Symptom | Cause | Fix |
|---|---|---|
| Multiple run dirs, not sure which | every start creates a new run_id | `latest.json` + `manifest.json` tell you config/commit/val_loss; eval tools default to newest checkpoint |
| `best.pt` missing | no best-val epoch saved yet | eval falls back to newest `checkpoint_epoch*.pt` |
| Checkpoint won't load in eval | model architecture differs from manifest | always point eval at the run dir that produced the checkpoint |
| Huge run dir | one .pt per epoch | full model ≈ few hundred MB; keep only what you need |

## 6. GPU / Colab

| Symptom | Cause | Fix |
|---|---|---|
| Colab runtime disconnects mid-train | idle/disconnect timeout | resume via Cell 10; checkpoints every epoch; don't kill before Cell 17 |
| `phase2_results.zip` missing on Drive | kernel killed before Cell 17 | re-run Cell 17; keep DuckDB writes local until sync |
| `$CHECKPOINT_DIR` wrong after resume | env var set in a previous session | re-run Cell 11 after resuming |
| nvidia-smi shows no GPU | runtime not GPU-enabled | Runtime → Change runtime type → T4/A100 |

## 7. Git / repo state

| Symptom | Cause | Fix |
|---|---|---|
| `git describe --tags` fails | not on `v1.0-baseline` | `git checkout v1.0-baseline` |
| Working tree dirty after experiments | eval/models/logs are git-ignored, so usually not | check `git status`; nothing under those dirs should be committed |
| Docs changed but git clean | docs committed at `dcc16c7` (HEAD) | commit new docs only when asked |

## 8. General debugging tips

- **Always read the log first:** `logs/train_teacher.log` has per-step
  loss/lr/grad-norm; eval CLI prints full JSON to stdout.
- **Reproduce against a known-good run:** the smoke run
  `models/foundation/teacher_v1/20260801_214517_smoke` + the two eval JSONs in
  `evaluation/` are the reference artifacts. If your run diverges wildly,
  compare fingerprints/commit first.
- **Determinism:** keep seed 42 and the same data; note GPU model + torch
  version for numeric-divergence explanations (`REPRODUCIBILITY.md` §4).
- If a fix changes data, model, or loss semantics, re-fingerprint and re-run
  baselines (`REPRODUCIBILITY.md` §8).
