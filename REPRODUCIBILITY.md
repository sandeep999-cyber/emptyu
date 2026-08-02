# Reproducibility — How a Run Is Pinned Down

Every training/evaluation run is tied to (a) a frozen dataset identity, (b)
exact configs, (c) the git commit, (d) deterministic seeds. This guide explains
each anchor and how to reproduce (or re-derive) a run.

## 1. The four anchors

| Anchor | Where | Example value |
|---|---|---|
| Dataset fingerprint | `storage/training/dataset_fingerprint.json` | `328a7b67b070b95e47ba450452032a93dfa410431e0cf329de6a4ac7b5ae3875` |
| Frozen manifest | `storage/training/training_manifest_v1.json` | splits + `time_split.train_end: 2024-11-30` |
| Configs (verbatim) | `models/foundation/teacher_v1/<run_id>/manifest.json` → `configs` | resolved model/optimizer/trainer |
| Git commit | `manifest.json` → `git_commit` | `d5dda35a5f7557a3c369c4ecb8473050eb55783d` (tag: `v1.0-baseline`) |

A checkpoint is reproducible iff all four match the intended run.

## 2. Dataset identity — the fingerprint

`storage/training/dataset_fingerprint.json` is computed over the canonical
dataset (not the raw downloads):

```json
{
  "fingerprint": "328a7b67b070b95e47ba450452032a93dfa410431e0cf329de6a4ac7b5ae3875",
  "file_count": 510,
  "snapshot": "2026-07-30",
  ...
}
```

Why it exists: any change in canonical files (re-alignment, resampling,
schema) changes the fingerprint. Training a model on a different fingerprint
than the one a previous result was computed on invalidates apples-to-apples
comparison. Always compare fingerprints before comparing runs.

### Dataset splits (frozen in `training_manifest_v1.json`)

| Split | Symbols | Time bounds |
|---|---|---|
| train | BTCUSDT, ETHUSDT | capped at `train_end` (2024-11-30) |
| validation | SOLUSDT | held-out symbol, unbounded in time |
| test | BTCUSDT, ETHUSDT | starts at `train_end` |

## 3. Config determinism

- Configs are **resolved and embedded verbatim** into each run's
  `manifest.json` at save time. The resolved value, not the file path, is what
  a run used.
- On resume, the run's own configs win over CLI flags (`TRAINING_GUIDE.md` §5).
- Evaluate/reproduce by pointing tools at the **run directory** — they read
  `manifest.json` and rebuild the exact model + normalizer + dataset.

## 4. Determinism and seeds

- Global seed `42` is applied at trainer startup (torch/numpy/random).
- Train mask generator: seed = run seed. **Validation mask generator:** seed =
  `(run_seed + 10**6) & 0xFFFFFFFF`, so every epoch evaluates the identical
  mask pattern — comparable val-loss curves.
- Window-capping subsets use `np.random.default_rng(seed)` with sorted indices
  → the same `max_train_windows`/`max_val_windows` picks the same windows.
- `num_workers` DataLoader uses a seeded generator (`create_dataloader`).
- Baseline harness: seeded window subsampling (`--seed 42`), `LogisticRegression
  random_state=42`.

Scope note: bit-exact GPU determinism is not guaranteed across PyTorch/CUDA
versions and hardware. The anchors above guarantee the *pipeline* is identical;
residual numeric noise between machines is expected and documented in results.

## 5. Snapshots

- `storage/training/snapshots/<date>/` (e.g. `2026-07-30`) holds a full copy of
  the canonical dataset that produced a fingerprint — the durable offline
  reference. Snapshots are tracked in git; derived DBs and eval JSONs are not.
- `storage/training/index.duckdb` and `experiment_registry.duckdb` are
  regenerable from the snapshot via the Phase 1 CLI (`DATA_FLOW.md` §9/§10).

## 6. How to reproduce a specific run

```bash
# 1. Same data
python main.py validate                # assert fingerprint == expected
python main.py snapshot --date 2026-07-30   # restore canonical data from snapshot if missing

# 2. Same code
git checkout v1.0-baseline             # or the exact manifest git_commit

# 3. Same configs
python -m src.training.train_teacher --resume models/foundation/teacher_v1/<run_id>
#    (resume enforces the run's manifest configs; running fresh with the same
#     three config files + seed 42 is equivalent for new runs)

# 4. Re-derive evaluations from the checkpoint (git-ignored, regenerable)
python -m src.evaluation.embedding.linear_probe \
  --checkpoint models/foundation/teacher_v1/<run_id> --pooling mean
python -m src.evaluation.baselines.runner \
  --checkpoint models/foundation/teacher_v1/<run_id> --pooling mean
```

## 7. What is / isn't in version control

| Path | Tracked? | Reason |
|---|---|---|
| `storage/training/*.json`, `snapshots/**` | yes | dataset identity + durable data |
| `configs/*.yaml|*.json` | yes | definitions |
| `src/`, `main.py`, `tests/`, docs | yes | code |
| `models/foundation/**` | **no** | checkpoints regenerable |
| `evaluation/embedding/**`, `evaluation/baselines/**` | **no** | eval outputs regenerable |
| `logs/`, `test_download.zip` | no | transient |

So: the *recipe* is tracked; the *outputs* (checkpoints, eval reports) are
reproduced from the recipe + checkpoint dir.

## 8. Change-control gate (when a "run" is no longer comparable)

Re-fingerprint and re-run baselines when any of these change:
- canonical schema / alignment / resampling (`DATA_FLOW.md` stages);
- the split manifest (`training_manifest_v1.json`);
- `feature_builder` feature definitions or `feature_style`;
- the masking/loss semantics (`masked_modeling.py`).

Non-impacting: doc-only changes, code formatting, tests that don't touch the
pipeline.
