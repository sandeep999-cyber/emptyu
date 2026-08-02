# Pure Market Foundation Model — Dataset Builder

Research-grade, reproducible data platform for training self-supervised market foundation models on raw Binance historical data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download raw archives (1 month of BTCUSDT)
python main.py download --symbols BTCUSDT --market futures --start-year 2024 --limit-months 1

# 2. Convert raw CSV/ZIP → canonical Snappy Parquet
python main.py convert --symbols BTCUSDT

# 3. Resample 1m → 5m, 15m, 1h, 4h, 1d
python main.py resample --symbols BTCUSDT

# 4. Run causal alignment and report coverage
python main.py align --symbols BTCUSDT

# 5. Build calendar and Data Lake views (generates per-symbol metadata)
python main.py build-lake --symbols BTCUSDT

# 6. Validate data integrity (SHA256 checksums)
python main.py validate

# 7. Generate quality_report.json per symbol
python main.py quality-report --symbols BTCUSDT

# 8. Create immutable dataset snapshot
python main.py snapshot --date 2026-07-30

# 9. Generate storage summary report
python main.py report

# 10. Benchmark DataLoader throughput
python main.py benchmark --symbols BTCUSDT

# Or via module entrypoint:
python -m src.training.benchmark --snapshot 2026-07-30 --symbols BTCUSDT
```

## Phase 2 — Teacher Foundation Model (Quickstart)

Train a self-supervised teacher encoder on frozen Phase 1 data. No labels, no trading.

```bash
# Smoke test — validates pipeline end-to-end on CPU (~minutes)
python -m src.training.train_teacher --smoke

# Full training — requires CUDA GPU (single command, ~hours)
python -m src.training.train_teacher \
  --model-config configs/model_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_v1.yaml

# Stationary returns variant (log-return features + span masking)
python -m src.training.train_teacher \
  --model-config configs/model_returns_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_returns_v1.yaml

# Resume a run — the run's own recorded configs are authoritative
python -m src.training.train_teacher --resume models/foundation/teacher_v1/<run_id>

# Evaluate embeddings (requires a completed training run)
python -m src.evaluation.embedding.clustering --checkpoint <run_dir>
python -m src.evaluation.embedding.linear_probe --checkpoint <run_dir>
python -m src.evaluation.embedding.retrieval --checkpoint <run_dir>
python -m src.evaluation.embedding.temporal_consistency --checkpoint <run_dir>
python -m src.evaluation.embedding.visualization --checkpoint <run_dir>
# clustering/retrieval/temporal/visualization accept --split {train,validation,test}
# (test is the temporal holdout after time_split.train_end)
```

- Model config: 8 layers, 8 heads, 512 hidden, 2048 FFN, RoPE, CLS pooling
- Objective: Masked Market Modeling (15% timestep masking; `model_returns_v1.yaml` uses contiguous-span masking)
- Data: Futures-only (BTCUSDT + ETHUSDT train, SOLUSDT validation), seen-once
- Splits: frozen in `storage/training/training_manifest_v1.json` — train ends `2024-11-30`, test is BTCUSDT+ETHUSDT after that date
- Checkpoints: `models/foundation/teacher_v1/<run_id>/` (see `CHECKPOINT_FORMAT.md`)
- Eval reports: `evaluation/embedding/*.json`, `evaluation/baselines/baseline_eval_*.json`, figures in `evaluation/embedding/figures/`

### Feature styles

`feature_builder.build_features` supports two layouts (see `src/data/feature_builder.py`):

- `raw` — price levels (OHLCV) + funding/OI + calendar (`trainer_v1.yaml`).
- `returns` — stationary market-only features (log returns, ranges, log volume,
  log OI) + funding + calendar input; calendar excluded from the reconstruction
  target (`trainer_returns_v1.yaml` + `model_returns_v1.yaml`).

The `feature_style` is recorded in each run's `manifest.json`. Evaluation and
baselines read it from the checkpoint so metrics stay meaningful for both
variants.

### GPU readiness

`trainer_v1.yaml` enables mixed precision on CUDA (`mixed_precision: true`, bf16
autocast + gradient scaling; runs fp32 on CPU). DataLoader workers use
`pin_memory=True` on CUDA and `persistent_workers` when `num_workers > 0`.

See `Phase2_Implementation_Plan_v1.md` for full specification.

## Documentation

The repo is self-documenting. Read in this order:

| Doc | Covers |
|---|---|
| `SYSTEM_OVERVIEW.md` | What this is, phases, principles, CLI table, repo map, reading order |
| `ARCHITECTURE.md` | 8 core principles + 4 contracts |
| `RESEARCH_PHILOSOPHY.md` | Thesis, why no labels, masking/loss design, leakage rules, non-goals |
| `DATA_FLOW.md` | Phase 1 pipeline: download → convert → resample → align → build-lake → validate → snapshot, with CLI examples |
| `DATASET.md` | Global HF-style data card |
| `CONFIG_REFERENCE.md` | Every key of every config: meaning, default, allowed values, effect |
| `MODULE_REFERENCE.md` | Every module: purpose, inputs, outputs, assumptions, side effects, failure modes, deps, config, tests |
| `TRAINING_GUIDE.md` | Phase 2 training: smoke/full/returns, resume, checkpoint lifecycle, expected outputs |
| `CHECKPOINT_FORMAT.md` | Run-dir layout, `.pt` state dict, `latest.json`, `manifest.json`, resume mechanics |
| `RESEARCH_BASELINE.md` | **Frozen ground truth** for the first GPU experiment + success criteria; compare every experiment against it |
| `EVALUATION_GUIDE.md` | Embedding probes (clustering/retrieval/linear_probe/temporal/viz) + baselines harness, how to read results |
| `MODEL_CARD.md` | `teacher_transformer_v1` specs, data, objective, current eval results, intended use |
| `REPRODUCIBILITY.md` | Fingerprint, manifest, configs, git, seeds — the four anchors |
| `GPU_TRAINING_GUIDE.md` | CUDA host: setup, data, run, resume, failure recovery |
| `COLAB_GUIDE.md` | Cell-by-cell notebook walkthrough for managed GPU |
| `TROUBLESHOOTING.md` | Failure-mode catalog across all areas |
| `DOCUMENTATION_AUDIT.md` | Audit of what existed, what was created, known divergences |
| `Phase1_Implementation_Plan_v4.md`, `Phase2_Implementation_Plan_v1.md`, `walkthrough.md` | Historical design/remediation records (authoritative code = `src/`) |

## Project Structure

```
├── configs/
│   ├── download.yaml              # Download settings
│   ├── storage.yaml               # Storage paths
│   ├── validation.yaml            # Validation rules per modality
│   ├── dataset.yaml               # Dataset metadata
│   ├── modalities_v1.yaml         # Modality registry (active/inactive)
│   ├── alignment_v1.yaml          # Versioned alignment contract
│   ├── market_state_schema_v1.json # Feature order + dimension
│   ├── windowing_v1.yaml          # Window specification
│   ├── model_v1.yaml              # Teacher architecture (Phase 2)
│   ├── model_returns_v1.yaml      # Span-mask variant (Phase 2)
│   ├── optimizer_v1.yaml          # AdamW + LR schedule (Phase 2)
│   ├── trainer_v1.yaml            # Training loop config (Phase 2)
│   └── trainer_returns_v1.yaml    # Returns-feature variant (Phase 2)
├── src/
│   ├── config.py                  # Central config loader
│   ├── logger.py                  # Stage loggers → logs/
│   ├── data/
│   │   ├── ...                    # Phase 1 data pipeline (unchanged)
│   │   └── windowing.py           # Fixed-length sequence windowing
│   ├── models/
│   │   └── teacher/               # Teacher Foundation Model (Phase 2)
│   │       ├── projection.py      # Feature project + reconstruction heads
│   │       ├── positional_encoding.py # Rotary Position Embeddings
│   │       ├── transformer.py     # Pre-LN transformer block
│   │       ├── encoder.py         # Stacked encoder
│   │       └── embeddings.py      # Pooling + extraction API
│   ├── training/
│   │   ├── benchmark.py           # Throughput/RAM/CPU benchmarks
│   │   ├── dataloader.py          # PyTorch DataLoader wrapper
│   │   ├── experiment_registry.py # Experiment tracking (DuckDB)
│   │   ├── normalizer.py          # z-score/log/robust normalization
│   │   ├── sampler.py             # Epoch-level shuffling
│   │   ├── seeding.py             # Global RNG seeding
│   │   ├── train_teacher.py       # CLI entry (Phase 2)
│   │   ├── trainer.py             # Training loop (Phase 2)
│   │   ├── checkpoint.py          # Versioned checkpoints (Phase 2)
│   │   ├── optimizer.py           # AdamW factory (Phase 2)
│   │   ├── scheduler.py           # Warmup + cosine (Phase 2)
│   │   └── losses/
│   │       ├── masked_modeling.py # Masked Market Modeling objective
│   │       ├── contrastive.py     # Placeholder (Phase 3+)
│   │       └── temporal.py        # Placeholder (Phase 3+)
│   └── evaluation/
│       ├── embedding/             # Embedding evaluation suite (Phase 2)
│       │   ├── clustering.py      # KMeans + regime separation
│       │   ├── retrieval.py       # kNN retrieval + cross-symbol
│       │   ├── linear_probe.py    # Frozen-encoder probing
│       │   ├── temporal_consistency.py # Adjacent-embedding smoothness
│       │   └── visualization.py   # PCA/t-SNE + loss curves
│       └── baselines/             # Baseline eval harness (Phase A)
├── tests/                         # Phase 1 + Phase 2 test suites (tests/unit/)
├── storage/
│   ├── raw/                       # Raw Binance archives
│   ├── canonical/                 # Snappy Parquets + per-symbol metadata
│   ├── lake/views/                # Virtualized aligned views
│   └── training/                  # Snapshots, manifests, DuckDB index
│       ├── training_manifest_v1.json  # Frozen split contract (tracked)
│       ├── dataset_fingerprint.json   # Dataset fingerprint (tracked)
│       └── snapshots/                 # Immutable snapshot ledger (tracked)
├── models/foundation/             # Teacher checkpoints (Phase 2)
├── evaluation/embedding/          # Eval reports + figures + embeddings
├── logs/                          # Per-stage log files
├── ARCHITECTURE.md                # 8 Core Principles + 4 Contracts
├── DATASET.md                     # Global HF-style Data Card
├── SYSTEM_OVERVIEW.md             # Repo-wide orientation + reading order
├── RESEARCH_PHILOSOPHY.md         # Thesis + design rationale
├── DATA_FLOW.md                   # Phase 1 pipeline guide
├── CONFIG_REFERENCE.md            # Every config key documented
├── MODULE_REFERENCE.md            # Every module documented
├── TRAINING_GUIDE.md              # Phase 2 training guide
├── CHECKPOINT_FORMAT.md           # Checkpoint/resume format
├── EVALUATION_GUIDE.md            # Eval probes + baselines harness
├── MODEL_CARD.md                  # Model spec + eval results
├── RESEARCH_BASELINE.md           # Frozen ground truth + success criteria
├── REPRODUCIBILITY.md             # Fingerprint/manifest/git/seed anchors
├── GPU_TRAINING_GUIDE.md          # CUDA host training
├── COLAB_GUIDE.md                 # Notebook walkthrough
├── TROUBLESHOOTING.md             # Failure-mode catalog
├── DOCUMENTATION_AUDIT.md         # Documentation audit deliverable
├── main.py                        # CLI entrypoint
├── Phase1_Implementation_Plan_v4.md  # Historical design record
├── Phase2_Implementation_Plan_v1.md  # Historical design record
├── walkthrough.md                 # Historical remediation record
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v
```

203 tests, all passing, covering:
- Causality/no-leakage property tests across all 4 Phase 1 modalities
- Resampler OHLCV mathematical accuracy
- DuckDB file index and asset registry queries
- Parquet converter provenance embedding
- Feature builder schema compliance (raw + returns layouts)
- Windowing gap validation
- Normalizer train-split enforcement (z-score, log, robust)
- Modality registry schema validation
- Validator rules for all modalities (klines, funding, aggTrades, depth, liquidations)
- Snapshot immutability and dataset fingerprint determinism
- DataLoader throughput benchmarks
- **Phase 2**: Transformer shapes, RoPE shift-equivariance, CLS/padding invariants
- **Phase 2**: Mask generator correctness, CLS exclusion, per-modality loss
- **Phase 2**: Checkpoint save/load roundtrip, manifest integrity, resume (incl. val-mask state)
- **Phase 2**: Training step determinism, normalizer train-split enforcement
- **Phase 2**: Eval split time bounds (train capped / test floored at `time_split.train_end`)
- **Phase 2**: Resume uses the run's recorded configs as authoritative
- **Phase 2**: Feature-style-aware window stats (linear probe + baselines)
- **Phase 2**: Embedding pooling shapes, extraction API
- **Phase 2**: Linear probe beats majority baseline on synthetic data

## Phase 1 Modalities

| Modality | Frequency | Alignment | Status |
|---|---|---|---|
| klines (OHLCV) | 1m | identity | Active |
| funding_rate | 8h | forward_fill | Active |
| open_interest | 5m | forward_fill | Active |
| calendar | 1m | derived | Active |
| agg_trades | variable | aggregate_to_minute | Declared (Phase 2) |
| depth | variable | snapshot_last_before_close | Declared (Phase 3) |
| liquidations | variable | aggregate_to_minute | Declared (Phase 3) |
