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

# Evaluate embeddings (requires a completed training run)
python -m src.evaluation.embedding.clustering --checkpoint <run_dir>
python -m src.evaluation.embedding.linear_probe --checkpoint <run_dir>
python -m src.evaluation.embedding.retrieval --checkpoint <run_dir>
python -m src.evaluation.embedding.temporal_consistency --checkpoint <run_dir>
python -m src.evaluation.embedding.visualization --checkpoint <run_dir>
```

- Model config: 8 layers, 8 heads, 512 hidden, 2048 FFN, RoPE, CLS pooling
- Objective: Masked Market Modeling (15% random timestep masking)
- Data: Futures-only (BTCUSDT + ETHUSDT train, SOLUSDT validation), seen-once
- Checkpoints: `models/foundation/teacher_v1/<run_id>/`
- Embeddings: `evaluation/embedding/embeddings/<run_id>/`

See `Phase2_Implementation_Plan_v1.md` for full specification.

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
│   ├── optimizer_v1.yaml          # AdamW + LR schedule (Phase 2)
│   └── trainer_v1.yaml            # Training loop config (Phase 2)
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
│       └── embedding/             # Embedding evaluation suite (Phase 2)
│           ├── clustering.py      # KMeans + regime separation
│           ├── retrieval.py       # kNN retrieval + cross-symbol
│           ├── linear_probe.py    # Frozen-encoder probing
│           ├── temporal_consistency.py # Adjacent-embedding smoothness
│           └── visualization.py   # PCA/t-SNE + loss curves
├── tests/                         # 55 Phase 2 unit tests (tests/unit/) + Phase 1 tests
├── storage/
│   ├── raw/                       # Raw Binance archives
│   ├── canonical/                 # Snappy Parquets + per-symbol metadata
│   ├── lake/views/                # Virtualized aligned views
│   └── training/                  # Snapshots, manifests, DuckDB index
├── models/foundation/             # Teacher checkpoints (Phase 2)
├── evaluation/embedding/          # Eval reports + figures + embeddings
├── logs/                          # Per-stage log files
├── ARCHITECTURE.md                # 8 Core Principles + 4 Contracts
├── DATASET.md                     # Global HF-style Data Card
├── main.py                        # CLI entrypoint
├── Phase2_Implementation_Plan_v1.md
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v
```

55 Phase 2 unit tests (all passing) covering:
- Causality/no-leakage property tests across all 4 Phase 1 modalities
- Resampler OHLCV mathematical accuracy
- DuckDB file index and asset registry queries
- Parquet converter provenance embedding
- Feature builder schema compliance
- Windowing gap validation
- Normalizer train-split enforcement (z-score, log, robust)
- Modality registry schema validation
- Validator rules for all modalities (klines, funding, aggTrades, depth, liquidations)
- Snapshot immutability and dataset fingerprint determinism
- DataLoader throughput benchmarks
- **Phase 2**: Transformer shapes, RoPE shift-equivariance, CLS/padding invariants
- **Phase 2**: Mask generator correctness, CLS exclusion, per-modality loss
- **Phase 2**: Checkpoint save/load roundtrip, manifest integrity, resume
- **Phase 2**: Training step determinism, normalizer train-split enforcement
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
