# Implementation Plan — Phase 1: Pure Market Foundation Model
## Dataset Builder & Data Lake Engine

Build a research-grade, high-throughput, self-contained dataset builder, DuckDB
Data Lake, and PyTorch-native `MarketDataset` API to acquire, verify, resample,
canonicalize, align, index, snapshot, and stream raw market data (USD-M Futures
and Spot, active and delisted) for self-supervised transformer training.

---

## 1. Transformer Input Pipeline & Terminology

```
Market State (Minute t)
       │
       ▼
Alignment Engine (causal, versioned per-modality contracts)
       │
       ▼
Modality Registry (which aligned modalities are active)
       │
       ▼
Feature Builder (versioned — fuses active modalities into one vector per minute)
       │
       ▼
Windowing (versioned — cuts the continuous timeline into sequences)
       │
       ▼
MarketDataset (PyTorch interface — exposes, never engineers, features)
       │
       ▼
Normalizer (training-layer, optional — z-score / log / robust scaling)
       │
       ▼
Projection Layer (linear projection to hidden dimension D)
       │
       ▼
Token Embedding (input tensor to Transformer blocks)
```

**Market State** — everything that exists at time *t* across all collected
modalities (OHLCV, Funding, OI, AggTrades, Depth, Liquidations, Calendar).

**Alignment Engine** — resolves every modality onto the shared 1-minute
timeline under a causal, no-leakage contract (see §3). Operates on the full
Market State, independent of which modalities any given model consumes.
Modality-specific behavior (e.g. depth snapshotting, trade aggregation,
liquidation aggregation) is declared through versioned alignment policies,
while the engine itself provides shared execution primitives — some
modalities genuinely need specialized handling, and the architecture allows
that without falling back to ad hoc per-modality code.

**Modality Registry** — a versioned, standalone artifact declaring which
aligned modalities are active for a given training configuration. Storage and
alignment are always richer than any single model; the registry is what lets
an encoder declare a subset without touching the pipeline.

**Feature Builder** — fuses the modalities selected by the active registry,
post-alignment, into one continuous feature vector per minute. Versioned
(§4a), since reordering features or adding a calendar field changes what an
embedding means. A data-layer responsibility, not part of the PyTorch dataset
interface.

**Windowing** — cuts the aligned, feature-built timeline into fixed-length
sequences. Versioned and deterministic (§4b) — two runs against the same
`windowing_v1.yaml` produce byte-identical windows, so this lives in the data
layer alongside `market_dataset.py`, not in the training layer.

**MarketDataset** — a thin PyTorch interface over already-built, already-cut
feature windows. It exposes samples; it does not compute, engineer, or cut
them.

**Normalizer** — a training-layer stage, distinct from the dataset, between
`MarketDataset` and the projection layer. Statistics (mean/std, etc.) are
always fit on the `train` split only, per snapshot, and stored — never
computed against validation/test data, which would leak distributional
information across the split boundary the same way an uncausal alignment
rule would leak across time.

### Generic MarketDataset Output Contract

```python
sample = dataset[i]
# returns {
#     "features": Tensor,      # Shape: [seq_len, feature_dim]
#     "feature_mask": Tensor,  # Shape: [seq_len, feature_dim] — per-feature validity (bool/int8)
#     "timestamps": Tensor,    # Shape: [seq_len]
#     "mask": Tensor,          # Shape: [seq_len] — per-timestep validity
#     "metadata": dict         # symbol, start_ts, end_ts, snapshot_id, modality_config, windowing_config
# }
```

`feature_mask` is separate from the per-timestep `mask`: a timestep can be
present while individual sparse modalities within it (depth, liquidations)
are unavailable. Stored as `bool`/`int8`, not float — it doubles tensor
storage otherwise. This becomes load-bearing once Phase 2/3 activate sparser
modalities.

---

## 2. Calendar Schema Definition

Mathematical temporal vector, free of human session labels (no Asia/London/NY
constructs — the model infers those if useful):

- `minute_of_day` (0–1439)
- `hour` (0–23)
- `day_of_week` (0–6)
- `day_of_month` (1–31)
- `month` (1–12)
- `quarter` (1–4)
- `year` (YYYY)
- `is_weekend` (0/1)

---

## 3. Alignment Contract (versioned, per modality)

`alignment.py` is a **standalone, modality-agnostic engine** — not a
sub-routine of `lake.py`. It executes policy declared in a **versioned
contract file**, `alignment_v1.yaml`, rather than hardcoding rules in code.
If a better depth aggregation strategy is found later, it becomes
`alignment_v2.yaml` — old experiments stay reproducible against the contract
version they were run with, exactly as `modalities_v1.yaml` is versioned
separately from any one training manifest.

`alignment_v1.yaml`:

```yaml
funding:
  frequency: 8h
  alignment: forward_fill
  known_at: settlement_time
  interpolation: false

open_interest:
  frequency: 5m
  alignment: forward_fill
  interpolation: false

agg_trades:
  frequency: variable
  alignment: aggregate_to_minute
  known_at: exchange_event_time      # not ingestion/capture time
  aggregation:
    - trade_count
    - base_volume
    - quote_volume
    - taker_buy_volume
    - taker_sell_volume
    - taker_buy_count
    - taker_sell_count

depth:
  frequency: variable
  alignment: snapshot_last_before_close
  known_at: exchange_event_time      # falls back to capture_time if unavailable, flagged in metadata

liquidations:
  frequency: variable
  alignment: aggregate_to_minute
  known_at: exchange_event_time

klines:
  frequency: 1m
  alignment: identity

calendar:
  frequency: 1m
  alignment: derived
```

**Causality rule (binding on every modality):**
> At minute *t*, a modality's aligned value is the most recent observation
> whose `known_at` timestamp is at or before *t*. Never the next observation.
> Never interpolated. Never a value whose true event time is after *t*.

This applies identically to funding (most recently settled rate) and open
interest (most recent observation, no interpolation between readings).

### Missing Data Policy

Declared explicitly per modality — no implicit assumptions:

```yaml
missing:
  funding:
    policy: forward_fill
  open_interest:
    policy: forward_fill
  depth:
    policy: null            # no synthetic depth; mark unavailable in feature_mask
  agg_trades:
    policy: zeros
  liquidations:
    policy: zeros
```

### AggTrades Statistics Schema (aggregation output)

- `trade_count`
- `base_volume`
- `quote_volume`
- `taker_buy_volume`
- `taker_sell_volume`
- `taker_buy_count`
- `taker_sell_count`

---

## 4. Modality Registry (standalone versioned artifact)

Lives at `configs/modalities_v1.yaml`, independent of any single training run,
so multiple manifests can reference the same registry version:

```yaml
price:
  enabled: true
funding:
  enabled: true
open_interest:
  enabled: true
calendar:
  enabled: true
agg_trades:
  enabled: false
depth:
  enabled: false
liquidations:
  enabled: false
```

Phase progression is a registry change, not a pipeline change:

| Phase | Active modalities |
|---|---|
| 1 | price, funding, open_interest, calendar |
| 2 | + agg_trades |
| 3 | + depth |

### 4a. Feature Order & Dimension Metadata (versioned)

`configs/market_state_schema_v1.json` — the model must never guess tensor
ordering. This is itself a versioned artifact (`feature_builder_version:
v1`), since reordering or extending it changes what every downstream
embedding means:

```json
{
  "feature_builder_version": "v1",
  "feature_order": [
    "open", "high", "low", "close", "volume",
    "funding_rate",
    "open_interest",
    "minute_of_day", "hour", "day_of_week", "day_of_month",
    "month", "quarter", "year", "is_weekend"
  ],
  "feature_dimension": 15
}
```

### 4b. Window Specification (versioned)

`configs/windowing_v1.yaml` — separates *what data exists* from *how it's
presented to the model*:

```yaml
sequence_length: 512
stride: 1
prediction_horizon: 0

padding:
  enabled: false

sampling: contiguous

drop_incomplete_windows: true
```

---

## 5. Dataset Version Locking

`training_manifest_v1.json` pins every artifact version an experiment
depends on, so it's never accidentally retrained against a newer alignment
engine, feature builder, windowing spec, schema, or registry:

```yaml
training_manifest:
  dataset:
    snapshot: 2026-07-30
    alignment_version: alignment_v1.yaml
    feature_builder_version: v1
    windowing_version: windowing_v1.yaml
    canonical_schema_version: v1
    market_state_schema_version: v1
    modality_registry: modalities_v1.yaml
  splits:
    train:
      symbols: [BTCUSDT, ETHUSDT]
    validation:
      symbols: [SOLUSDT]
    test:
      symbols: [DOGEUSDT]
    time_split:
      train_end: 2024-12-31
  random_seed:
    python: 42
    numpy: 42
    torch: 42
```

---

## 6. Data Provenance (attached to every canonical Parquet)

Every canonical Parquet file carries embedded metadata so any single file can
be traced back to its origin without cross-referencing other artifacts:

```json
{
  "created_by": "parquet_converter.py",
  "source": "binance_vision_archive",
  "checksum": "sha256:...",
  "download_date": "2026-07-28",
  "converter_version": "v1",
  "alignment_version": "alignment_v1.yaml",
  "schema_version": "canonical_schema_v1",
  "snapshot": "2026-07-30"
}
```

---

## 7. Multi-Tiered Data Lake Storage Layout

```
storage/
├── raw/                              # 100% untouched raw Binance archives & checksums
│   ├── futures/
│   │   ├── BTCUSDT/
│   │   │   ├── klines/1m/           # 1-minute raw klines ONLY
│   │   │   ├── aggTrades/
│   │   │   ├── trades/
│   │   │   ├── funding/
│   │   │   ├── open_interest/
│   │   │   ├── depth/
│   │   │   └── liquidations/        # Force-liquidation order snapshots
│   │   └── ...
│   └── spot/
├── canonical/                          # Clean Snappy Parquets matching raw Binance schemas
│   ├── futures/
│   │   ├── BTCUSDT/
│   │   │   ├── klines/
│   │   │   │   ├── 1m/               # Direct conversion from raw (provenance embedded)
│   │   │   │   ├── 5m/               # Resampled from 1m
│   │   │   │   ├── 15m/              # Resampled from 1m
│   │   │   │   ├── 1h/               # Resampled from 1m
│   │   │   │   ├── 4h/               # Resampled from 1m
│   │   │   │   └── 1d/               # Resampled from 1m
│   │   │   ├── aggTrades/
│   │   │   ├── trades/
│   │   │   ├── funding/
│   │   │   ├── open_interest/
│   │   │   ├── depth/
│   │   │   ├── liquidations/
│   │   │   ├── exchange_info/        # Historical contract specs (2022.json, 2023.json, ...)
│   │   │   └── metadata/
│   │   │       ├── dataset_version.json
│   │   │       ├── raw_schema_v1.json
│   │   │       ├── canonical_schema_v1.json
│   │   │       ├── market_state_schema_v1.json  # feature_order + feature_dimension (§4a)
│   │   │       ├── statistics_v1.json
│   │   │       ├── calendar_v1.parquet
│   │   │       ├── quality_report.json           # [NEW] see §9a
│   │   │       └── DATASET.md
│   │   └── ...
│   └── spot/
├── lake/                             # Logical Data Lake layer (virtualized, aligned market_state views)
│   └── views/
├── training/                         # Indexing, caching, snapshots, manifests, experiment metadata
│   ├── index.duckdb                  # Unified DuckDB database (file_index_v1 & asset_registry)
│   ├── experiment_registry.duckdb    # experiment_id, snapshot, alignment_version, feature_builder_version,
│   │                                 # windowing_version, modality_registry, objective, encoder, loss,
│   │                                 # seed, git_commit, hardware, software, metrics
│   ├── snapshots/                    # Immutable reproducible dataset snapshots
│   │   └── 2026-07-30/ -> manifest, checksums, stats
│   ├── training_manifest_v1.json     # Version-locked: see §5
│   ├── dataset_fingerprint.json
│   └── cache/
models/
└── foundation/
evaluation/                           # [NEW] Architecture placeholder — no implementation in Phase 1
└── embedding/                        # Future: linear probing, clustering quality, regime separation,
                                       # temporal consistency, downstream prediction benchmarks
```

---

## 8. Proposed Changes & Module Architecture

Data-engineering code (deterministic, versioned, training-strategy-agnostic)
is separated from ML/training code (which varies per run) at the top level of
`src/`:

```
d:/emptyu/
├── configs/
│   ├── download.yaml
│   ├── storage.yaml
│   ├── validation.yaml
│   ├── dataset.yaml
│   ├── modalities_v1.yaml            # standalone modality registry
│   ├── alignment_v1.yaml             # versioned alignment contract
│   ├── market_state_schema_v1.json   # feature_order + feature_dimension (§4a)
│   └── windowing_v1.yaml             # [NEW] versioned window specification (§4b)
├── logs/
│   ├── download.log
│   ├── convert.log
│   ├── resample.log
│   ├── alignment.log
│   ├── validator.log
│   └── errors.log
├── models/
│   └── foundation/
├── evaluation/
│   └── embedding/                    # [NEW] placeholder — reserved, not implemented in Phase 1
├── requirements.txt
├── README.md
├── ARCHITECTURE.md                   # Core Architecture Principles & 4 Contracts
├── DATASET.md                        # Global HF-style Data Card
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/                         # [REORGANIZED] deterministic data pipeline — no training strategy
│   │   ├── __init__.py
│   │   ├── db.py                     # Unified DuckDB database manager
│   │   ├── binance_vision.py         # 1m archive downloader (Futures & Spot, active & delisted)
│   │   ├── binance_rest.py           # REST client for exchangeInfo history, funding, open interest
│   │   ├── resampler.py              # Resampler engine: 1m -> 5m, 15m, 1h, 4h, 1d Parquet
│   │   ├── parquet_converter.py      # CSV -> Snappy Parquet converter, embeds provenance (§6)
│   │   ├── alignment.py              # Causal alignment engine; executes alignment_v1.yaml
│   │   ├── modality_registry.py      # Loads/validates modalities_v1.yaml
│   │   ├── feature_builder.py        # Versioned (§4a): fuses active modalities into feature vectors
│   │   ├── windowing.py              # [NEW] Versioned (§4b): cuts timeline into fixed-length sequences
│   │   ├── validator.py              # Modality validation rules, SHA256 integrity, gap analysis
│   │   ├── quality_report.py         # [NEW] Generates quality_report.json (§9a)
│   │   ├── metadata.py               # dataset_version.json, statistics_v1.json, versioned schemas
│   │   ├── calendar_builder.py       # calendar_v1.parquet temporal matrix builder
│   │   ├── lake.py                   # Logical Data Lake manager (consumes alignment.py output)
│   │   ├── market_dataset.py         # PyTorch-native MarketDataset — exposes windows, never builds them
│   │   ├── manifest_builder.py       # training_manifest_v1.json generator; writes version locks (§5)
│   │   ├── snapshot_manager.py       # Immutable snapshot creator
│   │   └── datacard_builder.py       # Hugging Face-style DATASET.md generator
│   └── training/                     # [REORGANIZED] training-strategy-specific, varies per run
│       ├── __init__.py
│       ├── normalizer.py             # Optional z-score/log/robust scaling; train-split-only stats
│       ├── dataloader.py             # [NEW] Batching
│       ├── sampler.py                # [NEW] Epoch-level sampling strategy (shuffling, curriculum, etc.)
│       ├── experiment_registry.py    # Reads/writes experiment_registry.duckdb
│       └── benchmark.py              # Throughput/resource benchmark suite (§9b)
├── reports.py                        # Reports (validation, missing data, storage summary)
└── tests/
    ├── test_binance_vision.py
    ├── test_resampler.py
    ├── test_db_duckdb.py
    ├── test_parquet_converter.py
    ├── test_alignment.py             # causality / no-leakage property tests
    ├── test_modality_registry.py
    ├── test_feature_builder.py
    ├── test_windowing.py             # [NEW]
    ├── test_quality_report.py        # [NEW]
    ├── test_validator.py
    ├── test_metadata.py
    ├── test_calendar.py
    ├── test_lake.py
    ├── test_market_dataset.py
    ├── test_normalizer.py            # asserts stats fit on train split only
    ├── test_manifest.py
    ├── test_experiment_registry.py
    └── test_snapshot.py
```

### `ARCHITECTURE.md` — 8 Core Principles

1. Raw data is immutable.
2. Canonical preserves semantics.
3. Alignment is causal.
4. Storage is richer than any encoder.
5. No hand-crafted indicators.
6. Modalities are additive.
7. Models never modify datasets.
8. Every experiment is reproducible.

Four system contracts: **Raw Contract**, **Canonical Contract**, **Alignment
Contract**, **Model Contract** (see §3–5 above for Alignment and Model).

### `src/data/alignment.py`

Executes the versioned policy in `alignment_v1.yaml` against the canonical
layer. Modality-specific behavior is declared through versioned alignment
policies, while the engine provides shared execution primitives — specialized
kernels (depth snapshotting, trade aggregation, liquidation aggregation) are
expected, not avoided; what's disallowed is undeclared, undocumented
special-casing. For Phase 1, only the four active modalities need working
logic (`identity`, `forward_fill` ×2, `derived`); `agg_trades`, `depth`, and
`liquidations` have their contract declared now but can raise
`NotImplementedError` until Phase 2/3 activate them.

### `src/data/validator.py`

Extensible modality validation rules:
- `funding`: monotonic timestamp, no missing values.
- `klines`: high >= low, close > 0.
- `aggTrades`: trade_count >= 0, volume >= 0.
- `depth`: best_bid <= best_ask, snapshot completeness.
- `liquidations`: side in {BUY, SELL}, quantity > 0.

### Symbol Metadata (asset registry)

`db.py`'s `asset_registry` table stores static per-symbol metadata alongside
the file index — already largely available from `exchangeInfo`, useful for
any future cross-symbol work:

- base asset / quote asset
- market type (futures/spot)
- listing date / delisting date
- contract size
- tick size
- lot size

---

## 9. Verification Plan

### 9a. Data Quality Report

`quality_report.json` — generated per snapshot, the first thing to inspect
before training:

- missing values per modality
- forward-fill percentages
- alignment coverage
- duplicate rows
- gap counts
- symbols with incomplete history
- resampling statistics

### Automated Tests (correctness)

```bash
python -m pytest tests/ -v
```

Test cases:
- Resampler accuracy (1m -> 5m, 15m, 1h, 4h, 1d OHLCV consistency).
- DuckDB `file_index` & `asset_registry` queries.
- **Causality / no-leakage property test (`test_alignment.py`):** perturb a
  future-dated observation in a modality (e.g. shift a funding rate's value)
  and assert no token timestamped before that observation's `known_at`
  changes. Run across all four Phase 1 modalities.
- Modality registry loads, validates against schema, rejects unknown keys.
- Feature builder output matches `feature_order` / `feature_dimension` from
  `market_state_schema_v1.json` exactly, tagged with `feature_builder_version`.
- Windowing produces sequences matching `windowing_v1.yaml` (length, stride,
  dropped-incomplete-window behavior).
- Modality validation rules execution.
- PyTorch `MarketDataset` sample dict structure (features, feature_mask,
  timestamps, mask, metadata).
- **Normalizer test:** statistics fit only on `train`-split data; asserts
  fitting against a manifest with only `validation`/`test` symbols raises.
- `training_manifest_v1.json` correctly resolves and pins all version-locked
  artifacts (§5): snapshot, alignment, feature builder, windowing, schema,
  registry.
- Experiment registry: writing and querying an experiment record round-trips
  all fields, including hardware/software environment.
- `quality_report.json` generation matches expected structure against a
  synthetic dataset with known gaps.

### 9b. Benchmark Suite (performance — lower priority than correctness)

```bash
python -m src.training.benchmark --snapshot 2026-07-30
```

Metrics: samples/sec, windows/sec, RAM usage, CPU usage, DuckDB query time,
DataLoader throughput, cache hit rate. Establishes a baseline before Phase 2
training starts; not a blocker for freezing Phase 1's design.

### End-to-End Execution

```bash
python main.py download --symbols BTCUSDT --market futures --start-year 2024 --limit-months 1
python main.py convert
python main.py resample
python main.py align
python main.py build-lake
python main.py validate
python main.py quality-report
python main.py snapshot
python main.py report
python main.py benchmark
```

---

## Deferred to Phase 2

- `websocket_depth.py` (live L2 order book recorder) — historical
  reproducibility is the Phase 1 focus; live collection moves to Phase 2.
- Full `alignment_v1.yaml` logic for `agg_trades`, `depth`, `liquidations` —
  contract declared now, implementation lands when each modality is activated
  in the registry.
- `evaluation/embedding/` implementation (linear probing, clustering quality,
  regime separation, temporal consistency, downstream prediction benchmarks)
  — placeholder only in Phase 1.

## Explicitly Out of Scope (not just deferred)

To keep Phase 1 focused on historical, reproducible dataset construction:

- Automatic feature engineering / technical indicators
- RL infrastructure
- Hyperparameter search
- Distributed storage, Kubernetes, Spark, Kafka
- Online inference
