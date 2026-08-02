# Config Reference — Every Key, Its Meaning, Default, and Effect

All 13 config files live in `configs/`. This reference covers **every key**:
its meaning, the default value, allowed values, and — where relevant — its
effect on training or data output.

Loading is centralized in `src/config.py`. Phase 1 configs are read at import
time by `Config`; Phase 2 configs (`model_*`, `optimizer_*`, `trainer_*`) are
loaded by `train_teacher.py` and **recorded verbatim in each run's
`manifest.json`** (the run's recorded configs are the source of truth on
resume).

---

## Phase 1 — data pipeline configs

### 1. `download.yaml`

Controls Binance download and REST clients.

| Key | Meaning | Default | Allowed values | Effect |
|---|---|---|---|---|
| `base_url` | Binance Vision base URL | `https://data.binance.vision/data` | any https URL | where archives are fetched |
| `rest_url_futures` | USD-M Futures REST base | `https://fapi.binance.com` | any https URL | exchangeInfo / funding / OI history |
| `rest_url_spot` | Spot REST base | `https://api.binance.com` | any https URL | exchangeInfo |
| `concurrency_limit` | max parallel downloads | `8` | int ≥ 1 | download speed / polite rate |
| `timeout_seconds` | HTTP timeout | `30` | int > 0 | REST + archive requests |
| `retry_attempts` | download retries | `5` | int ≥ 0 | robustness on flaky network |
| `retry_backoff_factor` | backoff multiplier | `1.5` | float ≥ 1 | delay between retries |
| `verify_checksum` | SHA256 verification of ZIPs | `true` | bool | if true, mismatched ZIPs are deleted |
| `markets` | markets to consider | `[futures, spot]` | `futures`, `spot` | which REST/metadata paths |
| `datasets` | dataset types to download/convert | `[klines, aggTrades, trades, funding, open_interest, depth, liquidations]` | any of those | which modalities are acquired |
| `klines_interval` | interval to download | `1m` | any Binance kline interval | what raw klines are fetched |

### 2. `storage.yaml`

Paths for every tier. All are resolved relative to the repo root by
`src/config.py`.

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `base_dir` | root of all storage | `storage` | anchor |
| `raw_dir` | raw archives | `storage/raw` | where `download` writes |
| `canonical_dir` | canonical Parquet | `storage/canonical` | where `convert`/`resample`/`build-lake` write |
| `lake_dir` | lake views | `storage/lake` | `build-lake` views |
| `training_dir` | indexes/manifests/snapshots | `storage/training` | manifests, fingerprints, DuckDB |
| `models_dir` | checkpoints root | `models` | `models/foundation/teacher_v1/...` |
| `logs_dir` | stage logs | `logs` | `src/logger.py` output |
| `parquet_compression` | Parquet codec | `snappy` | file size/speed trade-off |
| `db_path` | file index DuckDB | `storage/training/index.duckdb` | `file_index_v1` + `asset_registry` |
| `experiment_db_path` | experiment registry DuckDB | `storage/training/experiment_registry.duckdb` | Phase 2 run log |

### 3. `validation.yaml`

Declarative integrity rules for `src/data/validator.py`.

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `strict_chronological` | timestamps must be monotonic | `true` | validate failure on disorder |
| `allow_duplicate_timestamps` | permit duplicate ts | `false` | validate failure on dups |
| `max_timestamp_gap_seconds` | max allowed gap | `300` | validate failure on larger gaps |
| `verify_sha256` | integrity check on/off | `true` | `main.py validate` |
| `modalities.<name>.*` | per-modality rules (monotonic_timestamp, allow_missing, high_gte_low, close_positive, trade_count_gte_zero, volume_gte_zero, bid_lte_ask, snapshot_complete, valid_side, quantity_gt_zero) | see file | which rules run for klines/funding/aggTrades/depth/liquidations |

### 4. `dataset.yaml`

Dataset-level metadata, default symbols, and the manifest split defaults.

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `name` | dataset name | `PureMarketFoundationModelDataset` | label |
| `version` | dataset version | `1.0.0` | label |
| `schema_version` | market-state schema version | `v1` | label |
| `default_symbols` | CLI default symbols | `[BTCUSDT, ETHUSDT, SOLUSDT]` | used when `--symbols` omitted |
| `default_market` | CLI default market | `futures` | used when `--market` omitted |
| `snapshot_date` | snapshot label | `2026-07-30` | provenance + snapshot dir |
| `train_symbols` | train split symbols | `[BTCUSDT, ETHUSDT]` | snapshot split |
| `val_symbols` | validation split symbols | `[SOLUSDT]` | snapshot split |
| `test_symbols` | test split symbols | `[BTCUSDT, ETHUSDT]` | snapshot split |
| `train_end_date` | temporal split boundary | `2024-11-30` | train capped / test floored at this date |

### 5. `modalities_v1.yaml`

The versioned **modality registry** (`src/data/modality_registry.py`).

| Key | Meaning | Default (shipped) | Allowed values | Effect |
|---|---|---|---|---|
| `<modality>.enabled` | is the modality active | klines/funding/open_interest/calendar `true`; agg_trades/depth/liquidations `false` | bool | enabled set defines the feature vector's semantic content |

Registry validation rejects unknown modalities and non-bool `enabled`.

### 6. `alignment_v1.yaml`

The versioned **alignment contract** executed by `src/data/alignment.py`.

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `shift_to_known_at` | shift klines ts to `close_time` (known-at semantics) | `true` | makes klines timestamp mean "information available at close" |
| `funding.frequency` / `.alignment` / `.known_at` / `.interpolation` | 8h forward-fill from `settlement_time`, no interpolation | as shown | causal resolution of funding onto 1m timeline |
| `open_interest.frequency` / `.alignment` / `.interpolation` | 5m forward-fill | as shown | causal resolution of OI |
| `agg_trades`, `depth`, `liquidations` | declared contracts | — | inactive → raise `NotImplementedError` if data passed |
| `klines.alignment` | identity | `identity` | klines define the timeline |
| `calendar.alignment` | derived | `derived` | calendar generated inline or merged |
| `missing.<modality>.policy` | missing-data policy | funding/OI `forward_fill`; depth `null`; agg_trades/liquidations `zeros` | what the aligned output contains when absent |

### 7. `market_state_schema_v1.json`

The versioned feature schema — **the model must never guess column order.**

| Key | Meaning | Value |
|---|---|---|
| `feature_builder_version` | schema version | `v1` |
| `feature_order` | the 15-dim layout | open, high, low, close, volume, funding_rate, open_interest, minute_of_day, hour, day_of_week, day_of_month, month, quarter, year, is_weekend |
| `feature_dimension` | feature width | `15` |

`FeatureBuilder` raises if `len(feature_order) != feature_dimension`.

### 8. `windowing_v1.yaml`

The versioned window spec (`src/data/windowing.py`).

| Key | Meaning | Default | Allowed values | Effect |
|---|---|---|---|---|
| `sequence_length` | window length (data timesteps) | `512` | int ≥ 1 | must equal `model.context_length` (asserted) |
| `stride` | window stride | `1` | int ≥ 1 | window density; trainer overrides with `train_window_stride` |
| `prediction_horizon` | target offset | `0` | int ≥ 0 | 0 ⇒ same-timestep reconstruction (no future targets) |
| `padding.enabled` | pad incomplete windows | `false` | bool | if true, incomplete windows kept |
| `sampling` | window sampling mode | `contiguous` | `contiguous` | currently only contiguous |
| `drop_incomplete_windows` | drop windows shorter than seq | `true` | bool | if true, short histories yield no windows |

---

## Phase 2 — teacher training configs

### 9. `model_v1.yaml` (raw-feature variant)

| Key | Meaning | Default | Allowed values | Effect |
|---|---|---|---|---|
| `model.name` | model identifier | `teacher_transformer_v1` | string | logged in registry |
| `model.feature_dim` | input feature width | `15` | int | projection input |
| `model.context_length` | data timesteps per window | `512` | int | **must equal `windowing_v1.yaml.sequence_length`** |
| `model.cls_token` | prepend learned CLS | `true` | bool | encoder/RoPE/mask sized `context_length + 1 = 513` |
| `model.d_model` | hidden width | `512` | int divisible by `n_heads` | parameter count, capacity |
| `model.n_layers` | transformer blocks | `8` | int ≥ 1 | capacity |
| `model.n_heads` | attention heads | `8` | int dividing `d_model` | capacity |
| `model.d_ff` | MLP hidden width | `2048` | int | capacity |
| `model.dropout` | dropout rate | `0.1` | [0,1) | regularization |
| `model.rope_theta` | RoPE base | `10000.0` | float > 1 | positional geometry |
| `model.pooling` | pooling modes to expose | `[cls, mean, attention]` | any subset | what eval can extract |

`loss.masked_modeling`:

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `mask_ratio` | fraction of valid timesteps masked | `0.15` | (0,1] | harder/easier task |
| `price_indices` | price group columns | `[0,1,2,3,4]` | indices | Huber group |
| `funding_oi_indices` | funding/OI columns | `[5,6]` | indices | MSE group |
| `calendar.<field>.index/classes/offset` | CE head spec | minute_of_day 7/1440/0, hour 8/24/0, day_of_week 9/7/0, day_of_month 10/31/1, month 11/12/1, quarter 12/4/1, year 13/16/2020, is_weekend 14/2/0 | per field | calendar reconstruction head; `offset` maps raw→0-based target (1 for 1-based fields, 2020 for year) |
| `group_weights.price/.funding_oi/.calendar` | loss weighting | `1.0` each | float ≥ 0 | total = weighted sum of groups |

### 10. `model_returns_v1.yaml` (stationary returns variant)

Same model keys as `model_v1.yaml`. Loss differs:

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `mask_mode` | masking strategy | `span` | `random` or `span` | `span` masks contiguous blocks |
| `span_len` | span length for `span` mode | `16` | int ≥ 1 | span width |
| `reconstruct_calendar` | calendar as reconstruction target | `false` | bool | when false, calendar stays input only (returns style) |

Calendar CE spec mirrors `model_v1.yaml` (still needed by the head even when
`reconstruct_calendar: false`).

### 11. `optimizer_v1.yaml`

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `optimizer.adamw.lr` | AdamW learning rate | `3.0e-4` | float > 0 | step size |
| `optimizer.adamw.weight_decay` | weight decay | `0.01` | float ≥ 0 | regularization |
| `optimizer.adamw.betas` | Adam betas | `[0.9, 0.999]` | two floats | optimizer dynamics |
| `optimizer.adamw.eps` | epsilon | `1.0e-8` | float > 0 | numerical stability |
| `optimizer.grad_clip` | global grad norm clip | `1.0` | float > 0 | prevents explosion |
| `scheduler.warmup_frac` | linear warmup fraction of total steps | `0.05` | [0,1] | LR ramp |
| `scheduler.decay` | decay schedule | `cosine` | `cosine` | LR shape after warmup |
| `scheduler.lr_floor` | LR floor (as multiplier) | `1.0e-6` | float | cosine decay never goes below `floor/base_lr` |

### 12. `trainer_v1.yaml` (raw-feature training loop)

| Key | Meaning | Default | Allowed values | Effect |
|---|---|---|---|---|
| `trainer.batch_size` | batch size | `64` | int ≥ 1 | memory + steps/epoch |
| `trainer.epochs` | number of epochs | `10` | int ≥ 1 | total training |
| `trainer.num_workers` | DataLoader workers | `2` | int ≥ 0 | data pipeline parallelism (`persistent_workers` when > 0) |
| `trainer.device` | device | `auto` | `auto`, `cuda`, `cpu` | `auto` → cuda if available else cpu |
| `trainer.seed` | run seed | `42` | int | all RNGs |
| `trainer.market` | market to train on | `futures` | `futures`, `spot` | which canonical tree is read |
| `trainer.mixed_precision` | AMP bf16 + grad scaling on CUDA | `true` | bool | speed/memory on GPU; ignored on CPU |
| `trainer.train_window_stride` | stride over frozen window set for train | `16` | int ≥ 1 | # train windows (≈ density) |
| `trainer.val_window_stride` | stride for validation | `16` | int ≥ 1 | # val windows |
| `trainer.max_train_windows` | deterministic cap | `null` | int or `null` | smoke caps to 256; `null` = no cap |
| `trainer.max_val_windows` | deterministic cap | `null` | int or `null` | smoke caps to 64 |
| `trainer.log_every` | log cadence (steps) | `50` | int ≥ 1 | console + `logs/train_teacher.log` |
| `trainer.val_every` | validation cadence (epochs) | `1` | int ≥ 1 | SOL validation frequency |
| `trainer.checkpoint_every` | checkpoint cadence (epochs) | `1` | int ≥ 1 | per-epoch `.pt` files |
| `trainer.normalizer.mode` | normalizer mode | `zscore` | `zscore`, `log`, `robust` | scaling transform (always fit on train split only) |

### 13. `trainer_returns_v1.yaml`

Same as `trainer_v1.yaml` plus:

| Key | Meaning | Default | Effect |
|---|---|---|---|
| `trainer.feature_style` | feature builder layout | `returns` | selects `feature_builder.build_features(style="returns")`; recorded in manifest and read back by eval |

---

## Interaction matrix (how configs couple)

| Config A | Config B | Constraint |
|---|---|---|
| `windowing_v1.yaml.sequence_length` | `model_v1.yaml.context_length` | must be equal (asserted at model build) |
| `market_state_schema_v1.json.feature_dimension` | `model_v1.yaml.feature_dim` | 15 = 15 (projection input) |
| `model_v1.yaml.loss.masked_modeling.price_indices` | schema feature_order | indices 0–4 = OHLCV, 5–6 = funding/OI |
| `trainer_returns_v1.yaml.feature_style` | `model_returns_v1.yaml.loss.reconstruct_calendar` | returns style ⇒ calendar input-only |
| `trainer_v1.yaml.device` | `mixed_precision` | AMP only active when device is CUDA |

## How a config becomes authoritative

- Fresh run: the three Phase 2 YAMLs are loaded from CLI args.
- Resume: the run's own `manifest.json["configs"]` **overrides the CLI**
  (`train_teacher.py`). This prevents silent config drift on resume.
- Evaluation: reads the same recorded configs from the checkpoint.
