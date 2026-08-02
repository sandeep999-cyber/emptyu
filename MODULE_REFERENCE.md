# Module Reference — Every Source Module

Every Python module in `src/`, documented with the same nine attributes:

**Purpose · Inputs · Outputs · Assumptions · Side effects · Failure modes · Dependencies · Configuration · Tests**

Test files are listed by name under `tests/` (root) and `tests/unit/`.

---

## Entry points

### `main.py`
- **Purpose** — Unified Phase 1 CLI (`download`, `convert`, `resample`, `align`,
  `build-lake`, `validate`, `quality-report`, `snapshot`, `report`, `benchmark`).
- **Inputs** — CLI args (`--symbols`, `--market`, `--year/--start-year/--limit-months`, `--date`, …).
- **Outputs** — Fills `storage/`; prints stage summaries.
- **Assumptions** — Configs present; network for `download`.
- **Side effects** — Writes raw/canonical/training artifacts; appends to
  `pipeline_manifest_v1.json`; exits non-zero on validation/snapshot failure.
- **Failure modes** — Network errors (download), corrupt archives (convert), checksum mismatch (validate → exit 1).
- **Dependencies** — all `src.data.*` singletons, `rich`, `src.training.benchmark`.
- **Configuration** — `configs/*` (via `src/config.py`).
- **Tests** — covered indirectly via the module-level test files.

### `src/training/train_teacher.py`
- **Purpose** — Phase 2 training CLI (smoke / full / resume); logs experiment to DuckDB.
- **Inputs** — `--model-config`, `--optimizer-config`, `--trainer-config`, `--smoke`, `--resume <run_dir>`.
- **Outputs** — Run dir under `models/foundation/teacher_v1/<run_id>/`; registry row.
- **Assumptions** — Frozen data + manifest present; on resume, run's configs are authoritative.
- **Side effects** — Seeds global RNG; writes checkpoints; inserts into `experiment_registry.duckdb`.
- **Failure modes** — Missing `latest.json` on resume (argparse error); empty train/val datasets (RuntimeError).
- **Dependencies** — `TeacherTrainer`, `CheckpointManager`, `experiment_registry`, `seeding`.
- **Configuration** — `model_v1.yaml`, `optimizer_v1.yaml`, `trainer_v1.yaml` (+ returns variants); smoke overrides.
- **Tests** — `tests/unit/test_training.py`.

---

## Core

### `src/config.py`
- **Purpose** — Central config loader; exposes global `config` singleton with typed Path properties.
- **Inputs** — files in `configs/`.
- **Outputs** — `Config` object attributes (`config.download`, `config.storage`, `config.validation`, `config.dataset`, `config.modalities`, `config.alignment`, `config.market_state_schema`, `config.windowing`, and path properties).
- **Assumptions** — All config files exist; `BASE_DIR` = repo root.
- **Side effects** — None (pure read).
- **Failure modes** — `FileNotFoundError` for missing config; any parse error propagates.
- **Dependencies** — `yaml`, `json`.
- **Configuration** — every file under `configs/`.
- **Tests** — indirectly everywhere.

### `src/logger.py`
- **Purpose** — Per-stage file loggers writing to `logs/<stage>.log`; `errors.log` for warnings+.
- **Inputs** — stage name string.
- **Outputs** — log files under `logs/`.
- **Assumptions** — `config.storage.logs_dir` exists or is creatable.
- **Side effects** — Creates `logs/`; registers module-level handlers (once per stage).
- **Failure modes** — permission issues on `logs/`.
- **Dependencies** — `logging`, `config`.
- **Configuration** — `storage.logs_dir`.
- **Tests** — none dedicated.

---

## `src/data/` — Phase 1 pipeline

### `alignment.py`
- **Purpose** — Causal alignment engine executing `alignment_v1.yaml` (forward-fill via asof, calendar derivation, stale-value flagging).
- **Inputs** — klines/funding/open_interest/calendar DataFrames; contract from config.
- **Outputs** — aligned `DataFrame` on the 1m kline timeline, with `<value>_stale` flags.
- **Assumptions** — Causal only (`direction="backward"`); future modalities declared-not-implemented.
- **Side effects** — None (pure function of inputs + contract).
- **Failure modes** — `KeyError` on missing columns; `NotImplementedError` for agg_trades/depth/liquidations.
- **Dependencies** — `pandas`, `config`.
- **Configuration** — `alignment_v1.yaml`.
- **Tests** — `test_alignment.py` (9; causality property tests across modalities).

### `binance_vision.py`
- **Purpose** — Async downloader for Binance Vision monthly archives + checksum verification.
- **Inputs** — market, symbol, dataset_type, year, month.
- **Outputs** — ZIP paths under `storage/raw/...` (or `None` on 404/checksum failure).
- **Assumptions** — Binance Vision URL scheme; monthly archives for the requested type.
- **Side effects** — Writes ZIPs + `.checksum` files; deletes mismatched ZIPs.
- **Failure modes** — returns `None` on non-200/non-404/checksum mismatch (never writes corrupt files).
- **Dependencies** — `aiohttp`, `config`.
- **Configuration** — `download.yaml` (base_url, concurrency, retries).
- **Tests** — `test_binance_vision.py` (6).

### `binance_rest.py`
- **Purpose** — REST client for `exchangeInfo`, funding history, OI history.
- **Inputs** — market, symbol, time ranges.
- **Outputs** — parsed JSON (symbol metadata lists, funding/OI series).
- **Assumptions** — Binance REST reachable.
- **Side effects** — None.
- **Failure modes** — HTTP errors raise via `raise_for_status()`.
- **Dependencies** — `httpx`, `config`.
- **Configuration** — `download.yaml` (rest URLs, timeout).
- **Tests** — none dedicated (covered in download paths).

### `calendar_builder.py`
- **Purpose** — Builds full-year 1-minute `calendar_{year}_v1.parquet` temporal matrix.
- **Inputs** — market, symbol, year.
- **Outputs** — calendar Parquet in `storage/canonical/{market}/{symbol}/metadata/`; registered in index.
- **Assumptions** — deterministic time grid (no data dependency).
- **Side effects** — Writes Parquet; registers file in DuckDB.
- **Failure modes** — file write errors.
- **Dependencies** — `pandas`, `pyarrow`, `db`.
- **Configuration** — `storage.yaml`.
- **Tests** — `test_calendar.py` (3).

### `datacard_builder.py`
- **Purpose** — Generates HF-style `DATASET.md` cards (global + per-symbol).
- **Inputs** — symbol/market/stats/quality_report or symbol list.
- **Outputs** — markdown strings; `save_datacard` writes to disk.
- **Assumptions** — none.
- **Side effects** — writes `DATASET.md` (global root + per-symbol metadata dir).
- **Failure modes** — none significant.
- **Dependencies** — none beyond stdlib.
- **Configuration** — uses config strings passed as args.
- **Tests** — none dedicated.

### `db.py`
- **Purpose** — DuckDB manager: `file_index_v1` (files/checksums) + `asset_registry` (append-only symbol metadata).
- **Inputs** — file records / asset records / query filters.
- **Outputs** — rows (`query_files`, `query_assets`, `cleanup_orphaned`).
- **Assumptions** — DB writable; schema migrations not versioned.
- **Side effects** — creates/updates `index.duckdb`; registers files/assets.
- **Failure modes** — DuckDB lock/concurrency; `register_files_batch` length validation (15 fields).
- **Dependencies** — `duckdb`, `config`.
- **Configuration** — `storage.db_path`.
- **Tests** — `test_db_duckdb.py` (6).

### `feature_builder.py`
- **Purpose** — Fuses aligned modalities into the 15-dim feature matrix (+ mask + timestamps); `raw` and `returns` styles.
- **Inputs** — aligned DataFrame; `style`.
- **Outputs** — `features [N,15] float32`, `feature_mask [N,15] bool`, `timestamps [N] int64`.
- **Assumptions** — schema `feature_order` length == `feature_dimension`; NaN → masked + zeroed.
- **Side effects** — None.
- **Failure modes** — `ValueError` on schema mismatch.
- **Dependencies** — `pandas`, `numpy`, `config`.
- **Configuration** — `market_state_schema_v1.json`.
- **Tests** — `test_feature_builder.py` (10).

### `lake.py`
- **Purpose** — Virtual Data Lake: `market_state(symbol, market, start_ts, end_ts)` aligned view with SQL pushdown + 8h lookback.
- **Inputs** — symbol/market/time range.
- **Outputs** — aligned `DataFrame` (or empty).
- **Assumptions** — canonical Parquets present; alignment contract active.
- **Side effects** — None (in-memory DuckDB).
- **Failure modes** — `RuntimeError` on corrupt Parquet; empty DataFrame on missing modality.
- **Dependencies** — `duckdb`, `pandas`, `alignment`.
- **Configuration** — `storage.canonical_dir`, `alignment_v1.yaml`.
- **Tests** — `test_lake.py` (2).

### `manifest_builder.py`
- **Purpose** — Builds `training_manifest_v1.json` (splits + version pins + file ledger) and `dataset_fingerprint.json`.
- **Inputs** — snapshot date, split symbols, file hashes (from index).
- **Outputs** — manifest dict + fingerprint dict; `save_*` writes JSON.
- **Assumptions** — file ledger filtered by snapshot symbols.
- **Side effects** — writes the two JSON files (from `main.py snapshot`).
- **Failure modes** — none significant.
- **Dependencies** — `db`, `config`.
- **Configuration** — `dataset.yaml` (splits/train_end), `storage.training_dir`.
- **Tests** — `test_manifest.py` (4).

### `market_dataset.py`
- **Purpose** — PyTorch `Dataset` exposing window tensors (never engineers).
- **Inputs** — list of window dicts from `WindowingEngine`.
- **Outputs** — sample dict (`features`, `feature_mask`, `timestamps`, `mask`, `metadata`).
- **Assumptions** — windows well-formed.
- **Side effects** — None.
- **Failure modes** — None.
- **Dependencies** — `torch`, `numpy`.
- **Configuration** — none (windows pre-built).
- **Tests** — `test_market_dataset.py` (1); heavily exercised elsewhere.

### `metadata.py`
- **Purpose** — Versioned metadata JSON (`dataset_version.json`, `statistics_v1.json`).
- **Inputs** — version strings / DataFrame.
- **Outputs** — dicts; `save_json` writes files.
- **Assumptions** — `created` passed explicitly for reproducibility.
- **Side effects** — writes metadata JSON.
- **Failure modes** — none.
- **Dependencies** — `pandas`, `numpy`, `config`.
- **Configuration** — `storage.canonical_dir`.
- **Tests** — `test_metadata.py` (4).

### `modality_registry.py`
- **Purpose** — Loads/validates `modalities_v1.yaml`; query enabled modalities.
- **Inputs** — config dict (default: `config.modalities`).
- **Outputs** — `is_enabled(name)`, `get_active_modalities()`.
- **Assumptions** — known modality set fixed.
- **Side effects** — None.
- **Failure modes** — `ValueError` unknown modality / missing `enabled`; `TypeError` non-bool.
- **Dependencies** — `config`.
- **Configuration** — `modalities_v1.yaml`.
- **Tests** — `test_modality_registry.py` (7).

### `parquet_converter.py`
- **Purpose** — CSV/ZIP → canonical Snappy Parquet with robust header detection, timestamp normalization, embedded provenance, index registration.
- **Inputs** — zip path, market/symbol/dataset_type/interval/year/month, provenance params.
- **Outputs** — Parquet path under `storage/canonical/`.
- **Assumptions** — Binance CSV schemas (headerless or named) per `BINANCE_CSV_SCHEMAS`.
- **Side effects** — writes Parquet; registers in `file_index_v1`.
- **Failure modes** — `FileNotFoundError` on missing zip; warnings on dropped unparseable timestamps.
- **Dependencies** — `pandas`, `pyarrow`, `db`.
- **Configuration** — `storage.parquet_compression`.
- **Tests** — `test_parquet_converter.py` (3).

### `pipeline_manifest.py`
- **Purpose** — Lineage ledger `pipeline_manifest_v1.json` recording each stage's inputs/outputs/checksums/duration.
- **Inputs** — stage name, inputs, outputs, checksums, duration, metadata, timestamp.
- **Outputs** — `pipeline_manifest_v1.json`.
- **Assumptions** — stage keys mutable (auto-created).
- **Side effects** — writes JSON from CLI stages.
- **Failure modes** — none.
- **Dependencies** — `config`.
- **Configuration** — `storage.training_dir`.
- **Tests** — none dedicated.

### `quality_report.py`
- **Purpose** — Generates per-symbol `quality_report.json` (§9a of Phase 1 plan).
- **Inputs** — symbol, market, aligned DataFrame.
- **Outputs** — report dict with coverage/FF%/gaps/score; `save_report` writes JSON.
- **Assumptions** — aligned frame + index queryable.
- **Side effects** — writes metadata JSON.
- **Failure modes** — none.
- **Dependencies** — `pandas`, `numpy`, `db`, `config`.
- **Configuration** — `validation.max_timestamp_gap_seconds`.
- **Tests** — `test_quality_report.py` (2).

### `reports.py`
- **Purpose** — Storage summary markdown (`storage_summary.md`).
- **Inputs** — `file_index_v1` rows.
- **Outputs** — markdown report.
- **Side effects** — writes report file.
- **Failure modes** — none.
- **Dependencies** — `db`.
- **Configuration** — `storage.training_dir`.
- **Tests** — none dedicated.

### `resampler.py`
- **Purpose** — Aggregates 1m klines → 5m/15m/1h/4h/1d via DuckDB SQL, flags incomplete candles, embeds provenance.
- **Inputs** — 1m Parquet path, market/symbol/target interval/year/month.
- **Outputs** — resampled Parquet; index entry (status `RESAMPLED[_INCOMPLETE]`).
- **Assumptions** — DuckDB FLOOR-bucket aggregation matches exchange boundaries.
- **Side effects** — writes Parquet; registers file.
- **Failure modes** — `ValueError` for unsupported interval.
- **Dependencies** — `duckdb`, `pyarrow`, `db`.
- **Configuration** — `storage.parquet_compression`.
- **Tests** — `test_resampler.py` (2).

### `snapshot_manager.py`
- **Purpose** — Immutable snapshot creator (`storage/training/snapshots/<date>/` with manifest/checksums/stats/content_hash).
- **Inputs** — snapshot date, manifest/checksums/stats.
- **Outputs** — snapshot dir.
- **Assumptions** — date uniqueness (immutability).
- **Side effects** — writes snapshot files.
- **Failure modes** — `FileExistsError` on duplicate date.
- **Dependencies** — `db`, `config`.
- **Configuration** — `storage.training_dir`.
- **Tests** — `test_snapshot.py` (7).

### `validator.py`
- **Purpose** — Integrity auditor: SHA256 verification + declarative per-modality rules (`validation.yaml`).
- **Inputs** — files/DataFrames.
- **Outputs** — `verify_sha256` bool; `validate_<modality>` → `(ok, errors)`.
- **Assumptions** — rules declared in YAML.
- **Side effects** — None.
- **Failure modes** — none; errors returned as lists.
- **Dependencies** — `pandas`, `config`.
- **Configuration** — `validation.yaml`.
- **Tests** — `test_validator.py` (17).

### `windowing.py`
- **Purpose** — Cuts feature matrices into fixed-length windows with gap/order validation; builds per-position mask.
- **Inputs** — features/mask/timestamps, optional windowing dict, metadata.
- **Outputs** — list of window dicts.
- **Assumptions** — strictly increasing unique timestamps; `max_gap_ms` tolerance.
- **Side effects** — None.
- **Failure modes** — `ValueError` on duplicate/out-of-order timestamps.
- **Dependencies** — `numpy`, `config`.
- **Configuration** — `windowing_v1.yaml` (trainer may pass an override dict).
- **Tests** — `test_windowing.py` (4).

---

## `src/models/teacher/` — Phase 2 model

### `projection.py`
- **Purpose** — `FeatureProjection` (15→d_model linear) + `ReconstructionHead` (price/funding-oi/calendar CE heads).
- **Inputs** — `[B, T, 15]`; latent `[B, T, d_model]`.
- **Outputs** — projected `[B, T, d_model]`; reconstruction dict.
- **Assumptions** — calendar spec present (else head is `None`).
- **Side effects** — None.
- **Failure modes** — `RuntimeError` if `reconstruct` called with no head.
- **Dependencies** — `torch`.
- **Configuration** — `model_v1.yaml` (loss calendar spec).
- **Tests** — `tests/unit/test_projection.py` (9).

### `positional_encoding.py`
- **Purpose** — RoPE with time-aware positions (CLS=0, data=1+minute_offset), cached cos/sin.
- **Inputs** — q/k `[..., T, D]`, positions.
- **Outputs** — rotated q/k.
- **Assumptions** — positions fit int; cache rebuilds on demand.
- **Side effects** — None.
- **Failure modes** — none.
- **Dependencies** — `torch`.
- **Configuration** — `model.rope_theta`.
- **Tests** — `tests/unit/test_transformer.py` (shift-equivariance).

### `transformer.py`
- **Purpose** — Pre-LN transformer block: LN → QKV(+RoPE) → SDPA (key_padding_mask) → residual → LN → MLP → residual.
- **Inputs** — `[B, T, D]`, key_padding_mask `[B, T]` (True=valid), positions.
- **Outputs** — `[B, T, D]`.
- **Assumptions** — `d_model % n_heads == 0`.
- **Side effects** — None.
- **Failure modes** — assert failure on head mismatch.
- **Dependencies** — `torch`, `positional_encoding`.
- **Configuration** — `model_v1.yaml` (dropout, dims).
- **Tests** — `tests/unit/test_transformer.py` (10).

### `encoder.py`
- **Purpose** — `TeacherEncoder`: project → prepend CLS → RoPE → N blocks → final LN; `reconstruct()` via head.
- **Inputs** — `[B, T_data, 15]`, timestamps, data mask.
- **Outputs** — `latent [B, T_data+1, D]`, key_padding_mask, positions, `T_data`.
- **Assumptions** — CLS always valid; `context_length == windowing.sequence_length`.
- **Side effects** — None.
- **Failure modes** — `RuntimeError` if reconstruct without calendar spec.
- **Dependencies** — all teacher modules.
- **Configuration** — `model_v1.yaml`.
- **Tests** — `tests/unit/test_transformer.py`, `tests/unit/test_training.py`.

### `embeddings.py`
- **Purpose** — Pooling (`cls`, mask-aware `mean`, `attention`) + `extract_embeddings` (→ npz-ready dict) + `EmbeddingExtractor`-style API.
- **Inputs** — model, dataloader, pooling, device.
- **Outputs** — `{embedding [N, D], symbols, window_start_ms, window_end_ms}`.
- **Assumptions** — `attention` pooler is untrained (deterministic init); prefer cls/mean.
- **Side effects** — None (CPU tensors returned).
- **Failure modes** — `ValueError` on unknown pooling.
- **Dependencies** — `torch`, `numpy`.
- **Configuration** — pooling choice (CLI).
- **Tests** — `tests/unit/test_embeddings.py` (13).

---

## `src/training/` — Phase 2 training infrastructure

### `benchmark.py`
- **Purpose** — DataLoader throughput/RAM/CPU benchmark.
- **Inputs** — dataset, batch_size, num_batches.
- **Outputs** — metrics dict; `__main__` CLI (`--snapshot`, `--symbols`, `--batch-size`, `--num-batches`).
- **Assumptions** — lake/feature/windowing data available.
- **Side effects** — none.
- **Failure modes** — none.
- **Dependencies** — `psutil`, `torch`, lake/feature/windowing.
- **Configuration** — none (CLI).
- **Tests** — none dedicated (covered by `test_coverage.py`).

### `checkpoint.py`
- **Purpose** — Versioned checkpoint save/load: per-epoch `.pt`, `best.pt`, `latest.json`, `manifest.json` (configs verbatim, git commit, checkpoint hashes).
- **Inputs** — model/optimizer/scheduler/normalizer/mask states + losses.
- **Outputs** — run-dir files; `load_latest_checkpoint` restores state tuple.
- **Assumptions** — latest.json pointer (no Windows symlinks).
- **Side effects** — writes to `models/foundation/teacher_v1/<run_id>/`.
- **Failure modes** — `FileNotFoundError` on missing latest.json; corrupt checkpoint raises on load.
- **Dependencies** — `torch`, `hashlib`, `subprocess`.
- **Configuration** — configs recorded verbatim.
- **Tests** — `tests/unit/test_checkpoint.py` (4). See `CHECKPOINT_FORMAT.md`.

### `dataloader.py`
- **Purpose** — `create_dataloader` with seeded epoch shuffling, worker seeding, `drop_last`, pin_memory/persistent_workers.
- **Inputs** — dataset, batch size, workers, seed, sampler.
- **Outputs** — `DataLoader`.
- **Assumptions** — reproducibility via `EpochMarketSampler` + worker init.
- **Side effects** — none.
- **Failure modes** — none.
- **Dependencies** — `torch`, `sampler`.
- **Configuration** — worker/pin/persistent flags.
- **Tests** — `tests/unit/test_sampler.py`, `tests/unit/test_training.py`.

### `experiment_registry.py`
- **Purpose** — DuckDB experiment registry (`experiment_registry.duckdb`): log/query run metadata.
- **Inputs** — run metadata fields.
- **Outputs** — row in `experiment_registry` table; `query_experiments` list of dicts.
- **Assumptions** — `experiment_id` primary key (duplicate insert raises).
- **Side effects** — DB writes.
- **Failure modes** — duplicate id insert error.
- **Dependencies** — `duckdb`, `config`.
- **Configuration** — `storage.experiment_db_path`.
- **Tests** — `test_experiment_registry.py` (3).

### `normalizer.py`
- **Purpose** — `FeatureNormalizer` (zscore/log/robust), fit on train split only; state dict save/load.
- **Inputs** — train features + mask + manifest splits.
- **Outputs** — transform(x); state dict.
- **Assumptions** — never fit on val/test (raises without `train` symbols).
- **Side effects** — none.
- **Failure modes** — `ValueError` non-train fit; `RuntimeError` transform-before-fit.
- **Dependencies** — `torch`, `numpy`.
- **Configuration** — `trainer.normalizer.mode`.
- **Tests** — `test_normalizer.py` (9).

### `optimizer.py`
- **Purpose** — AdamW factory from `optimizer_v1.yaml`.
- **Inputs** — model, opt_cfg.
- **Outputs** — `optim.AdamW`.
- **Assumptions** — config shape `optimizer.adamw.*`.
- **Side effects** — none.
- **Failure modes** — KeyError on malformed config.
- **Dependencies** — `torch`.
- **Configuration** — `optimizer_v1.yaml`.
- **Tests** — via `tests/unit/test_training.py`.

### `sampler.py`
- **Purpose** — `EpochMarketSampler`: deterministic per-epoch shuffling (seed+epoch).
- **Inputs** — dataset length, shuffle, seed.
- **Outputs** — index iterators.
- **Assumptions** — epoch set before each epoch.
- **Side effects** — none.
- **Failure modes** — none.
- **Dependencies** — `torch`, `numpy`.
- **Configuration** — seed.
- **Tests** — `tests/unit/test_sampler.py` (5).

### `scheduler.py`
- **Purpose** — Linear warmup (5%) → cosine decay to floor.
- **Inputs** — optimizer, opt_cfg, total_steps.
- **Outputs** — `LambdaLR`.
- **Assumptions** — step-level; restored exactly on resume.
- **Side effects** — none.
- **Failure modes** — division by zero guarded with `max(1, ...)`.
- **Dependencies** — `torch`.
- **Configuration** — `optimizer_v1.yaml` (warmup_frac, decay, lr_floor).
- **Tests** — via `tests/unit/test_checkpoint.py` (restore).

### `seeding.py`
- **Purpose** — `seed_everything`: seeds python/numpy/torch (+CUDA), deterministic cuDNN.
- **Inputs** — seed int.
- **Outputs** — seeded global state.
- **Assumptions** — called before run setup.
- **Side effects** — mutates global RNG + cuDNN flags.
- **Failure modes** — none.
- **Dependencies** — `random`, `numpy`, `torch`.
- **Configuration** — seed from trainer config.
- **Tests** — via `tests/unit/test_training.py` (determinism).

### `trainer.py`
- **Purpose** — `TeacherTrainer`: loads manifest splits → builds windows → fits normalizer (train-only) → training loop (mask/forward/loss/backward/clip/step/scheduler) → per-epoch validation (fixed mask) → checkpoint save → best-val tracking.
- **Inputs** — model/opt/trainer configs, run_dir, resume_dir.
- **Outputs** — checkpoints, logs; metrics per epoch.
- **Assumptions** — frozen manifest; data present; `train_window_stride` built directly at stride (RAM-efficient); `max_windows` seeded subset.
- **Side effects** — writes checkpoints/history; may write experiment registry from CLI.
- **Failure modes** — `RuntimeError` if train/val datasets empty; resume failures warn + start fresh.
- **Dependencies** — lake, feature_builder, windowing, MarketDataset, dataloader, sampler, normalizer, optimizer, scheduler, checkpoint, masked_modeling, TeacherEncoder, psutil.
- **Configuration** — `model_v1.yaml`, `optimizer_v1.yaml`, `trainer_v1.yaml`.
- **Tests** — `tests/unit/test_training.py` (10), `tests/unit/test_split_integrity.py` (5).

### `losses/masked_modeling.py`
- **Purpose** — `MaskGenerator` (random or span, seeded, data-positions only) + `MaskedMarketModelingLoss` (Huber/MSE/per-field CE with `ignore_index`).
- **Inputs** — mask tensor / reconstruction dict + normalized/raw features + feature_mask + masked positions.
- **Outputs** — mask bool tensor / `{price, funding_oi, calendar, total}` losses.
- **Assumptions** — CLS seam invariant (masks never touch CLS); `feature_mask==False` contributes 0.
- **Side effects** — none.
- **Failure modes** — none.
- **Dependencies** — `torch`.
- **Configuration** — `model_v1.yaml` / `model_returns_v1.yaml` (loss section).
- **Tests** — `tests/unit/test_masking.py` (11), `tests/unit/test_losses.py` (5).

### `losses/contrastive.py` and `losses/temporal.py`
- **Purpose** — Documented placeholders for Phase 3+ objectives.
- **Inputs** — none (import raises).
- **Outputs** — `NotImplementedError` on import.
- **Assumptions** — intentionally unimplemented.
- **Side effects** — none.
- **Failure modes** — `NotImplementedError` (by design).
- **Dependencies** — none.
- **Configuration** — none.
- **Tests** — import-raise covered in `tests/unit/test_losses.py`.

---

## `src/evaluation/embedding/` — Phase 2 evaluation

### `_common.py`
- **Purpose** — Shared eval utilities: load model+normalizer from a run dir, rebuild split datasets through the frozen pipeline (time-split aware), extract normalized embeddings.
- **Inputs** — run_dir, device, split, pooling, trainer_cfg.
- **Outputs** — model, normalizer, configs; `MarketDataset`; embedding dicts.
- **Assumptions** — checkpoint manifest present; split boundaries match trainer (`train` capped / `test` floored at `train_end`).
- **Side effects** — none.
- **Failure modes** — `FileNotFoundError` missing manifest/checkpoint.
- **Dependencies** — teacher modules, normalizer, dataloader, trainer helpers.
- **Configuration** — from checkpoint manifest.
- **Tests** — via all eval module tests.

### `clustering.py`
- **Purpose** — KMeans on pooled embeddings; silhouette + AMI vs norm-derived regime labels; per-symbol cluster distribution (how unseen SOL populates clusters).
- **Inputs** — `--checkpoint`, `--split`, `--pooling`, `--clusters`, `--max-windows`.
- **Outputs** — `evaluation/embedding/clustering_<split>_<pooling>.json`.
- **Assumptions** — ≥ clusters samples.
- **Side effects** — writes JSON.
- **Failure modes** — returns `{"error": ...}` for too few samples.
- **Dependencies** — sklearn, numpy, `_common`.
- **Configuration** — CLI.
- **Tests** — `tests/unit/test_embeddings.py` (integration), `test_coverage.py`.

### `retrieval.py`
- **Purpose** — Cosine kNN retrieval; same-symbol hit rate / cross-symbol neighbor fraction.
- **Inputs** — `--checkpoint`, `--split`, `--pooling`, `--k`.
- **Outputs** — `retrieval_<split>_<pooling>.json`.
- **Side effects** — writes JSON.
- **Failure modes** — `{"error": ...}` if too few samples.
- **Dependencies** — sklearn, numpy, `_common`.
- **Configuration** — CLI.
- **Tests** — via `tests/unit/test_embeddings.py`.

### `linear_probe.py`
- **Purpose** — Logistic regression probes on frozen embeddings (volatility/range_expansion/liquidity); thresholds from train split; reports cross-symbol (SOL held-out) + in-sample (BTC) vs majority baseline; pooling comparison.
- **Inputs** — `--checkpoint`, `--pooling`, `--max-windows`.
- **Outputs** — `linear_probe_<pooling>.json`.
- **Assumptions** — feature-style-aware window stats; train split for thresholds only.
- **Side effects** — writes JSON.
- **Failure modes** — none.
- **Dependencies** — sklearn, numpy, `_common`.
- **Configuration** — CLI.
- **Tests** — `tests/unit/test_linear_probe.py` (7).

### `temporal_consistency.py`
- **Purpose** — Cosine similarity of temporally adjacent same-symbol windows vs random pairs; separation AUC.
- **Inputs** — `--checkpoint`, `--split`, `--pooling`.
- **Outputs** — `temporal_<split>_<pooling>.json`.
- **Side effects** — writes JSON.
- **Failure modes** — `{"error": ...}` if < 100 samples.
- **Dependencies** — sklearn, numpy, `_common`.
- **Configuration** — CLI.
- **Tests** — via `tests/unit/test_embeddings.py`.

### `visualization.py`
- **Purpose** — PCA/t-SNE scatter (≤ 5000 pts, seeded) + training loss curves from manifest → `evaluation/embedding/figures/`.
- **Inputs** — `--checkpoint`, `--method`, `--pooling`.
- **Outputs** — PNG figures.
- **Side effects** — writes figures.
- **Failure modes** — skips empty splits.
- **Dependencies** — matplotlib (Agg), sklearn, numpy, `_common`.
- **Configuration** — CLI.
- **Tests** — none dedicated.

---

## `src/evaluation/baselines/` — Phase A baseline harness

### `tasks.py`
- **Purpose** — Task/label definitions (`future_return`, `volatility`, `range_expansion`, `liquidity`), window stats, handcrafted vector, threshold binarization.
- **Inputs** — window features / close series.
- **Outputs** — labels, stats, feature vectors.
- **Assumptions** — feature-style-aware (`raw` vs `returns`) stats.
- **Side effects** — none.
- **Failure modes** — none.
- **Dependencies** — `numpy`.
- **Configuration** — none (constants).
- **Tests** — `test_baselines_eval.py` (13).

### `models.py`
- **Purpose** — Baseline predictors: `MajorityBaseline`, `LogisticBaseline` (scaler fit on fit split), `RandomProjectionBaseline` (seeded control).
- **Inputs** — X/y.
- **Outputs** — predictions / probabilities.
- **Assumptions** — scalers/classifiers fit on fit split only.
- **Side effects** — none.
- **Failure modes** — caught as `ValueError` in runner.
- **Dependencies** — sklearn, numpy.
- **Configuration** — constants.
- **Tests** — `test_baselines_eval.py`.

### `runner.py`
- **Purpose** — Baseline evaluation harness: builds causally-separated labeled windows, fits baselines, reports bacc/acc/auc vs majority; optional frozen-embedding comparison.
- **Inputs** — CLI (fit/eval symbols + dates, seq-len, stride, horizon, max-windows, seed, optional `--checkpoint`, `--pooling`, `--feature-style`).
- **Outputs** — `evaluation/baselines/baseline_eval_<timestamp>.json`.
- **Assumptions** — causal separation (fit windows end ≥ `horizon_min` before eval start); thresholds/scalers/classifiers fit-split-only.
- **Side effects** — writes JSON.
- **Failure modes** — empty fit/eval → early return.
- **Dependencies** — lake, feature_builder, windowing, MarketDataset, tasks, models, optional `_common`.
- **Configuration** — CLI.
- **Tests** — `test_baselines_eval.py` (13).

---

## `tests/` — coverage map

| Suite | Files | Purpose |
|---|---|---|
| Phase 1 | `test_alignment, test_binance_vision, test_calendar, test_coverage, test_db_duckdb, test_experiment_registry, test_feature_builder, test_lake, test_manifest, test_market_dataset, test_metadata, test_modality_registry, test_normalizer, test_parquet_converter, test_quality_report, test_resampler, test_snapshot, test_validator, test_windowing` | causality, resampling, index/registry, provenance, feature schema, normalizer train-only, validator rules, snapshot immutability, fingerprint determinism |
| Phase 2 | `tests/unit/test_checkpoint, test_embeddings, test_linear_probe, test_losses, test_masking, test_projection, test_sampler, test_split_integrity, test_training, test_transformer` | shapes, RoPE, CLS invariants, mask correctness, loss groups, checkpoint roundtrip + resume, determinism, split time bounds, linear probe beats majority |
| Baselines | `test_baselines_eval.py` | harness tasks/models/evaluation |

Run: `python -m pytest tests/ -v` (203 tests, verified passing).
