# ARCHITECTURE.md — Core Architecture Principles & 4 Contracts

## Core Principles

1. **Raw data is immutable.** Once downloaded, raw Binance archives are never modified. Checksums are verified on ingestion.
2. **Canonical preserves exchange semantics.** Canonical Parquets maintain the same schema as the source, with only format conversion (CSV→Parquet). All fields, units, and types match the exchange.
3. **Alignment is causal and versioned.** Every modality is aligned to the shared 1-minute timeline using only past observations (`known_at ≤ t`). No future information leaks into any aligned value. Alignment policies are declared in versioned YAML contracts (`alignment_v1.yaml`).
4. **Storage is richer than any encoder.** The data lake stores all raw, canonical, and aligned modalities — more than any single model consumes. Models select subsets via the Modality Registry.
5. **No hand-crafted indicators.** No RSI, MACD, EMA, ATR, VWAP, or any technical indicator is generated. Models receive raw features only.
6. **Modalities are additive.** Enabling a new modality adds columns without modifying existing ones. Phase progression is a registry change, not a pipeline change.
7. **Models never modify datasets.** The `MarketDataset` exposes samples; it does not compute, engineer, or cut them. Training code writes to `models/`, never to `storage/`.
8. **Every experiment is reproducible.** Snapshots, manifests, and fingerprints lock every artifact version. Re-running the same manifest against the same snapshot produces byte-identical results.

---

## Four System Contracts

### 1. Raw Contract

Raw data in `storage/raw/` is downloaded from Binance Vision and verified via SHA256 checksums. Files are organized as:

```
storage/raw/{market}/{symbol}/{dataset_type}/[1m/]{symbol}-{type}-{YYYY-MM}.zip
```

Raw files are never modified after download.

### 2. Canonical Contract

Canonical data in `storage/canonical/` is converted from raw CSV/ZIP to Snappy Parquet with embedded provenance metadata. Every Parquet file carries:

- `provenance_created_by` — converter script name
- `provenance_source` — data origin (e.g., `binance_vision_archive`)
- `provenance_source_checksum` — SHA256 of the raw source
- `provenance_download_date` — explicit, deterministic date parameter
- `provenance_converter_version` — converter version
- `provenance_alignment_version` — alignment contract version
- `provenance_schema_version` — canonical schema identifier
- `provenance_snapshot` — snapshot date

### 3. Alignment Contract

Defined in `configs/alignment_v1.yaml`. The alignment engine:

- **Is standalone and modality-agnostic** — a single engine dispatches to per-modality handlers based on the contract
- **Is policy-driven** — alignment type, `known_at` field, and missing-data policy are read from the YAML, not hardcoded
- **Is causal** — `merge_asof(direction="backward")` ensures only past observations are used
- **Is versioned** — `alignment_v1.yaml` is pinned in the training manifest; old experiments stay reproducible against their contract version
- **Raises `NotImplementedError`** for future modalities (agg_trades, depth, liquidations) until Phase 2/3

### 4. Model Contract

The `MarketDataset` exposes a strict output contract:

```python
sample = dataset[i]
# {
#     "features": Tensor[seq_len, feature_dim],      # float32
#     "feature_mask": Tensor[seq_len, feature_dim],  # bool — per-feature validity
#     "timestamps": Tensor[seq_len],                 # int64 — epoch ms
#     "mask": Tensor[seq_len],                       # bool — per-timestep validity
#     "metadata": dict                               # symbol, start_ts, end_ts, snapshot_id, ...
# }
```

- `feature_mask` is separate from `mask`: a timestep can be present while individual sparse modalities within it are unavailable.
- The `Normalizer` is a training-layer stage, fit only on the `train` split, never on validation/test data.
- Models never modify datasets.

---

## System Flow

```
Raw Data (immutable)
    ↓
Canonical Parquets (Snappy, provenance-embedded)
    ↓
Alignment Engine (causal, contract-driven, versioned)
    ↓
Modality Registry (which aligned modalities are active)
    ↓
Feature Builder (versioned — fuses active modalities → 15-dim vector)
    ↓
Windowing Engine (versioned — fixed-length sequences)
    ↓
MarketDataset (PyTorch interface — exposes, never engineers)
    ↓
Normalizer (optional — train-split-only statistics)
    ↓
Transformer (Phase 2+)
```

## Phase 2 — Teacher Foundation Model

The Phase 2 teacher encoder consumes the Model Contract above **as a read-only reader**. Key points:

- **No dataset modifications.** The teacher trains via `MarketDataset` without touching `src/data/`, `storage/`, or the frozen manifest.
- **No new alignment contracts.** The feature set (15 dimensions), windowing (seq_len 512, stride 1), and modality registry are exactly as frozen in the Phase 1 snapshot.
- **Derived artifacts are written to `models/` and `evaluation/`**, never to `storage/`. Checkpoints go to `models/foundation/teacher_v1/`; evaluation reports and extracted embeddings go to `evaluation/embedding/`.
- **All 8 core principles remain unchanged.** Reproducibility, causality, immutability, and read-only model contract are preserved.
- **Masked Market Modeling** as the SSL objective — no labels, no prediction head on the latent path, no RL or trading signals.
```
