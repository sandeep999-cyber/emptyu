# Pure Market Foundation Model — Dataset Card

## Overview

Self-supervised dataset for training market foundation models on raw, unengineered crypto market data from Binance Vision historical archives. No technical indicators, no feature engineering, no human session labels.

## Snapshot

- **Date**: 2026-07-30
- **Exchange**: Binance (USD-M Futures + Spot)
- **Primary Interval**: 1-minute
- **Resampled Timeframes**: 5m, 15m, 1h, 4h, 1d
- **Compression**: Apache Parquet (Snappy)

## Modalities (Phase 1)

| Modality | Frequency | Alignment | Known At |
|---|---|---|---|
| OHLCV (open, high, low, close, volume) | 1m | identity | candle close |
| funding_rate | 8h | forward_fill | settlement_time |
| open_interest | 5m | forward_fill | create_time |
| calendar (8 fields) | 1m | derived | — |

## Feature Vector

15-dimensional, versioned (`feature_builder_version: v1`):

```
open, high, low, close, volume,
funding_rate, open_interest,
minute_of_day, hour, day_of_week, day_of_month,
month, quarter, year, is_weekend
```

## Dataset Output Contract

Each sample contains:
- `features` — `Tensor[seq_len, 15]` (float32)
- `feature_mask` — `Tensor[seq_len, 15]` (bool) — per-feature validity
- `timestamps` — `Tensor[seq_len]` (int64, epoch ms)
- `mask` — `Tensor[seq_len]` (bool) — per-timestep validity
- `metadata` — dict (symbol, start_ts, end_ts, snapshot_id, modality_config, windowing_config)

## Quality Guarantees

- **Causality**: Forward-fill alignment uses only `known_at ≤ t` observations. Zero future leakage.
- **No engineered indicators**: 0 RSI, MACD, EMA, ATR, or VWAP.
- **Deterministic**: Same manifest + snapshot = byte-identical output.
- **Version-locked**: Alignment, features, windowing, registry, and schema are all pinned via training manifest.
- **Provenance**: Every canonical Parquet carries embedded metadata (source checksum, download date, converter version, alignment version, schema version, snapshot).

## Configuration

- **Alignment contract**: `alignment_v1.yaml`
- **Modality registry**: `modalities_v1.yaml`
- **Windowing spec**: `windowing_v1.yaml` (seq_len=512, stride=1, drop_incomplete=true)
- **Market state schema**: `market_state_schema_v1.json`

## Provenance

Every canonical Parquet file carries:
```json
{
  "provenance_created_by": "parquet_converter.py",
  "provenance_source": "binance_vision_archive",
  "provenance_source_checksum": "sha256:...",
  "provenance_download_date": "2026-07-30",
  "provenance_converter_version": "v1",
  "provenance_alignment_version": "alignment_v1.yaml",
  "provenance_schema_version": "canonical_schema_v1",
  "provenance_snapshot": "2026-07-30"
}
```

## Core Principles

1. Raw data is immutable.
2. Canonical preserves exchange semantics.
3. Alignment is causal and versioned.
4. Storage is richer than any encoder.
5. No hand-crafted indicators.
6. Modalities are additive.
7. Models never modify datasets.
8. Every experiment is reproducible.

## License

Research use only. Data sourced from Binance public archives.
