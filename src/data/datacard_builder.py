"""Hugging Face-style DATASET.md card generator."""

from pathlib import Path
from typing import Any, Dict, List, Optional


class DataCardBuilder:
    """Generates HF-style DATASET.md cards for global and per-symbol dataset documentation."""

    def build_symbol_datacard(
        self, symbol: str, market: str = "futures",
        stats: Optional[Dict] = None, quality_report: Optional[Dict] = None,
    ) -> str:
        content = f"""# Data Card: {symbol} ({market.upper()})

## Summary
Raw, unengineered market dataset for symbol `{symbol}` acquired from Binance Vision historical archives.

## Dataset Specifications
- **Market Type**: {market}
- **Primary Interval**: 1-minute
- **Resampled Timeframes**: 5m, 15m, 1h, 4h, 1d
- **Compression**: Apache Parquet (Snappy)

## Modalities
- **OHLCV**: Open, High, Low, Close, Volume
- **Funding Rate**: 8-hour settled funding rate (causally forward-filled)
- **Open Interest**: 5-minute open interest (causally forward-filled)
- **Calendar**: Temporal vector (`minute_of_day`, `hour`, `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `is_weekend`)

## Quality & Integrity
- **Causality Guarantee**: Forward-fill alignment based strictly on `known_at` event timestamps. Zero future leakage.
- **Engineered Indicators**: 0 (No RSI, MACD, EMA, ATR, or VWAP).
"""
        if quality_report:
            content += f"""
## Quality Report
- **Quality Score**: {quality_report.get('quality_score', 'N/A')}
- **Total Records**: {quality_report.get('total_records', 0):,}
- **Duplicate Rows**: {quality_report.get('duplicate_rows', 0)}
- **Gap Count**: {quality_report.get('gap_count', 0)}
- **Alignment Coverage**:
"""
            for mod, cov in quality_report.get("alignment_coverage", {}).items():
                content += f"  - {mod}: {cov*100:.1f}%\n"

        if stats:
            content += "\n## Feature Statistics (Train Split)\n```json\n"
            content += str(stats)
            content += "\n```\n"
        return content

    def build_global_datacard(
        self,
        symbols: List[str],
        market: str = "futures",
        snapshot_date: str = "2026-07-30",
        modality_config: str = "modalities_v1.yaml",
        alignment_config: str = "alignment_v1.yaml",
        windowing_config: str = "windowing_v1.yaml",
        feature_dim: int = 15,
        seq_len: int = 512,
    ) -> str:
        sym_list = ", ".join(f"`{s}`" for s in symbols)
        content = f"""# Pure Market Foundation Model — Dataset Card

## Overview
Self-supervised dataset for training market foundation models on raw, unengineered crypto market data from Binance Vision historical archives.

## Snapshot
- **Date**: {snapshot_date}
- **Symbols**: {sym_list}
- **Market**: {market}

## Architecture
1. Raw Data → 2. Canonical Parquets → 3. Causal Alignment → 4. Feature Builder → 5. Windowing → 6. MarketDataset

## Configuration
- **Alignment**: {alignment_config}
- **Modality Registry**: {modality_config}
- **Windowing**: {windowing_config}
- **Feature Dimension**: {feature_dim}
- **Sequence Length**: {seq_len}

## Modalities (Phase 1)
| Modality | Frequency | Alignment | Status |
|---|---|---|---|
| klines (OHLCV) | 1m | identity | Active |
| funding_rate | 8h | forward_fill | Active |
| open_interest | 5m | forward_fill | Active |
| calendar | 1m | derived | Active |

## Core Principles
1. Raw data is immutable.
2. Canonical preserves exchange semantics.
3. Alignment is causal and versioned.
4. Storage is richer than any encoder.
5. No hand-crafted indicators.
6. Modalities are additive.
7. Models never modify datasets.
8. Every experiment is reproducible.

## Provenance
Every canonical Parquet carries embedded metadata: creator, source checksum, download_date, converter_version, alignment_version, schema_version, snapshot.

## License
Research use only. Data sourced from Binance public archives.
"""
        return content

    def save_datacard(self, content: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)


datacard_builder = DataCardBuilder()
