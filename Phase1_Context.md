# Phase 1 Context — Pure Market Foundation Model

## Objective
Build a reproducible, research-grade data platform for training self-supervised market foundation models. Phase 1 is strictly focused on deterministic historical data acquisition, canonicalization, alignment, feature construction, window generation, and dataset delivery. No model training logic, feature engineering, reinforcement learning, or online systems belong in this phase.

## Core Architecture
Raw Data
→ Canonical Data
→ Versioned Alignment
→ Modality Registry
→ Versioned Feature Builder
→ Versioned Windowing
→ MarketDataset
→ Optional Training Normalizer
→ Transformer

## Core Principles
1. Raw data is immutable.
2. Canonical data preserves exchange semantics.
3. Alignment is causal and versioned.
4. Storage is richer than any encoder.
5. No handcrafted indicators.
6. Modalities are additive.
7. Models never modify datasets.
8. Every experiment is reproducible.

## Phase 1 Modalities
- OHLCV
- Funding
- Open Interest
- Calendar

Future (declared only):
- AggTrades
- Depth
- Liquidations

## Versioned Components
- Alignment Contract
- Modality Registry
- Feature Builder
- Windowing Specification
- Market State Schema
- Canonical Schema
- Dataset Snapshot
- Training Manifest

## Dataset Guarantees
- No future leakage
- Deterministic feature generation
- Explicit missing-data policies
- Version-locked experiments
- Quality reports
- Provenance on canonical data

## Dataset Output
Each sample contains:
- features
- feature_mask
- timestamps
- mask
- metadata

## Data Layer
- Download
- Convert
- Validate
- Resample
- Align
- Build features
- Window sequences
- Snapshot

## Training Layer
- Normalization
- Sampling
- DataLoader
- Benchmarking
- Experiment registry

## Out of Scope
- Technical indicators
- Feature engineering
- Reinforcement learning
- Hyperparameter search
- Distributed systems
- Live inference

## Next Step
Freeze Phase 1 and implement the Foundation Model specification separately.
