# Research Philosophy — Pure Market Foundation Model

This document explains the *why* behind the design. It is not a tutorial; for
mechanics see `SYSTEM_OVERVIEW.md`, `DATA_FLOW.md`, and `TRAINING_GUIDE.md`.
Read this before you change anything, because most design constraints here are
intentional and load-bearing.

---

## 1. The thesis

> A market foundation model should learn a **representation** of market
> behavior from **raw exchange data alone**, using a **self-supervised
> objective**, under a **reproducibility-first engineering regime**. It is not
> a trading system. It does not forecast price. It produces embeddings that
> later, narrower systems may (or may not) build on.

The pilot is deliberately tiny — 3 symbols, 1 year, futures only, snapshot
`2026-07-30` — to validate the *pipeline* on controlled data. Scaling symbols,
years, and modalities later must require **zero architecture changes**.

## 2. The 8 core principles

1. **Raw data is immutable.** Once downloaded, Binance archives are never
   modified. Integrity is verified by SHA256 at ingestion.
2. **Canonical preserves exchange semantics.** The canonical layer changes only
   the container (CSV→Parquet), never the fields, units, or types.
3. **Alignment is causal and versioned.** Each aligned value at minute *t* uses
   only observations with `known_at ≤ t`. No future information ever enters.
   Alignment rules live in versioned YAML contracts, so old experiments stay
   reproducible against the contract they were run with.
4. **Storage is richer than any encoder.** The lake holds everything; a model
   declares a subset via the modality registry.
5. **No hand-crafted indicators.** No RSI, MACD, EMA, ATR, VWAP. If a feature
   can be computed from past data with a fixed formula, the *model* should
   learn it — the dataset must not encode an analyst's priors.
6. **Modalities are additive.** Enabling a modality adds columns; it never
   rewrites existing ones. Phase progression is a registry flip.
7. **Models never modify datasets.** `MarketDataset` exposes samples; it does
   not engineer, cut, or compute them. Training writes to `models/`, never to
   `storage/`.
8. **Every experiment is reproducible.** Same snapshot + same manifest +
   same seeds ⇒ byte-identical results. Version pins are first-class artifacts.

## 3. Why no labels and no trading

- Markets have no reliable, stationary supervision signal for "good behavior".
  Labels derived from future returns leak information and encode a strategy.
- A foundation model's job is **transferable context**, not decisions.
  Trading, RL, and prediction are treated as downstream consumers of embeddings.
- Success criteria are therefore: stable embeddings, meaningful regime
  separation, cross-symbol generalization, reusable features. **Explicitly
  not**: making money, predicting price, executing trades.

## 4. Why Masked Market Modeling (and not something else)

The Phase 2 objective is a single SSL task: **mask 15% of the timesteps, then
reconstruct the masked feature vector from surrounding context**.

- Same-timestep reconstruction (contract `prediction_horizon=0`) means targets
  are in-window; bidirectional attention is legitimate and there is no causal
  gap to guard.
- The grouped loss respects the semantics of each feature group:

  | Group | Indices | Loss | Why |
  |---|---|---|---|
  | price (OHLCV) | 0–4 | Huber (δ=1) | continuous, tail-heavy, robust to outliers |
  | funding/OI | 5–6 | MSE | continuous, near-stationary |
  | calendar | 7–14 | per-field CrossEntropy | categorical by construction |

- Unavailable modalities never contribute: `feature_mask==False` → zero weight
  (CE uses `ignore_index=-100`).
- `contrastive.py` and `temporal.py` exist only as documented placeholders that
  raise `NotImplementedError` (Phase 3+), mirroring the alignment engine's
  future-modality pattern. One objective only.

### Masking modes

- `random` (default, `model_v1.yaml`): uniformly ~15% of valid timesteps.
- `span` (`model_returns_v1.yaml`): contiguous spans of `span_len=16`, forcing
  prediction from surrounding context instead of trivially interpolating
  randomly-masked neighbors.
- **CLS seam invariant:** masks are drawn from the 512 *data* positions only;
  the CLS slot is never masked, never padding. Runtime-asserted in training
  (`key_padding_mask[:, 0].all()`) and covered by tests.

## 5. Why the architecture looks the way it does

- **Encoder-only, Pre-LN, ~25M params.** Boring, standard, debuggable. The
  pilot is about the *pipeline and representation*, not novel attention.
- **RoPE with time-aware positions.** Data token *j* gets position
  `1 + (ts_j − ts_0)/60000`, so relative distances reflect real minute gaps
  (the windowing contract tolerates small intra-window gaps). CLS gets
  position 0. RoPE is used because it handles positions that are *relative and
  shifted* cleanly (shift-equivariance is tested).
- **CLS token.** A learned aggregate slot; the padding mask hardcodes it valid.
- **Three pooling modes** (`cls`, `mean`, `attention`), all compared in the
  linear probe. **Caveat:** the `attention` pooler is *not trained* by the
  training loop — it is initialized deterministically and its output reflects a
  random query. Prefer `cls`/`mean` for research. This is documented in code
  and here so nobody misreads attention-pooled numbers.
- **Normalizer is a training-layer stage.** Statistics are fit on the `train`
  split only (train symbols, capped at `time_split.train_end`), stored in the
  checkpoint, and re-applied identically at evaluation. Fitting on val/test
  would leak distributional information — treated with the same seriousness as
  uncausal alignment.

## 6. Why the data layer is versioned to the extreme

Every downstream embedding *means* something specific to:
- the alignment contract (how funding/OI were resolved),
- the feature builder version (the 15-dim layout),
- the windowing spec (512×15), and
- the modality registry (which columns are active).

Changing any of these silently changes what embeddings mean. So all four are
**versioned artifacts pinned in `training_manifest_v1.json`**, and the manifest
plus a SHA256 ledger over every file produce a **dataset fingerprint**
(`dataset_fingerprint.json`). If the data changes, the fingerprint changes, and
old runs are recognized as not-comparable.

## 7. Leakage rules (the hard line)

1. **Time causality:** forward-fill uses only `known_at ≤ t`. Tested by
   perturbation property tests across all four Phase 1 modalities.
2. **Split causality:** `train_end = 2024-11-30`. Train windows end before it;
   test windows begin at/after it. Evaluation capping/flooring is tested.
3. **Distributional isolation:** normalizer fit and probe thresholds use train
   split data only. Baselines fit scalers/classifiers on the fit split only.
4. **Stale-value rule:** a forward-filled funding/OI value older than the
   modality's declared frequency is flagged `_stale` and marked unobserved in
   `feature_mask` — it may remain *input context*, but it is **not a
   reconstruction target**.

## 8. Reproducibility stance

- Seeds: `python`/`numpy`/`torch` all 42 (from the manifest).
- Determinism stack: `seed_everything` (incl. `cudnn.deterministic`),
  `EpochMarketSampler` (seed+epoch), persistent seeded `MaskGenerator`,
  a *fixed-seed* validation mask generator so every epoch evaluates the same
  mask pattern, and restored RNG states on resume.
- Git: the run's `manifest.json` records `git rev-parse HEAD`
  (`"unknown"` fallback). Repo is tagged `v1.0-baseline`.
- See `REPRODUCIBILITY.md` for the complete mechanics and known limits.

## 9. Non-goals (equally binding)

Trading strategies · RL / PPO/DQN · portfolio optimization · risk management ·
order execution · student models · Mixture-of-Experts · live inference ·
websocket depth collection (Phase 1) · hyperparameter search · distributed
training infra (K8s/Spark/Kafka).

## 10. How to evaluate a change against this philosophy

Ask three questions before merging:

1. Does it add information the model could have learned itself? (violates #5)
2. Could it change what an existing embedding means without bumping a version?
   (violates #6 and #8)
3. Does it make any value at time *t* depend on data known after *t*? (violates #3)
