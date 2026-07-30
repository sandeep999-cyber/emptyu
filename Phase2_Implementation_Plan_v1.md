# Phase 2 — Teacher Foundation Model · Implementation Plan v1

## 1. Objective

Build the first Market Foundation Model: an encoder that learns compressed representations of market behavior from the frozen Phase 1 dataset, **without labels, indicators, or trading rules**.

```
MarketDataset → Normalizer → Projection → Positional Encoding (RoPE)
→ Transformer Encoder → Latent Representation → SSL Objective → Loss
```

Output: **embeddings**. Not buy/sell. No prediction head on the latent path, no RL, no trading, no strategy generation.

### Success criteria

- Learn stable embeddings · learn market context · separate different market regimes · generalize across symbols · produce reusable embeddings
- Explicitly **not**: make money · predict price · execute trades

### Pilot framing

Phase 2 is a **pilot** on the frozen 3-symbol / 2024 / snapshot `2026-07-30`. Goal: validate the learning pipeline on controlled data. If embeddings show meaningful structure and transfer, later phases scale symbols/years/modalities **without changing the Phase 2 architecture**.

## 2. Locked decisions

| # | Decision | Resolution |
|---|---|---|
| D1 | Hardware strategy | **Design for GPU.** `model_v1.yaml` is the canonical spec; this CPU-only box runs a `--smoke` validation only. Full pilot run = one documented command on a CUDA host |
| D2 | Market scope | **Futures only** (full 15-feature coverage incl. funding/OI; spot would carry permanently masked funding/OI) |
| D3 | Eval dependencies | Add `scikit-learn>=1.3.0`, `matplotlib>=3.8.0` to requirements.txt |
| D4 | Eval split framing | **SOL = held-out cross-symbol test** (manifest validation split); BTC = in-sample memorization baseline. Frozen manifest untouched |
| D5 | Train window density | `train_window_stride: 16` (~65K windows ≈ 33M tokens/epoch), `val_window_stride: 16` |

## 3. Phase 1 contracts consumed (read-only)

- **Model contract** per sample: `features [512,15] float32` · `feature_mask [512,15] bool` · `timestamps [512] int64 (epoch ms)` · `mask [512] bool` · `metadata dict`
- **Feature order** (`market_state_schema_v1.json`): idx 0–4 `open, high, low, close, volume` · idx 5 `funding_rate` · idx 6 `open_interest` · idx 7–14 calendar (`minute_of_day, hour, day_of_week, day_of_month, month, quarter, year, is_weekend`; raw integers; day_of_month/month/quarter are 1-based)
- **Frozen manifest** `storage/training/training_manifest_v1.json`: snapshot `2026-07-30` · train = BTCUSDT, ETHUSDT · validation = SOLUSDT · test = BTCUSDT · seeds 42/42/42 · dataset fingerprint in `storage/training/dataset_fingerprint.json`
- **Data path**: `lake.market_state(sym, market="futures")` → `feature_builder.build_features` → `windowing_engine.create_windows` (frozen `windowing_v1.yaml`: seq_len 512, stride 1) → `MarketDataset`
- **Existing infrastructure reused as-is**: `FeatureNormalizer` (train-split-only fit enforced), `create_dataloader` + `EpochMarketSampler` (seeded), `ExperimentRegistry` (DuckDB; has `objective`, `encoder`, `loss`, `seed`, `git_commit`, `hardware`, `software`, `metrics` columns), stage loggers
- **Zero modifications** to `src/data/`, `storage/`, or any frozen config/manifest

## 4. Architecture — model_v1

~25M params. Encoder-only, Pre-LN, no custom attention, no flashiness.

| Component | Spec |
|---|---|
| Layers | 8 |
| Heads | 8 |
| d_model | 512 |
| FFN | 2048 (GELU) |
| Dropout | 0.1 |
| context_length | 512 **data** timesteps — asserted equal to `windowing_v1.yaml sequence_length` at build time |
| cls_token | true → encoder / RoPE cache / padding mask sized **513** |
| Embedding dim | 512 (per-timestep latent + pooled) |

### 4.1 Projection (`projection.py`)

`Linear(15→512)`. Nothing else. `FeatureReconstructionHead` (SSL-only, same module):
- `Linear(512→5)` price group · `Linear(512→2)` funding/OI · 8 `Linear(512→classes)` calendar CE heads (cardinalities in §5).

### 4.2 Positional encoding (`positional_encoding.py`)

RoPE (θ=10000), cos/sin cache precomputed to `context_length + 1`. **Time-aware**: CLS (sequence index 0) gets RoPE position 0; data token *j* gets position `1 + (ts_j − ts_0)/60000` (integer minute offsets). Data–data relative geometry preserved exactly; small intra-window gaps (tolerated by the windowing contract) get correct relative distances.

### 4.3 Transformer block (`transformer.py`)

`LayerNorm → MultiheadAttention(RoPE, key_padding_mask) → +residual → LayerNorm → Linear 512→2048 → GELU → Linear 2048→512 → +residual`, dropout 0.1. Uses standard `nn.MultiheadAttention(batch_first=True)`.

### 4.4 Encoder (`encoder.py`)

8 blocks + final LayerNorm → latent `[B, 513, 512]`. Data latents = positions 1..512.

### 4.5 Pooling (`embeddings.py`)

Three pooling modes, all extracted, compared in `linear_probe.py`:
- `cls` — CLS-slot latent
- `mean` — mask-aware mean over data positions 1..512 (respects key_padding_mask)
- `attention` — learned query vector attending over data positions (mask-aware; excludes CLS)

## 5. SSL objective — Masked Market Modeling (`losses/masked_modeling.py`)

ONE objective only. `contrastive.py` / `temporal.py` exist as documented placeholders raising `NotImplementedError` (mirrors the alignment engine's future-modality pattern).

### 5.1 MaskGenerator

Per sample, uniformly select **15% of the 512 data positions** where `mask==True`. Persistent `torch.Generator` seeded from run seed → deterministic with the seeded `EpochMarketSampler`.

**CLS seam invariant**: masking is applied to the `[B,512,15]` data tensor **before** the model prepends CLS internally — the generator never sees the 513-length sequence. The 513-length `key_padding_mask` is built inside the model forward as `torch.cat([all-valid CLS row], batch_mask)`; the CLS slot is hardcoded valid, never padding, never a mask/corruption candidate. Runtime assertion in the training step: `key_padding_mask[:, 0].all()`. Covered by dedicated tests.

### 5.2 Corruption

Zero the full 15-dim normalized feature vector at masked data positions (whole-timestep masking).

### 5.3 No future leakage

Targets are same-timestep features; contract `prediction_horizon=0`; bidirectional attention over the window is legitimate.

### 5.4 Loss

Over masked data positions only:
- price group (idx 0–4) → **Huber** (δ=1.0) on normalized values
- funding/OI (idx 5–6) → **MSE** on normalized values
- calendar (idx 7–14) → per-field **CrossEntropy** on raw integer targets (0-based offsets; see cardinalities table)
- elements with `feature_mask==False` contribute 0 (CE: `ignore_index=-100`)
- group weights configurable (default 1.0); total = weighted sum

**Calendar CE head cardinalities** (pure 0-based, no sentinel classes; missing modality → `ignore_index`):

| field | raw range | target | classes |
|---|---|---|---|
| minute_of_day | 0–1439 | raw | 1440 |
| hour | 0–23 | raw | 24 |
| day_of_week | 0–6 | raw | 7 |
| day_of_month | 1–31 | raw−1 | 31 |
| month | 1–12 | raw−1 | 12 |
| quarter | 1–4 | raw−1 | 4 |
| is_weekend | 0/1 | raw | 2 |
| year | 2020–2035 | raw−2020 | 16 (out-of-range → ignore) |

## 6. Training pipeline

### 6.1 Stride invariant

Windows are cut **only** through frozen `windowing_v1.yaml` (stride 1). `train_window_stride` / `val_window_stride` are deterministic index subsamples (`windows[::k]`) of the version-locked window set — never a second windowing configuration, never passed to `WindowingEngine`. `max_train_windows` / `max_val_windows` (smoke/debug) select a deterministic seeded subset of the subsampled list.

### 6.2 Optimizer (`optimizer_v1.yaml`)

AdamW, lr 3e-4, weight_decay 0.01, betas (0.9, 0.999), eps 1e-8, grad clip 1.0.

### 6.3 Scheduler (`scheduler.py`)

Linear warmup over 5% of total steps → cosine decay to floor (1e-6). Step-level; restored exactly on resume.

### 6.4 Checkpoint (`checkpoint.py`)

Run dir `models/foundation/teacher_v1/<run_id>/` (run_id = UTC timestamp + short config hash):
- `checkpoint_epoch{N}.pt` — weights, optimizer, scheduler, epoch/step, loss history, RNG states, normalizer state
- `manifest.json` — run_id, snapshot ID, dataset fingerprint, git commit (subprocess, `"unknown"` fallback — repo is not git), all three configs verbatim, seeds, best val loss, per-checkpoint SHA256
- `latest.json` — pointer (no Windows symlinks) · `history.json` — loss curves · `normalizer.json`
- Full resume support; best-val-loss checkpoint tracked

### 6.5 Trainer (`trainer.py`)

Load frozen manifest splits (train BTC+ETH, val SOL, futures) → build windows via frozen contract → stride-subsample → fit `FeatureNormalizer` (zscore) on **train symbols only** → loop: batch → normalize → mask (data positions) → forward → loss → backward → clip 1.0 → AdamW step → scheduler step → periodic SOL validation (loss only) → checkpoint cadence.

Logs per `log_every` steps: train loss, val loss/epoch, LR, grad norm, tokens/sec, RAM (psutil), GPU memory (CUDA-guarded), checkpoint hash → `logs/train_teacher.log` + console.

Run end → `ExperimentRegistry.log_experiment(objective="masked_market_modeling", encoder="teacher_transformer_v1", …)`.

Determinism: seeds 42 from manifest (python/numpy/torch), seeded sampler, seeded mask generator.

### 6.6 CLI (`train_teacher.py`)

```
python -m src.training.train_teacher \
  --model-config configs/model_v1.yaml \
  --optimizer-config configs/optimizer_v1.yaml \
  --trainer-config configs/trainer_v1.yaml \
  [--resume models/foundation/teacher_v1/<run_id>] [--smoke]
```

`--smoke` overrides: 2 layers / 4 heads / d_model 128 / d_ff 512, batch 8, `max_train_windows 256`, `max_val_windows 64`, 1 epoch, CPU — validates the full loop here in minutes (loss decreases, checkpoint written, registry logged).

### 6.7 trainer_v1.yaml (GPU defaults)

```yaml
trainer:
  batch_size: 64
  epochs: 10
  num_workers: 4
  device: auto               # "cuda" if available else "cpu" (smoke overrides)
  seed: 42
  market: futures
  train_window_stride: 16    # sampling stride over frozen stride-1 window set (invariant, §6.1)
  val_window_stride: 16      # same invariant; keeps SOL validation tractable
  max_train_windows: null    # deterministic subset cap (smoke: 256)
  max_val_windows: null      # deterministic subset cap (smoke: 64)
  log_every: 50              # steps
  val_every: 1               # epochs
  checkpoint_every: 1        # epochs
  normalizer:
    mode: zscore
```

### 6.8 GPU pilot scale estimate (informational)

~65K windows/epoch ≈ 33M tokens; batch 64 → ~1,030 steps/epoch; single modern GPU → minutes/epoch; 10 epochs under a few hours.

## 7. Embedding extraction API (`models/teacher/embeddings.py`)

`EmbeddingExtractor(checkpoint_dir, pooling={"cls"|"mean"|"attention"}, device)` — loads model + normalizer state from checkpoint, iterates any `MarketDataset` via existing `create_dataloader` (unshuffled), returns per-window 512-d embeddings + `{symbol, window_start_ms, window_end_ms}` → saved as `evaluation/embedding/embeddings/<run_id>/<split>_<pooling>.npz`.

This is the **reusable-embeddings deliverable** and the input to all five eval modules.

## 8. Evaluation suite (`src/evaluation/embedding/` → reports to `evaluation/embedding/`)

No trading metrics. Regime labels are derived **at eval time** from raw window data (never model input):
- volatility: std of intra-window log returns → quantile buckets (low/mid/high, quantiles fit on train split)
- trend: window net return |R| vs train-median → trending/ranging
- liquidity: mean volume → quantile buckets
- probe targets: range expansion (next-window range vs train median, binary) · next-hour return direction (close[t+60m] vs window-end close, ±0.1% dead zone → up/flat/down)

| Module | Function |
|---|---|
| **clustering.py** | sklearn KMeans (k default 8); silhouette score; AMI vs derived volatility/trend/liquidity labels; per-symbol cluster distribution (does unseen SOL populate the same clusters?) |
| **retrieval.py** | cosine kNN (k=10); same-regime hit rate; cross-symbol neighbor fraction (BTC/ETH queries ↔ SOL neighbors) |
| **temporal_consistency.py** | cosine similarity of embeddings of grid-adjacent windows vs random pairs; separation AUC; embedding-velocity curve over time |
| **linear_probe.py** | frozen encoder; sklearn LogisticRegression probes (volatility bucket, range expansion, next-hour direction, liquidity regime); trained on BTC+ETH embeddings, tested on **SOL (held-out)** and BTC (in-sample); baselines: majority class + LogReg on last-timestep raw 15 features; outputs the **CLS/mean/attention pooling comparison table** |
| **visualization.py** | PCA + t-SNE (capped ~5K points) colored by symbol/regime/month; loss curves from `history.json` → `evaluation/embedding/figures/` |

Each module exposes functions + `__main__` CLI guard (repo convention, cf. `benchmark.py`) and writes JSON reports.

## 9. Configs (interface of record)

### model_v1.yaml

```yaml
model:
  name: teacher_transformer_v1
  feature_dim: 15
  context_length: 512   # DATA timesteps per window. MUST equal windowing_v1.yaml sequence_length (asserted at build).
  cls_token: true       # Prepends 1 learned token → encoder/RoPE/padding-mask sized context_length + 1 = 513.
  d_model: 512
  n_layers: 8
  n_heads: 8
  d_ff: 2048
  dropout: 0.1
  rope_theta: 10000.0
  pooling: [cls, mean, attention]

loss:
  masked_modeling:
    mask_ratio: 0.15
    price_indices: [0,1,2,3,4]
    funding_oi_indices: [5,6]
    calendar:
      minute_of_day: {index: 7,  classes: 1440, offset: 0}
      hour:          {index: 8,  classes: 24,   offset: 0}
      day_of_week:   {index: 9,  classes: 7,    offset: 0}
      day_of_month:  {index: 10, classes: 31,   offset: 1}
      month:         {index: 11, classes: 12,   offset: 1}
      quarter:       {index: 12, classes: 4,    offset: 1}
      year:          {index: 13, classes: 16,   offset: 2020}
      is_weekend:    {index: 14, classes: 2,    offset: 0}
    group_weights:
      price: 1.0
      funding_oi: 1.0
      calendar: 1.0
```

### optimizer_v1.yaml

```yaml
optimizer:
  adamw:
    lr: 3.0e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1.0e-8
  grad_clip: 1.0

scheduler:
  warmup_frac: 0.05
  decay: cosine
  lr_floor: 1.0e-6
```

### trainer_v1.yaml

```yaml
trainer:
  batch_size: 64
  epochs: 10
  num_workers: 4
  device: auto
  seed: 42
  market: futures

  # stride: sampling over frozen stride-1 window set, NOT re-invoking windowing_engine
  train_window_stride: 16
  val_window_stride: 16

  max_train_windows: null   # deterministic subset cap (smoke: 256)
  max_val_windows: null     # deterministic subset cap (smoke: 64)

  log_every: 50
  val_every: 1
  checkpoint_every: 1

  normalizer:
    mode: zscore
```

## 10. File layout

```
configs/model_v1.yaml  optimizer_v1.yaml  trainer_v1.yaml
src/
  models/__init__.py
  models/teacher/
    __init__.py
    projection.py              # FeatureProjection + ReconstructionHead
    positional_encoding.py     # RoPE, time-aware
    transformer.py             # Pre-LN block
    encoder.py                 # N blocks + final LN
    embeddings.py              # CLS/mean/attention pooling + EmbeddingExtractor
  training/
    train_teacher.py           # CLI entry
    trainer.py                 # loop
    checkpoint.py              # versioned run dirs + manifest
    optimizer.py               # AdamW factory
    scheduler.py               # warmup + cosine
    losses/
      __init__.py
      masked_modeling.py       # MaskGenerator + grouped loss
      contrastive.py           # NotImplementedError placeholder
      temporal.py              # NotImplementedError placeholder
  evaluation/__init__.py
  evaluation/embedding/
    __init__.py
    clustering.py
    retrieval.py
    linear_probe.py
    temporal_consistency.py
    visualization.py
tests/
  test_projection.py
  test_transformer.py
  test_masking.py
  test_losses.py
  test_checkpoint.py
  test_training.py
  test_embeddings.py
  test_linear_probe.py
```

Artifacts at run time:
- `models/foundation/teacher_v1/<run_id>/` — checkpoints, manifest, history
- `evaluation/embedding/` — reports JSON, figures, extracted embeddings `.npz`

## 11. Tests (8 new files; existing 98-test suite must stay green)

| File | Key cases |
|---|---|
| `test_projection.py` | 15→512 shape; head output shapes/cardinalities (incl. 31/12/4 exact, year 16); init determinism |
| `test_transformer.py` | output `[B,513,512]`; **CLS slot never padding** (all-data-padded-except-one batch still yields finite CLS); padding a real timestep never alters valid outputs; RoPE shift-equivariance; gradient flow to projection |
| `test_masking.py` | ~15% of **valid data positions**; **CLS index never masked** (Amendment E); masked rows zeroed; seed determinism; targets untouched (no leakage) |
| `test_losses.py` | per-group correctness on hand-built tensors; `feature_mask=False` → zero contribution; CE offsets (raw−1 for 1-based fields; raw−2020 for year; OOR year ignored) |
| `test_checkpoint.py` | save/load roundtrip restores weights/optimizer/scheduler/history/RNG; manifest fields present; SHA256 stable; resume continues at correct step |
| `test_training.py` | one smoke step reduces loss; full-run determinism under fixed seed; normalizer fit touches train split only (raises on val) |
| `test_embeddings.py` | pooling shapes; masked-mean correctness (padding excluded); CLS ≠ mean ≠ attention outputs; extraction determinism |
| `test_linear_probe.py` | probe beats majority baseline on synthetic separable embeddings; frozen encoder receives no gradients |

## 12. Deliverables checklist

| Deliverable | Status |
|---|---|
| Teacher Foundation Model (§4) | Implementation |
| Reproducible training pipeline (§6) | Implementation |
| Versioned checkpoints with hashes/manifest (§6.4) | Implementation |
| Embedding extraction API (§7) | Implementation |
| Embedding evaluation suite (§8) | Implementation |
| Linear probing framework (§8.4) | Implementation |
| `MODEL_CARD.md` + `TRAINING_REPORT.md` | Written after pilot run |
| `Phase2_Implementation_Plan_v1.md` | This document |

## 13. Out of scope (unchanged)

Trading strategies · RL agents · PPO/DQN · portfolio optimization · risk management · order execution · student model · Mixture of Experts · multi-modal fusion beyond the Phase 1 feature set · tool use / autonomous agents. No modifications to `src/data/`, `storage/`, frozen configs, or the manifest.

## 14. Risks

- **CPU-only environment**: full pilot deferred to GPU; this box delivers implementation + smoke validation + runbook (accepted, D1)
- **Single-year, 3-symbol pilot**: embeddings may show limited regime diversity; by design — scale follows in later phases without architecture changes
- **Year CE head**: snapshot contains only 2024, so the year head is effectively trivial during the pilot; cardinality 16 with `year_base 2020` keeps the head correct for future years
- **Repo is not git**: checkpoint manifest records git commit via subprocess with `"unknown"` fallback

## 15. Execution order

1. requirements + three configs
2. `src/models/teacher/` (projection → RoPE → transformer → encoder → embeddings) + tests: projection, transformer, embeddings
3. `losses/` (masked_modeling + 2 placeholders) + tests: masking, losses
4. optimizer / scheduler / checkpoint + test: checkpoint
5. trainer + train_teacher + test: training → **CPU `--smoke` run** (end-to-end validation on this box)
6. `src/evaluation/embedding/` suite + test: linear_probe
7. Full pilot training — **deferred to CUDA host** (single command, documented in TRAINING_REPORT.md)
8. `MODEL_CARD.md`, `TRAINING_REPORT.md`, README/ARCHITECTURE updates
9. Full test suite green (98 existing + 8 new files)
