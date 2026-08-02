# Model Card — teacher_transformer_v1

## 1. Identity

| Field | Value |
|---|---|
| Name | `teacher_transformer_v1` |
| Type | Masked Market Modeling (MMM) encoder, Transformer, Phase 2 teacher |
| Paradigm | Self-supervised, **no labels** (see `RESEARCH_PHILOSOPHY.md`) |
| Feature layout | 15-dim market state (OHLCV 0-4, funding_rate/OI 5-6, calendar 7-14) |
| Context | 512 one-minute steps |
| Version | v1 (`configs/model_v1.yaml` + `model_returns_v1.yaml`) |

## 2. Architecture

| Parameter | Full config | Smoke (CPU test) |
|---|---|---|
| `d_model` | 512 | 128 |
| `n_layers` | 8 | 2 |
| `n_heads` | 8 | 4 |
| `d_ff` | 2048 | 512 |
| `dropout` | 0.1 | 0.1 |
| `context_length` | 512 | 512 |
| `cls_token` | true | true |
| `rope_theta` | 10000.0 | 10000.0 |
| `feature_dim` | 15 | 15 |
| Pooling heads | `cls`, `mean`, `attention` (attention pooler untrained) | same |

- Time-aware positions via RoPE; CLS is the seam at position 0, data positions
  are 1 + minute offset (see `RESEARCH_PHILOSOPHY.md` §architecture).
- The three pooling heads exist for evaluation; `mean` is the default pooling.

## 3. Training objective

Masked reconstruction, in three groups weighted equally (`group_weights` 1.0):

| Group | Feature indices | Loss |
|---|---|---|
| price | 0-4 (OHLC, close, volume) | Huber |
| funding_oi | 5-6 | MSE |
| calendar | 7-14 | Cross-entropy (per-field classes) |

- `mask_ratio` 0.15 over valid data positions; the CLS seam is never masked.
- Raw variant (`model_v1.yaml`): random-position masking, calendar
  reconstructed.
- Returns variant (`model_returns_v1.yaml`): contiguous-span masking
  (`mask_mode: span`, `span_len: 16`), calendar kept as input but **excluded**
  from the reconstruction target (`reconstruct_calendar: false`), trained on
  `feature_style: returns`.
- Optimizer: AdamW lr 3e-4, wd 0.01, betas (0.9, 0.999), eps 1e-8, grad clip
  1.0; scheduler warmup 5% + cosine decay to floor 1e-6. Seed 42.

## 4. Training data

| Field | Value |
|---|---|
| Market | Binance futures, 1-minute |
| Symbols | train BTCUSDT+ETHUSDT; validation SOLUSDT (held-out); test BTCUSDT+ETHUSDT |
| Time split | train_end `2024-11-30` (train windows capped; test starts at boundary) |
| Canonical rows | 510 files across symbols (see `DATA_FLOW.md`) |
| Dataset fingerprint | `328a7b67b070b95e47ba450452032a93dfa410431e0cf329de6a4ac7b5ae3875` |
| Snapshot | `storage/training/snapshots/2026-07-30` |
| Preprocessing | aligned market-state rows; z-score normalizer fit on train symbols only |

## 5. Training

- See `TRAINING_GUIDE.md` for commands, smoke/full variants, resume.
- Runtime shape (full): stride 16 ⇒ ~65 K windows/epoch ≈ 33 M tokens; batch
  64 ⇒ ~1,030 steps/epoch; CUDA recommended (`GPU_TRAINING_GUIDE.md`).

## 6. Evaluation

Protocol details: `EVALUATION_GUIDE.md`. Summary of the latest artifact
(**smoke checkpoint** `20260801_214517_smoke`, `mean` pooling, so read this as
a pipeline-working check, not model quality):

### 6.1 Linear probe (from `evaluation/embedding/linear_probe_mean.json`)

| Probe | cross-symbol bacc (SOL) | in-sample bacc | majority baseline |
|---|---|---|---|
| volatility | 0.5 | 0.8672 | 0.1094 |
| range_expansion | 0.5 | 0.8594 | 0.0938 |
| liquidity | 1.0 | 0.9805 | 0.0 |

- In-sample the encoder is highly readable (volatility 0.8672, liquidity
  0.9805 vs. near-zero majority baselines).
- Cross-symbol on SOL: liquidity generalizes perfectly (1.0), but
  volatility/range sit at 0.5 (chance) — the 2-layer smoke model does **not**
  yet transfer regime structure across symbols. This is the key open question
  the full model should move (see §8).

### 6.2 Baseline harness (from `evaluation/baselines/baseline_eval_20260801_215055.json`)

Eval period 2024-12-01 → 2024-12-31 across BTCUSDT, ETHUSDT, SOLUSDT;
fit period ends 2024-11-30; horizon 15 min; 4,497 eval windows.

| Task | Majority | Persistence | Raw linear | Handcrafted | **Embedding (mean)** |
|---|---|---|---|---|---|
| future_return | 0.5 | 0.5089 | 0.4952 | 0.5053 | 0.5016 |
| volatility | 0.5 | — | 0.9561 | 0.9912 | 0.9558 |
| range_expansion | 0.5 | — | 0.9297 | 0.9966 | 0.9488 |
| liquidity | 0.5 | — | 0.7597 | 0.9966 | 0.5896 |

(bacc, ROC-AUC available in the JSON.)

## 7. Intended use / limitations

- **Use for**: representation learning research, regime/state retrieval, probe
  baselines; as the teacher for a later student / downstream Phase 3.
- **Do NOT use for**: trading decisions, return forecasting, deployment.
  `future_return` ≈ 0.5 by design (no-label objective); never read positive
  return bacc as signal without investigating leakage first.
- **Known limitations**: pilot-scale data; smoke checkpoints are tiny; the
  attention pooler is untrained; no market-share claims.

## 8. Intended next-step checks (how to read a full run)

When a full (`d_model 512`, 10-epoch) run exists, compare against §6:

1. Does **cross-symbol** linear-probe bacc on SOL clear the majority baseline
   for volatility / range_expansion (i.e. did regime structure transfer)?
2. Does the embedding-vs-handcrafted gap in the baseline harness narrow?
3. Are loss curves converged (see `visualization.py` loss_curves.png) without
   train/val divergence?
4. Confirm `future_return` stays at chance — evidence of no label leakage.
