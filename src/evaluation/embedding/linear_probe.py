"""Linear probing evaluation on frozen encoder embeddings.

Trains tiny logistic regression heads on frozen embeddings for
window-level pseudo-labels (volatility bucket, range expansion,
liquidity regime) derived from raw window features. Thresholds are
computed on the TRAIN split only, then applied to the held-out
cross-symbol split. Reports balanced accuracy vs majority baselines.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# feature_order indices (market_state_schema_v1.json)
_IDX_HIGH, _IDX_LOW, _IDX_CLOSE, _IDX_VOLUME = 1, 2, 3, 4
# returns-style layout (feature_builder.py "returns"): 0 log_return, 1 hl_range, 3 log_volume
_RET_LOG_RET, _RET_RANGE, _RET_LOG_VOLUME = 0, 1, 3
_STAT_KEYS = ("volatility", "range", "volume")


def _vectorized_window_stats(features: np.ndarray, style: str = "raw") -> dict:
    """Batched per-window stats for stacked windows ``features: [N, T, F]``.

    Vectorized equivalent of :func:`window_stats` (numerically identical), the
    dominant cost of the probe. Values are plain ndarrays (not floats).
    """
    if style == "returns":
        lr = features[:, :, _RET_LOG_RET]
        lr = np.nan_to_num(np.where(np.isfinite(lr), lr, np.nan), nan=0.0)
        volatility = np.std(lr, axis=1)
        range_ = np.mean(features[:, :, _RET_RANGE], axis=1)
        log_volume = features[:, :, _RET_LOG_VOLUME]
        volume = np.mean(np.expm1(np.clip(log_volume, 0.0, None)), axis=1)
    else:
        close = features[:, :, _IDX_CLOSE]
        high = features[:, :, _IDX_HIGH]
        low = features[:, :, _IDX_LOW]
        volume = features[:, :, _IDX_VOLUME]
        log_ret = np.diff(np.log(np.maximum(close, 1e-10)), axis=1)
        volatility = np.std(log_ret, axis=1)
        range_ = np.mean((high - low) / np.maximum(close, 1e-10), axis=1)
        volume = np.mean(volume, axis=1)
    return {"volatility": volatility, "range": range_, "volume": volume}


def window_stats(features: np.ndarray, style: str = "raw") -> dict:
    """Per-window scalar stats from (unnormalized) window features [T, 15].

    ``raw`` reads the OHLCV price/volume columns; ``returns`` derives the same
    scalars from the stationary return-style features so probe pseudo-labels
    stay meaningful for both feature styles.
    """
    if style == "returns":
        lr = features[:, _RET_LOG_RET]
        lr = lr[np.isfinite(lr)]
        return {
            "volatility": float(np.std(lr)) if len(lr) else 0.0,
            "range": float(np.mean(features[:, _RET_RANGE])),
            "volume": float(np.mean(np.expm1(np.clip(features[:, _RET_LOG_VOLUME], 0.0, None)))),
        }
    close = features[:, _IDX_CLOSE]
    high = features[:, _IDX_HIGH]
    low = features[:, _IDX_LOW]
    volume = features[:, _IDX_VOLUME]
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    return {
        "volatility": float(np.std(log_ret)) if len(log_ret) else 0.0,
        "range": float(np.mean((high - low) / np.maximum(close, 1e-10))),
        "volume": float(np.mean(volume)),
    }


def _labels_from_stats(stats: list, thresholds: dict) -> dict:
    vol = np.array([s["volatility"] for s in stats])
    rng = np.array([s["range"] for s in stats])
    liq = np.array([s["volume"] for s in stats])
    return {
        "volatility": (vol > thresholds["volatility"]).astype(np.int64),
        "range_expansion": (rng > thresholds["range"]).astype(np.int64),
        "liquidity": (liq > thresholds["volume"]).astype(np.int64),
    }


def evaluate_probe(train_emb: dict, test_emb: dict, test_train_emb: dict) -> dict:
    results = {}
    baselines = {}
    targets = ["volatility", "range_expansion", "liquidity"]

    for target_name in targets:
        y_train = np.array(train_emb[target_name])
        y_test = np.array(test_emb[target_name])
        y_test_train = np.array(test_train_emb[target_name])

        classes, counts = np.unique(y_train, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        baseline_acc = float((y_test == majority_class).mean())
        baselines[target_name] = round(baseline_acc, 4)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_emb["embedding"], y_train)

        pred_sol = clf.predict(test_emb["embedding"])
        bacc_sol = float(balanced_accuracy_score(y_test, pred_sol))

        pred_btc = clf.predict(test_train_emb["embedding"])
        bacc_btc = float(balanced_accuracy_score(y_test_train, pred_btc))

        results[target_name] = {
            "cross_symbol_bacc": round(bacc_sol, 4),
            "in_sample_bacc": round(bacc_btc, 4),
            "majority_baseline": round(baseline_acc, 4),
        }

    return {
        "probes": results,
        "baselines": baselines,
        "n_train": len(train_emb["embedding"]),
        "n_test_cross": len(test_emb["embedding"]),
        "n_test_insample": len(test_train_emb["embedding"]),
    }


def _extract_with_labels(model, normalizer, split, pooling, trainer_cfg, device, max_windows):
    import time
    from src.evaluation.embedding._common import build_split_dataset, extract_split_embeddings

    dataset = build_split_dataset(split, trainer_cfg, max_windows)
    style = trainer_cfg.get("feature_style", "raw")
    windows = dataset.windows
    n = len(windows)

    keys = _STAT_KEYS
    cols = {k: np.empty(n, dtype=np.float64) for k in keys}
    chunk = 8192
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        feats = np.stack([w["features"] for w in windows[start:end]])
        s = _vectorized_window_stats(feats, style)
        for k in keys:
            cols[k][start:end] = s[k]
        if start % (chunk * 4) == 0:
            print(f"[{split}] stats {end}/{n} ({end / max(n, 1):.0%}) "
                  f"{time.time() - t0:.1f}s", flush=True)
    stats = [
        {"volatility": float(cols["volatility"][i]),
         "range": float(cols["range"][i]),
         "volume": float(cols["volume"][i])}
        for i in range(n)
    ]
    emb = extract_split_embeddings(
        model, normalizer, split, pooling, trainer_cfg, device,
        max_windows=max_windows, dataset=dataset,
    )
    assert len(emb["embedding"]) == len(stats)
    return emb, stats


def main():
    from src.evaluation.embedding._common import load_model_and_normalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, configs = load_model_and_normalizer(Path(args.checkpoint), device)
    trainer_cfg = configs["trainer_config"]

    train_emb, train_stats = _extract_with_labels(
        model, normalizer, "train", args.pooling, trainer_cfg, device, args.max_windows)
    print(f"[{args.pooling}] train extracted "
          f"({len(train_emb['embedding'])} windows)", flush=True)
    test_emb, test_stats = _extract_with_labels(
        model, normalizer, "validation", args.pooling, trainer_cfg, device, args.max_windows)
    print(f"[{args.pooling}] validation extracted "
          f"({len(test_emb['embedding'])} windows)", flush=True)

    # In-sample baseline: same train windows used for fitting the probe
    test_train_emb, test_train_stats = train_emb, train_stats

    # Thresholds from TRAIN split only (no cross-split leakage)
    thresholds = {
        "volatility": float(np.median([s["volatility"] for s in train_stats])),
        "range": float(np.median([s["range"] for s in train_stats])),
        "volume": float(np.median([s["volume"] for s in train_stats])),
    }

    train_emb.update(_labels_from_stats(train_stats, thresholds))
    test_emb.update(_labels_from_stats(test_stats, thresholds))
    test_train_emb = dict(test_train_emb)
    test_train_emb.update(_labels_from_stats(test_train_stats, thresholds))

    results = evaluate_probe(train_emb, test_emb, test_train_emb)
    results["thresholds"] = {k: round(v, 8) for k, v in thresholds.items()}

    out_dir = Path("evaluation/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"linear_probe_{args.pooling}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Linear probe results written to {path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
