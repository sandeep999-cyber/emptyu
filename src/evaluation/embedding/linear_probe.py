"""Linear probing evaluation on frozen encoder embeddings.

Trains tiny logistic regression heads on frozen embeddings for
window-level pseudo-labels (volatility bucket, range expansion,
liquidity regime) derived from raw window features. Thresholds are
computed on the TRAIN split only, then applied to the held-out
cross-symbol split. Reports balanced accuracy vs majority baselines.
"""

import argparse
import hashlib
import json
import os
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
_CACHE_VERSION = 1


def _vectorized_window_stats(features: np.ndarray, style: str = "raw") -> dict:
    """Batched per-window stats for stacked windows ``features: [N, T, F]``.

    Vectorized equivalent of :func:`window_stats` (numerically identical), the
    dominant cost of the probe. Values are plain ndarrays (not floats).
    """
    if style == "returns":
        lr = features[:, :, _RET_LOG_RET]
        # Match per-window window_stats: non-finite are filtered OUT, so std is
        # taken over the finite subset (nanstd ignores NaN = same as dropping).
        vol = np.nanstd(np.where(np.isfinite(lr), lr, np.nan), axis=1)
        volatility = np.nan_to_num(vol, nan=0.0)  # all-non-finite window -> 0.0 (as before)
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


def _stats_for(windows, style: str) -> list:
    import time
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
            print(f"stats {end}/{n} ({end / max(n, 1):.0%}) "
                  f"{time.time() - t0:.1f}s", flush=True)
    return [
        {"volatility": float(cols["volatility"][i]),
         "range": float(cols["range"][i]),
         "volume": float(cols["volume"][i])}
        for i in range(n)
    ]


def _cache_key(checkpoint: Path, trainer_cfg: dict, max_windows, poolings, device, batch_size: int) -> str:
    manifest = checkpoint / "manifest.json"
    fingerprint = Path("storage/training/dataset_fingerprint.json")
    payload = {
        "version": _CACHE_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "manifest": manifest.read_bytes().hex() if manifest.exists() else "",
        "fingerprint": fingerprint.read_bytes().hex() if fingerprint.exists() else "",
        "trainer": trainer_cfg,
        "max_windows": max_windows,
        "poolings": sorted(poolings),
        "device": device.type,
        "batch_size": batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _stats_array(stats: list) -> np.ndarray:
    return np.asarray([[s[k] for k in _STAT_KEYS] for s in stats], dtype=np.float64)


def _stats_list(values: np.ndarray) -> list:
    return [
        {key: float(row[i]) for i, key in enumerate(_STAT_KEYS)}
        for row in values
    ]


def _load_cached(cache_dir: Path, poolings: list[str]):
    required = [cache_dir / "complete.json", cache_dir / "train_stats.npy", cache_dir / "validation_stats.npy"]
    required += [cache_dir / f"train_{p}.npy" for p in poolings]
    required += [cache_dir / f"validation_{p}.npy" for p in poolings]
    if not all(path.exists() for path in required):
        return None
    train_stats = _stats_list(np.load(cache_dir / "train_stats.npy", mmap_mode="r"))
    validation_stats = _stats_list(np.load(cache_dir / "validation_stats.npy", mmap_mode="r"))
    train = {
        p: {"embedding": np.load(cache_dir / f"train_{p}.npy", mmap_mode="r")}
        for p in poolings
    }
    validation = {
        p: {"embedding": np.load(cache_dir / f"validation_{p}.npy", mmap_mode="r")}
        for p in poolings
    }
    return train, validation, train_stats, validation_stats


def _save_cached(cache_dir: Path, train_embeddings: dict, validation_embeddings: dict,
                 train_stats: list, validation_stats: list) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train_stats.npy", _stats_array(train_stats))
    np.save(cache_dir / "validation_stats.npy", _stats_array(validation_stats))
    for pooling, result in train_embeddings.items():
        np.save(cache_dir / f"train_{pooling}.npy", result["embedding"])
    for pooling, result in validation_embeddings.items():
        np.save(cache_dir / f"validation_{pooling}.npy", result["embedding"])
    (cache_dir / "complete.json").write_text(json.dumps({"version": _CACHE_VERSION}, indent=2))


def _extract_with_oom_fallback(
    model, normalizer, split, poolings, trainer_cfg, device, dataset, batch_size
):
    """Retry CUDA extraction with a smaller batch if VRAM is insufficient."""
    from src.evaluation.embedding._common import extract_split_embeddings_multi

    current = batch_size
    while True:
        try:
            return extract_split_embeddings_multi(
                model, normalizer, split, poolings, trainer_cfg, device,
                batch_size=current, dataset=dataset,
            )
        except RuntimeError as exc:
            is_oom = device.type == "cuda" and "out of memory" in str(exc).lower()
            if not is_oom or current <= 1:
                raise
            torch.cuda.empty_cache()
            current = max(1, current // 2)
            print(f"[runtime] CUDA OOM on batch; retrying {split} with batch_size={current}",
                  flush=True)


def main():
    from src.evaluation.embedding._common import (
        load_model_and_normalizer, build_split_dataset, extract_split_embeddings_multi)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="all",
                        choices=["cls", "mean", "attention", "all"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Embedding extraction batch. CPU-only machines are "
                             "often fastest around 32 on older CPUs; GPU use 32-64 "
                             "(VRAM). Default: 32 on CPU, 32 on CUDA.")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU torch threads. Default: min(12, CPU count).")
    parser.add_argument("--cache-dir", type=str, default=".cache/linear_probe",
                        help="Persistent embedding cache directory; use --no-cache to disable.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force dataset/model extraction instead of using the local cache.")
    args = parser.parse_args()

    poolings = ["cls", "mean", "attention"] if args.pooling == "all" else [args.pooling]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        threads = args.threads or min(12, os.cpu_count() or 1)
        torch.set_num_threads(threads)
    else:
        threads = torch.get_num_threads()
    batch_size = args.batch_size or 32
    print(f"[runtime] device={device}, batch_size={batch_size}, threads={threads}", flush=True)
    model, normalizer, configs = load_model_and_normalizer(Path(args.checkpoint), device)
    trainer_cfg = configs["trainer_config"]
    style = trainer_cfg.get("feature_style", "raw")
    checkpoint = Path(args.checkpoint)
    cache_dir = Path(args.cache_dir) / _cache_key(
        checkpoint, trainer_cfg, args.max_windows, poolings, device, batch_size)

    cached = None if args.no_cache else _load_cached(cache_dir, poolings)
    if cached is not None:
        train_embeddings, test_embeddings, train_stats, test_stats = cached
        print(f"[cache] loaded {cache_dir}", flush=True)
    else:
        # Build each split dataset ONCE and share across all poolings.
        print(f"[data] building train split...", flush=True)
        train_ds = build_split_dataset("train", trainer_cfg, args.max_windows)
        train_stats = _stats_for(train_ds.windows, style)
        print(f"[data] train windows={len(train_ds.windows)}", flush=True)
        print(f"[data] building validation split...", flush=True)
        test_ds = build_split_dataset("validation", trainer_cfg, args.max_windows)
        test_stats = _stats_for(test_ds.windows, style)
        print(f"[data] validation windows={len(test_ds.windows)}", flush=True)

        print(f"[encoder] extracting {', '.join(poolings)} from train in one pass", flush=True)
        train_embeddings = _extract_with_oom_fallback(
            model, normalizer, "train", poolings, trainer_cfg, device, train_ds, batch_size)
        print(f"[encoder] extracting {', '.join(poolings)} from validation in one pass", flush=True)
        test_embeddings = _extract_with_oom_fallback(
            model, normalizer, "validation", poolings, trainer_cfg, device, test_ds, batch_size)
        if not args.no_cache:
            _save_cached(cache_dir, train_embeddings, test_embeddings, train_stats, test_stats)
            print(f"[cache] saved {cache_dir}", flush=True)

    # Thresholds from TRAIN split only (no cross-split leakage).
    thresholds = {
        "volatility": float(np.median([s["volatility"] for s in train_stats])),
        "range": float(np.median([s["range"] for s in train_stats])),
        "volume": float(np.median([s["volume"] for s in train_stats])),
    }
    train_labels = _labels_from_stats(train_stats, thresholds)
    test_labels = _labels_from_stats(test_stats, thresholds)

    out_dir = Path("evaluation/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pooling in poolings:
        print(f"=== Pooling: {pooling} ===", flush=True)
        train_emb = train_embeddings[pooling]
        test_emb = test_embeddings[pooling]
        print(f"[{pooling}] embeddings extracted "
              f"(train={len(train_emb['embedding'])}, validation={len(test_emb['embedding'])})",
              flush=True)

        train_emb.update(train_labels)
        test_emb.update(test_labels)
        results = evaluate_probe(train_emb, test_emb, train_emb)
        results["thresholds"] = {k: round(v, 8) for k, v in thresholds.items()}

        path = out_dir / f"linear_probe_{pooling}.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"Linear probe results written to {path}")
        print(json.dumps(results, indent=2))
    del train_embeddings, test_embeddings


if __name__ == "__main__":
    main()
