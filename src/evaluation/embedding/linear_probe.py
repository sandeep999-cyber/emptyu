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
    from src.evaluation.embedding._common import build_split_dataset, extract_split_embeddings

    dataset = build_split_dataset(split, trainer_cfg, max_windows)
    style = trainer_cfg.get("feature_style", "raw")
    stats = [window_stats(w["features"], style=style) for w in dataset.windows]
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
    test_emb, test_stats = _extract_with_labels(
        model, normalizer, "validation", args.pooling, trainer_cfg, device, args.max_windows)

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
