"""Baseline evaluation harness (Phase A).

Builds causally-separated, labeled windows; fits simple baselines on the fit
split; reports balanced accuracy, plain accuracy, and ROC AUC against a
majority baseline on temporal and cross-symbol eval splits. When a trained
checkpoint is provided, the same windows are embedded and evaluated as an
additional representation.

Protocol guarantees:
  - fit windows end at least ``horizon_min`` before the eval start, so no fit
    label is derived from eval-period data
  - eval windows begin at or after the eval start
  - thresholds, scalers, and classifiers are fit only on the fit split
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.data.lake import lake
from src.data.feature_builder import feature_builder
from src.data.windowing import WindowingEngine
from src.data.market_dataset import MarketDataset
from src.evaluation.baselines.tasks import (
    MINUTE_MS,
    TASKS,
    THRESHOLD_TASKS,
    binarize,
    future_return_label,
    handcrafted_vector,
    window_stats,
)
from src.evaluation.baselines.models import BASELINE_MODELS, MajorityBaseline

DEFAULT_GAP_MS = 300_000


def _ms(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def windows_from_df(
    df: pd.DataFrame,
    symbol: str,
    market: str,
    seq_len: int,
    stride: int,
    max_windows: Optional[int],
    seed: int,
    feature_style: str = "raw",
) -> List[Dict[str, Any]]:
    """Cut aligned market state into windows using the frozen windowing policy."""
    if df.empty or len(df) < seq_len:
        return []
    feats, fm, ts = feature_builder.build_features(df, style=feature_style)
    engine = WindowingEngine(
        {
            "sequence_length": seq_len,
            "stride": max(1, stride),
            "drop_incomplete_windows": True,
            "max_gap_ms": DEFAULT_GAP_MS,
        }
    )
    metadata = {"symbol": symbol, "market": market}
    wins = engine.create_windows(feats, fm, ts, metadata=metadata)
    if max_windows is not None and len(wins) > max_windows:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(wins), size=max_windows, replace=False).tolist())
        wins = [wins[i] for i in idx]
    return wins


def annotate(
    df: pd.DataFrame,
    windows: List[Dict[str, Any]],
    horizon_ms: int,
    budget_end_ts: Optional[int],
    feature_style: str = "raw",
) -> Dict[str, Any]:
    """Attach raw features, stats, and labels to windows from the same aligned frame.

    ``budget_end_ts`` (fit-split guarantee): windows whose horizon label would
    resolve after this timestamp are dropped, so no fit label uses data from the
    eval period.
    """
    ts_arr = df["timestamp"].to_numpy()
    close_arr = df["close"].to_numpy()

    features = []
    stats = []
    last_returns = []
    end_ts_list = []
    future_returns = []
    stat_values = {t: [] for t in THRESHOLD_TASKS}
    kept_windows = []

    for w in windows:
        end = int(w["timestamps"][-1])
        target_ts = end + horizon_ms
        if budget_end_ts is not None and target_ts > budget_end_ts:
            continue
        idx_now = int(np.searchsorted(ts_arr, end, side="right") - 1)
        idx_fut = int(np.searchsorted(ts_arr, target_ts))
        if idx_now < 0 or idx_fut >= len(ts_arr):
            continue
        close_now = float(close_arr[idx_now])
        close_fut = float(close_arr[idx_fut])
        if close_now <= 0.0 or close_fut <= 0.0:
            continue
        st = window_stats(w["features"], style=feature_style)
        features.append(w["features"].reshape(-1))
        stats.append(handcrafted_vector(st))
        last_returns.append(st["last_return"])
        end_ts_list.append(end)
        future_returns.append(future_return_label(close_now, close_fut))
        stat_values["volatility"].append(st["volatility"])
        stat_values["range_expansion"].append(st["range"])
        stat_values["liquidity"].append(st["volume"])
        kept_windows.append(w)

    if not kept_windows:
        return {
            "windows": [], "features": np.empty((0, 0)), "stats": np.empty((0, 0)),
            "last_returns": np.empty(0), "end_ts": np.empty(0, dtype=np.int64),
            "future_return": np.empty(0, dtype=np.int64),
            "stat_values": {t: np.empty(0) for t in THRESHOLD_TASKS},
        }

    return {
        "windows": kept_windows,
        "features": np.stack(features).astype(np.float64),
        "stats": np.stack(stats).astype(np.float64),
        "last_returns": np.asarray(last_returns, dtype=np.float64),
        "end_ts": np.asarray(end_ts_list, dtype=np.int64),
        "future_return": np.asarray(future_returns, dtype=np.int64),
        "stat_values": {t: np.asarray(stat_values[t], dtype=np.float64) for t in THRESHOLD_TASKS},
    }


def build_set(
    symbols: List[str],
    market: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
    seq_len: int,
    stride: int,
    horizon_ms: int,
    max_windows: Optional[int],
    seed: int,
    budget_end_ts: Optional[int],
    feature_style: str = "raw",
) -> Dict[str, Any]:
    """Build a labeled set across symbols over [start_ts, end_ts]."""
    sets = []
    for sym in symbols:
        df = lake.market_state(sym, market=market, start_ts=start_ts, end_ts=end_ts)
        if df.empty:
            continue
        wins = windows_from_df(df, sym, market, seq_len, stride, max_windows, seed, feature_style=feature_style)
        sets.append(annotate(df, wins, horizon_ms, budget_end_ts, feature_style=feature_style))
    if not sets:
        return {
            "windows": [], "features": np.empty((0, 0)), "stats": np.empty((0, 0)),
            "last_returns": np.empty(0), "end_ts": np.empty(0, dtype=np.int64),
            "future_return": np.empty(0, dtype=np.int64),
            "stat_values": {t: np.empty(0) for t in THRESHOLD_TASKS},
        }
    keys = ["windows", "features", "stats", "last_returns", "end_ts", "future_return"]
    merged = {k: _concat_any(k, sets) for k in keys}
    merged["stat_values"] = {t: np.concatenate([s["stat_values"][t] for s in sets]) for t in THRESHOLD_TASKS}
    return merged


def _concat_any(key: str, sets: List[Dict[str, Any]]):
    if key == "windows":
        out = []
        for s in sets:
            out.extend(s["windows"])
        return out
    return np.concatenate([s[key] for s in sets])


def _labels_for(data: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
    labels = {"future_return": data["future_return"]}
    for t in THRESHOLD_TASKS:
        labels[t] = binarize(data["stat_values"][t], thresholds[t])
    return labels


def _score(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "bacc": float("nan"), "acc": float("nan"), "auc": float("nan")}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_pred == y_true))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            auc = float("nan")
    return {"n": n, "bacc": round(bacc, 4), "acc": round(acc, 4), "auc": round(auc, 4)}


def _persistence_score(y_true: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "bacc": float("nan"), "acc": float("nan"), "auc": float("nan")}
    pred = (signal > 0.0).astype(np.int64)
    proba = np.zeros((len(y_true), 2))
    proba[:, 1] = (signal > 0.0).astype(np.float64)
    return _score(y_true, pred, proba)


def run_task(
    fit: Dict[str, Any],
    eval_data: Dict[str, Any],
    task: str,
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    fit_y = _labels_for(fit, thresholds)[task]
    eval_y = _labels_for(eval_data, thresholds)[task]
    results: Dict[str, Dict[str, Any]] = {}

    majority = MajorityBaseline().fit(None, fit_y)
    x_eval = np.empty((len(eval_y), 0))
    results["majority"] = _score(eval_y, majority.predict(x_eval), majority.predict_proba(x_eval))

    if task == "future_return":
        results["persistence"] = _persistence_score(eval_y, eval_data["last_returns"])

    reps = {"raw_linear": fit["features"], "handcrafted_linear": fit["stats"], "random_proj": fit["features"]}
    reps_eval = {"raw_linear": eval_data["features"], "handcrafted_linear": eval_data["stats"], "random_proj": eval_data["features"]}
    for name in ("raw_linear", "handcrafted_linear", "random_proj"):
        model = BASELINE_MODELS[name]()
        if len(fit_y) == 0 or len(np.unique(fit_y)) < 2:
            results[name] = {"n": len(eval_y), "bacc": float("nan"), "acc": float("nan"), "auc": float("nan")}
            continue
        try:
            model.fit(reps[name], fit_y)
            results[name] = _score(eval_y, model.predict(reps_eval[name]), model.predict_proba(reps_eval[name]))
        except ValueError:
            results[name] = {"n": len(eval_y), "bacc": float("nan"), "acc": float("nan"), "auc": float("nan")}
    return results


def extract_embeddings_for_set(
    model: Any,
    normalizer: Any,
    data: Dict[str, Any],
    pooling: str,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Extract pooled embeddings for the exact windows in ``data`` (same order)."""
    if not data["windows"]:
        return np.empty((0, model.d_model))

    from src.models.teacher.embeddings import extract_embeddings
    from src.training.dataloader import create_dataloader

    class _NormDataset(MarketDataset):
        def __init__(self, windows, norm):
            super().__init__(windows)
            self.norm = norm

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            flat = item["features"].reshape(-1, item["features"].shape[-1])
            item["features"] = self.norm.transform(flat).reshape(item["features"].shape)
            return item

    dataset = _NormDataset(data["windows"], normalizer)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    result = extract_embeddings(model, loader, pooling, device)
    return result["embedding"]


def evaluate_embeddings(
    emb_fit: np.ndarray,
    emb_eval: np.ndarray,
    fit: Dict[str, Any],
    eval_data: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for task in TASKS:
        fit_y = _labels_for(fit, thresholds)[task]
        eval_y = _labels_for(eval_data, thresholds)[task]
        if len(fit_y) == 0 or len(np.unique(fit_y)) < 2:
            results[task] = {"n": len(eval_y), "bacc": float("nan"), "acc": float("nan"), "auc": float("nan")}
            continue
        model = BASELINE_MODELS["handcrafted_linear"]()
        model.fit(emb_fit, fit_y)
        results[task] = _score(eval_y, model.predict(emb_eval), model.predict_proba(emb_eval))
    return results


def evaluate(
    fit: Dict[str, Any],
    eval_name: str,
    eval_data: Dict[str, Any],
    model: Any,
    normalizer: Any,
    pooling: str,
    device: Any,
    batch_size: int,
) -> Dict[str, Any]:
    thresholds = {
        t: float(np.median(fit["stat_values"][t])) for t in THRESHOLD_TASKS
    }
    report: Dict[str, Any] = {"split": eval_name, "n_fit": len(fit["windows"]), "n_eval": len(eval_data["windows"]), "thresholds": thresholds, "tasks": {}}
    for task in TASKS:
        report["tasks"][task] = run_task(fit, eval_data, task, thresholds)

    if model is not None:
        emb_fit = extract_embeddings_for_set(model, normalizer, fit, pooling, device, batch_size)
        emb_eval = extract_embeddings_for_set(model, normalizer, eval_data, pooling, device, batch_size)
        report["embedding"] = {"pooling": pooling, "tasks": evaluate_embeddings(emb_fit, emb_eval, fit, eval_data, thresholds)}
    return report


def _load_checkpoint(run_dir: str, device: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    from src.evaluation.embedding._common import load_model_and_normalizer

    return load_model_and_normalizer(Path(run_dir), device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation harness")
    parser.add_argument("--market", default="futures")
    parser.add_argument("--fit-symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--fit-start", default="2024-01-01")
    parser.add_argument("--fit-end", default="2024-11-30", help="Exclusive end of the fit period")
    parser.add_argument("--eval-symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--eval-start", default="2024-12-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--horizon-min", type=int, default=15)
    parser.add_argument("--max-windows", type=int, default=1500, help="Per-symbol window cap (null = unlimited)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Run dir to evaluate frozen embeddings")
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-style", default=None, choices=["raw", "returns"],
                        help="Feature style for window building (defaults to checkpoint config, else 'raw')")
    parser.add_argument("--out", default="evaluation/baselines")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalizer, configs = None, None, None
    feature_style = args.feature_style or "raw"
    if args.checkpoint:
        model, normalizer, configs = _load_checkpoint(args.checkpoint, device)
        seq_len = int(configs["model_config"]["model"]["context_length"])
        if args.seq_len != seq_len:
            print(f"[baselines] Overriding seq-len to model context_length={seq_len}")
        args.seq_len = seq_len
        ckpt_style = configs.get("trainer_config", {}).get("feature_style", "raw")
        if args.feature_style is None:
            feature_style = ckpt_style
            print(f"[baselines] Using checkpoint feature_style={feature_style}")
        elif feature_style != ckpt_style:
            print(f"[baselines] WARNING: --feature-style={feature_style} differs from checkpoint {ckpt_style}")

    fit_start = _ms(args.fit_start)
    fit_end = _ms(args.fit_end)
    eval_start = _ms(args.eval_start)
    eval_end = _ms(args.eval_end)
    horizon_ms = args.horizon_min * MINUTE_MS

    print("[baselines] Building fit set...")
    fit = build_set(
        args.fit_symbols.split(","), args.market, fit_start, fit_end,
        args.seq_len, args.stride, horizon_ms, args.max_windows, args.seed, budget_end_ts=fit_end,
        feature_style=feature_style,
    )
    print(f"[baselines] fit windows: {len(fit['windows'])}")

    print("[baselines] Building eval set...")
    eval_data = build_set(
        args.eval_symbols.split(","), args.market, eval_start, eval_end,
        args.seq_len, args.stride, horizon_ms, args.max_windows, args.seed, budget_end_ts=None,
        feature_style=feature_style,
    )
    print(f"[baselines] eval windows: {len(eval_data['windows'])}")

    if len(fit["windows"]) == 0 or len(eval_data["windows"]) == 0:
        print("[baselines] Empty fit or eval set; nothing to report.")
        return

    report = evaluate(fit, "eval", eval_data, model, normalizer, args.pooling, device, args.batch_size)
    args.feature_style = feature_style
    report["config"] = vars(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[baselines] Report written to {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
