"""Label and task definitions for the baseline evaluation harness.

Feature indices map to ``configs/market_state_schema_v1.json`` ``feature_order``:
open, high, low, close, volume, funding_rate, open_interest, then eight
calendar fields (indices 7-14).

Downstream tasks:
  - ``future_return``     direction of the return ``horizon_min`` minutes after
                          the window end (up = 1, down = 0)
  - ``volatility``        window log-return std above the fit-split median
  - ``range_expansion``   mean (high-low)/close above the fit-split median
  - ``liquidity``         mean volume above the fit-split median
"""

from typing import Dict

import numpy as np

F_OPEN, F_HIGH, F_LOW, F_CLOSE, F_VOLUME = 0, 1, 2, 3, 4
MINUTE_MS = 60_000

TASKS = ["future_return", "volatility", "range_expansion", "liquidity"]
THRESHOLD_TASKS = ["volatility", "range_expansion", "liquidity"]


def _log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(close, 1e-12)))


def window_stats(features: np.ndarray, style: str = "raw") -> Dict[str, float]:
    """Per-window scalar statistics derived from window features [T, 15].

    ``raw`` reads the OHLCV price/volume columns; ``returns`` derives the same
    quantities from the stationary return-style layout (index 0 = log return,
    1 = hl_range, 3 = log volume) so labels stay comparable across feature styles.
    """
    if style == "returns":
        lr = features[:, 0]
        if len(lr) == 0:
            return {
                "volatility": 0.0,
                "range": 0.0,
                "volume": 0.0,
                "last_return": 0.0,
                "abs_return_mean": 0.0,
                "up_ratio": 0.5,
                "max_abs_return": 0.0,
                "return_skew": 0.0,
            }
        return {
            "volatility": float(np.std(lr)),
            "range": float(np.mean(features[:, 1])),
            "volume": float(np.mean(np.expm1(np.clip(features[:, 3], 0.0, None)))),
            "last_return": float(lr[-1]),
            "abs_return_mean": float(np.mean(np.abs(lr))),
            "up_ratio": float(np.mean(lr > 0)),
            "max_abs_return": float(np.max(np.abs(lr))),
            "return_skew": float(np.mean((lr - lr.mean()) ** 3) / max(lr.std() ** 3, 1e-12)),
        }
    close = features[:, F_CLOSE]
    high = features[:, F_HIGH]
    low = features[:, F_LOW]
    volume = features[:, F_VOLUME]
    lr = _log_returns(close)
    if len(lr) == 0:
        return {
            "volatility": 0.0,
            "range": 0.0,
            "volume": 0.0,
            "last_return": 0.0,
            "abs_return_mean": 0.0,
            "up_ratio": 0.5,
            "max_abs_return": 0.0,
            "return_skew": 0.0,
        }
    return {
        "volatility": float(np.std(lr)),
        "range": float(np.mean((high - low) / np.maximum(close, 1e-12))),
        "volume": float(np.mean(volume)),
        "last_return": float(lr[-1]),
        "abs_return_mean": float(np.mean(np.abs(lr))),
        "up_ratio": float(np.mean(lr > 0)),
        "max_abs_return": float(np.max(np.abs(lr))),
        "return_skew": float(np.mean((lr - lr.mean()) ** 3) / max(lr.std() ** 3, 1e-12)),
    }


def handcrafted_vector(stats: Dict[str, float]) -> np.ndarray:
    """Fixed 8-dim feature vector for the handcrafted linear baseline."""
    return np.array(
        [
            stats["volatility"],
            stats["range"],
            np.log1p(stats["volume"]),
            stats["abs_return_mean"],
            stats["last_return"],
            stats["up_ratio"],
            stats["max_abs_return"],
            stats["return_skew"],
        ],
        dtype=np.float64,
    )


def future_return_label(close_now: float, close_future: float) -> int:
    if close_now <= 0.0 or close_future <= 0.0:
        return 0
    return int(np.log(close_future / close_now) > 0.0)


def binarize(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values > threshold).astype(np.int64)
