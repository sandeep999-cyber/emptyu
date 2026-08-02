import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines.tasks import (
    MINUTE_MS,
    THRESHOLD_TASKS,
    binarize,
    future_return_label,
    handcrafted_vector,
    window_stats,
)
from src.evaluation.baselines.models import (
    LogisticBaseline,
    MajorityBaseline,
    RandomProjectionBaseline,
)
from src.evaluation.baselines.runner import annotate, windows_from_df

START_TS = 1704067200000


def _synthetic_df(n: int, seed: int = 0, vol: float = 0.001, autocorr: float = 0.0, start_ts: int = START_TS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, vol, n)
    for i in range(1, n):
        rets[i] += autocorr * rets[i - 1]
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, vol, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, vol, n)))
    timestamps = start_ts + np.arange(n) * 60000
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.random(n) * 10.0 + 1.0,
        }
    )


class TestTasks:
    def test_future_return_label_sign(self):
        assert future_return_label(100.0, 101.0) == 1
        assert future_return_label(101.0, 100.0) == 0
        assert future_return_label(100.0, 100.0) == 0

    def test_window_stats_no_nan(self):
        df = _synthetic_df(100)
        feats = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        stats = window_stats(feats)
        for v in stats.values():
            assert np.isfinite(v)

    def test_window_stats_returns_style(self):
        rng = np.random.default_rng(0)
        T = 64
        log_ret = rng.normal(0.0, 0.01, T)
        hl_range = np.abs(rng.normal(0.0, 0.002, T))
        log_vol = np.log1p(np.abs(rng.normal(5.0, 1.0, T)))
        feats = np.zeros((T, 15), dtype=np.float32)
        feats[:, 0] = log_ret
        feats[:, 1] = hl_range
        feats[:, 3] = log_vol
        stats = window_stats(feats, style="returns")
        assert abs(stats["volatility"] - float(np.std(log_ret))) < 1e-6
        assert abs(stats["range"] - float(np.mean(hl_range))) < 1e-6
        assert abs(stats["volume"] - float(np.mean(np.expm1(log_vol)))) < 1e-4
        assert abs(stats["last_return"] - float(log_ret[-1])) < 1e-6
        assert stats["up_ratio"] == float(np.mean(log_ret > 0))

    def test_handcrafted_vector_expected_length(self):
        stats = window_stats(np.random.randn(64, 15))
        assert handcrafted_vector(stats).shape == (8,)

    def test_binarize_threshold(self):
        assert np.array_equal(binarize(np.array([1.0, 2.0, 0.5]), 1.0), np.array([0, 1, 0]))


class TestAnnotate:
    def test_labels_use_future_close(self):
        df = _synthetic_df(2000)
        wins = windows_from_df(df, "SYM", "futures", seq_len=64, stride=8, max_windows=None, seed=0)
        ann = annotate(df, wins, horizon_ms=15 * MINUTE_MS, budget_end_ts=None)
        assert len(ann["windows"]) > 0
        for i, w in enumerate(ann["windows"]):
            end = int(w["timestamps"][-1])
            target = end + 15 * MINUTE_MS
            idx_now = int(np.searchsorted(df["timestamp"].to_numpy(), end, side="right") - 1)
            idx_fut = int(np.searchsorted(df["timestamp"].to_numpy(), target))
            expected = future_return_label(float(df["close"].iloc[idx_now]), float(df["close"].iloc[idx_fut]))
            assert ann["future_return"][i] == expected

    def test_fit_budget_guarantees_no_eval_label_use(self):
        df = _synthetic_df(4000)
        fit_end = START_TS + 2000 * 60000
        eval_start = fit_end
        fit_df = df[df["timestamp"] < fit_end]
        eval_df = df[df["timestamp"] >= eval_start]

        fit_wins = windows_from_df(fit_df, "SYM", "futures", seq_len=64, stride=8, max_windows=None, seed=0)
        fit_ann = annotate(fit_df, fit_wins, horizon_ms=15 * MINUTE_MS, budget_end_ts=fit_end)
        eval_wins = windows_from_df(eval_df, "SYM", "futures", seq_len=64, stride=8, max_windows=None, seed=0)
        eval_ann = annotate(eval_df, eval_wins, horizon_ms=15 * MINUTE_MS, budget_end_ts=None)

        assert len(fit_ann["windows"]) > 0 and len(eval_ann["windows"]) > 0
        assert fit_ann["end_ts"].max() + 15 * MINUTE_MS <= fit_end
        assert fit_ann["end_ts"].max() < eval_ann["end_ts"].min()
        assert eval_ann["end_ts"].min() >= eval_start

    def test_budget_drops_windows_without_label_resolution(self):
        df = _synthetic_df(500)
        wins = windows_from_df(df, "SYM", "futures", seq_len=64, stride=1, max_windows=None, seed=0)
        horizon = 15 * MINUTE_MS
        all_ann = annotate(df, wins, horizon_ms=horizon, budget_end_ts=None)
        budget = all_ann["end_ts"].max() + horizon - MINUTE_MS
        tight = annotate(df, wins, horizon_ms=horizon, budget_end_ts=int(budget))
        assert len(tight["windows"]) < len(all_ann["windows"])


class TestModels:
    def test_majority_baseline(self):
        y = np.array([1, 1, 1, 0, 1, 1])
        m = MajorityBaseline().fit(None, y)
        assert m.cls == 1
        assert np.all(m.predict(np.empty((6, 0))) == 1)
        assert np.allclose(m.predict_proba(np.empty((6, 0)))[:, 1], 5 / 6)

    def test_logistic_baseline_separable(self):
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(-3, 1, (200, 4)), rng.normal(3, 1, (200, 4))])
        y = np.array([0] * 200 + [1] * 200)
        Xe = np.concatenate([rng.normal(-3, 1, (100, 4)), rng.normal(3, 1, (100, 4))])
        ye = np.array([0] * 100 + [1] * 100)
        model = LogisticBaseline().fit(X, y)
        acc = np.mean(model.predict(Xe) == ye)
        assert acc > 0.95

    def test_random_projection_baseline_preserves_signal(self):
        rng = np.random.default_rng(1)
        X = np.concatenate([rng.normal(-2, 1, (150, 32)), rng.normal(2, 1, (150, 32))])
        y = np.array([0] * 150 + [1] * 150)
        Xe = np.concatenate([rng.normal(-2, 1, (80, 32)), rng.normal(2, 1, (80, 32))])
        ye = np.array([0] * 80 + [1] * 80)
        model = RandomProjectionBaseline().fit(X, y)
        acc = np.mean(model.predict(Xe) == ye)
        assert acc > 0.9


class TestEndToEnd:
    def _volatility_sets(self, n_fit=400, n_eval=400, seq_len=64, stride=4):
        hi = _synthetic_df(n_fit * 2, seed=1, vol=0.005)
        lo = _synthetic_df(n_fit * 2, seed=2, vol=0.0003, start_ts=hi["timestamp"].iloc[-1] + 60000)
        fit_df = pd.concat([hi, lo]).reset_index(drop=True)
        hi_e = _synthetic_df(n_eval * 2, seed=3, vol=0.005, start_ts=fit_df["timestamp"].iloc[-1] + 60000)
        lo_e = _synthetic_df(n_eval * 2, seed=4, vol=0.0003, start_ts=hi_e["timestamp"].iloc[-1] + 60000)
        eval_df = pd.concat([hi_e, lo_e]).reset_index(drop=True)

        fit_wins = windows_from_df(fit_df, "SYM", "futures", seq_len=seq_len, stride=stride, max_windows=None, seed=0)
        eval_wins = windows_from_df(eval_df, "SYM", "futures", seq_len=seq_len, stride=stride, max_windows=None, seed=0)
        fit = annotate(fit_df, fit_wins, horizon_ms=15 * MINUTE_MS, budget_end_ts=None)
        eval_data = annotate(eval_df, eval_wins, horizon_ms=15 * MINUTE_MS, budget_end_ts=None)
        return fit, eval_data

    def test_handcrafted_separates_volatility_regime(self):
        fit, eval_data = self._volatility_sets()
        threshold = float(np.median(fit["stat_values"]["volatility"]))
        y_fit = binarize(fit["stat_values"]["volatility"], threshold)
        y_eval = binarize(eval_data["stat_values"]["volatility"], threshold)
        assert 0.2 < y_fit.mean() < 0.8
        model = LogisticBaseline().fit(fit["stats"], y_fit)
        from sklearn.metrics import balanced_accuracy_score

        bacc = balanced_accuracy_score(y_eval, model.predict(eval_data["stats"]))
        assert bacc > 0.85

    def test_persistence_beats_chance_on_momentum(self):
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0, 0.002, 3000)
        for i in range(1, len(rets)):
            rets[i] += 0.5 * rets[i - 1]
        close = 100.0 * np.exp(np.cumsum(rets))
        timestamps = START_TS + np.arange(len(close)) * 60000
        df = pd.DataFrame({"timestamp": timestamps, "close": close})
        wins = windows_from_df(df, "SYM", "futures", seq_len=64, stride=8, max_windows=None, seed=0)
        ann = annotate(df, wins, horizon_ms=5 * MINUTE_MS, budget_end_ts=None)
        pred = (ann["last_returns"] > 0).astype(int)
        from sklearn.metrics import balanced_accuracy_score

        bacc = balanced_accuracy_score(ann["future_return"], pred)
        assert bacc > 0.52
