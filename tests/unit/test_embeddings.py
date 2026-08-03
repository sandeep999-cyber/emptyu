import pytest
import numpy as np
import torch

from src.models.teacher.embeddings import pool_cls, pool_mean, AttentionPooling, extract_embeddings
from src.evaluation.embedding.clustering import evaluate_clustering
from src.evaluation.embedding._common import _resolve_run_dir
from src.evaluation.embedding.linear_probe import _vectorized_window_stats, window_stats
from src.models.teacher.encoder import TeacherEncoder
from src.data.market_dataset import MarketDataset
from src.training.dataloader import create_dataloader


MODEL_CFG = {
    "context_length": 128,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.0,
    "feature_dim": 15,
    "rope_theta": 10000.0,
    "loss": {
        "masked_modeling": {
            "mask_ratio": 0.15,
            "price_indices": [0, 1, 2, 3, 4],
            "funding_oi_indices": [5, 6],
            "calendar": {},
        }
    },
}

SEQ_LEN = 128
FEATURE_DIM = 15
BASE_TS = 1704067200000  # 2024-01-01 00:00:00 UTC


def _make_window(symbol: str, start_minute: int, contiguous: bool = True) -> dict:
    """Build a window dict matching the WindowingEngine output contract."""
    rng = np.random.default_rng(start_minute)
    ts = BASE_TS + (start_minute + np.arange(SEQ_LEN)) * 60000
    if not contiguous:
        ts = ts.copy()
        ts[SEQ_LEN // 2 :] += 600000  # introduce a 10-minute gap
    return {
        "features": rng.normal(0, 1, (SEQ_LEN, FEATURE_DIM)).astype(np.float32),
        "feature_mask": np.ones((SEQ_LEN, FEATURE_DIM), dtype=bool),
        "timestamps": ts.astype(np.int64),
        "mask": np.ones(SEQ_LEN, dtype=bool),
        "metadata": {
            "symbol": symbol,
            "snapshot_id": "2026-07-30",
            "market": "futures",
            "window_start_ms": int(ts[0]),
            "window_end_ms": int(ts[-1]),
        },
    }


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return TeacherEncoder(MODEL_CFG)


@pytest.fixture
def dataset():
    windows = [_make_window("BTCUSDT", i * SEQ_LEN) for i in range(6)]
    windows += [_make_window("SOLUSDT", 10000 + i * SEQ_LEN) for i in range(4)]
    return MarketDataset(windows)


class TestPoolingFunctions:
    def test_pool_cls(self):
        x = torch.randn(2, 513, 64)
        out = pool_cls(x)
        assert out.shape == (2, 64)

    def test_pool_mean(self):
        latent = torch.randn(2, 129, 64)  # 128 data + 1 CLS
        kpm = torch.ones(2, 129, dtype=torch.bool)
        out = pool_mean(latent, kpm, t_data=128)
        assert out.shape == (2, 64)

    def test_pool_mean_mask_aware(self):
        latent = torch.randn(2, 129, 64)
        kpm = torch.ones(2, 129, dtype=torch.bool)
        kpm[:, -10:] = False  # last 10 positions are padding
        out = pool_mean(latent, kpm, t_data=128)
        assert out.shape == (2, 64)

    def test_cls_vs_mean_different(self):
        latent = torch.randn(2, 129, 64)
        kpm = torch.ones(2, 129, dtype=torch.bool)
        cls_out = pool_cls(latent)
        mean_out = pool_mean(latent, kpm, 128)
        assert not torch.allclose(cls_out, mean_out)


class TestAttentionPooling:
    def test_output_shape(self):
        ap = AttentionPooling(d_model=64)
        latent = torch.randn(2, 129, 64)
        kpm = torch.ones(2, 129, dtype=torch.bool)
        out = ap(latent, kpm, t_data=128)
        assert out.shape == (2, 64)

    def test_gradient_flow(self):
        ap = AttentionPooling(64)
        latent = torch.randn(2, 65, 64, requires_grad=True)
        kpm = torch.ones(2, 65, dtype=torch.bool)
        out = ap(latent, kpm, t_data=64)
        out.sum().backward()
        assert latent.grad is not None

    def test_deterministic(self):
        ap = AttentionPooling(64)
        latent = torch.randn(2, 33, 64)
        kpm = torch.ones(2, 33, dtype=torch.bool)
        torch.manual_seed(0)
        out1 = ap(latent, kpm, 32)
        torch.manual_seed(0)
        out2 = ap(latent, kpm, 32)
        assert torch.allclose(out1, out2)


class TestExtractEmbeddings:
    """Real end-to-end tests of the extraction API all eval modules depend on.

    Uses the real MarketDataset + create_dataloader path with synthetic
    windows matching the WindowingEngine output contract — no mocks.
    """

    @pytest.mark.parametrize("pooling", ["cls", "mean", "attention"])
    def test_extract_shape_and_keys(self, encoder, dataset, pooling):
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        result = extract_embeddings(encoder, loader, pooling, torch.device("cpu"))
        assert result["embedding"].shape == (10, 64)
        assert set(result.keys()) == {"embedding", "symbols", "window_start_ms", "window_end_ms"}
        assert len(result["symbols"]) == 10
        assert len(result["window_start_ms"]) == 10
        assert len(result["window_end_ms"]) == 10

    def test_symbols_and_timestamps_propagated(self, encoder, dataset):
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        result = extract_embeddings(encoder, loader, "mean", torch.device("cpu"))
        assert result["symbols"] == ["BTCUSDT"] * 6 + ["SOLUSDT"] * 4
        assert result["window_start_ms"][0] == BASE_TS
        assert all(e > s for s, e in zip(result["window_start_ms"], result["window_end_ms"]))

    def test_embeddings_are_finite(self, encoder, dataset):
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        result = extract_embeddings(encoder, loader, "mean", torch.device("cpu"))
        assert np.isfinite(result["embedding"]).all()

    def test_deterministic(self, encoder, dataset):
        loader1 = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        loader2 = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        r1 = extract_embeddings(encoder, loader1, "cls", torch.device("cpu"))
        r2 = extract_embeddings(encoder, loader2, "cls", torch.device("cpu"))
        assert np.allclose(r1["embedding"], r2["embedding"])

    def test_unknown_pooling_raises(self, encoder, dataset):
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, seed=42)
        with pytest.raises(ValueError, match="Unknown pooling"):
            extract_embeddings(encoder, loader, "median", torch.device("cpu"))

    def test_mask_awareness_changes_output(self, encoder, dataset):
        """Windows with invalid tail positions must pool differently than fully-valid ones."""
        valid = _make_window("BTCUSDT", 0)
        masked = _make_window("BTCUSDT", 0)
        masked["mask"][64:] = False  # second half invalid
        ds = MarketDataset([valid, masked])
        loader = create_dataloader(ds, batch_size=2, shuffle=False, seed=42)
        result = extract_embeddings(encoder, loader, "mean", torch.device("cpu"))
        assert not np.allclose(result["embedding"][0], result["embedding"][1])


class TestVectorizedWindowStats:
    """Vectorized batch stats must match the reference per-window stats,
    including windows with NaN/Inf in the log-return column (real data has
    these), and for both raw and returns feature styles."""
    _KEYS = ["volatility", "range", "volume"]

    def test_equivalence_with_non_finite(self):
        rng = np.random.default_rng(0)
        for style in ["raw", "returns"]:
            for _ in range(40):
                n = int(rng.integers(1, 8))
                t = int(rng.integers(2, 600))
                feats = rng.standard_normal((n, t, 15)).astype(np.float32)
                if style == "returns":
                    hit = rng.random((n, t)) < 0.02
                    feats[hit, 0] = np.nan
                    hit = rng.random((n, t)) < 0.01
                    feats[hit, 0] = np.inf
                vec = _vectorized_window_stats(feats, style)
                ref = {
                    k: [window_stats(feats[i], style)[k] for i in range(n)]
                    for k in self._KEYS
                }
                for k in self._KEYS:
                    assert np.allclose(vec[k], ref[k], atol=1e-6, equal_nan=True), (style, k)

    def test_all_nonfinite_returns_volatility_is_zero(self):
        feats = np.full((2, 512, 15), np.nan, dtype=np.float32)
        vec = _vectorized_window_stats(feats, "returns")["volatility"]
        ref = [window_stats(feats[i], "returns")["volatility"] for i in range(2)]
        assert np.allclose(vec, ref, atol=1e-6)
        assert vec[0] == 0.0


class TestClustering:
    def test_small_run_full(self):
        rng = np.random.default_rng(1)
        emb = {"embedding": rng.normal(0, 1, (500, 64)).astype(np.float32)}
        res = evaluate_clustering(emb, n_clusters=4)
        assert res["n_samples"] == 500
        assert res["n_silhouette_samples"] == 500
        assert -1.0 <= res["silhouette"] <= 1.0
        assert -1.0 <= res["ami_vs_norm_regime"] <= 1.0
        assert len(res["cluster_sizes"]) == 4

    def test_large_run_subsamples_silhouette(self):
        rng = np.random.default_rng(2)
        emb = {"embedding": rng.normal(0, 1, (20000, 16)).astype(np.float32)}
        res = evaluate_clustering(emb, n_clusters=8, max_silhouette_samples=2000)
        assert res["n_samples"] == 20000
        assert res["n_silhouette_samples"] == 2000
        assert res["silhouette"] == res["silhouette"]  # not NaN
        assert len(res["cluster_sizes"]) == 8


class TestResolveRunDir:
    def test_returns_valid_run_as_is(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "manifest.json").write_text("{}")
        assert _resolve_run_dir(run, base=tmp_path) == run

    def test_falls_back_to_latest_run(self, tmp_path):
        (tmp_path / "20260801_100000_full").mkdir()
        (tmp_path / "20260802_072453_full").mkdir()
        (tmp_path / "20260802_072453_full" / "manifest.json").write_text("{}")
        resolved = _resolve_run_dir(tmp_path / "missing", base=tmp_path)
        assert resolved == tmp_path / "20260802_072453_full"

    def test_returns_input_when_no_runs_exist(self, tmp_path):
        assert _resolve_run_dir(tmp_path / "missing", base=tmp_path) == tmp_path / "missing"
