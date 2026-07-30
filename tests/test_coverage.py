"""Coverage tests for unlisted modules (pipeline_manifest, sampler, dataloader, benchmark)."""

import tempfile
from pathlib import Path
import json
import pytest
import torch
import numpy as np
from src.data.pipeline_manifest import PipelineManifest
from src.data.market_dataset import MarketDataset
from src.training.sampler import EpochMarketSampler
from src.training.benchmark import BenchmarkSuite
from src.training.dataloader import create_dataloader


class TestPipelineManifest:
    def test_record_and_save_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            pm = PipelineManifest(manifest_path=path)
            pm.record_stage("download", ["input1"], ["output1"], ["checksum1"], 1.5, {"key": "val"}, timestamp="2026-07-30T00:00:00")
            pm.save()
            assert path.exists()
            with open(path, "r") as f:
                data = json.load(f)
            assert "download" in data
            assert len(data["download"]) == 1
            assert data["download"][0]["duration_seconds"] == 1.5

    def test_stage_appends(self):
        pm = PipelineManifest(manifest_path=Path(tempfile.mktemp(suffix=".json")))
        pm.record_stage("download", [], [], [], 1.0)
        pm.record_stage("download", [], [], [], 2.0)
        assert len(pm.stages["download"]) == 2

    def test_unknown_stage_is_created(self):
        pm = PipelineManifest(manifest_path=Path(tempfile.mktemp(suffix=".json")))
        pm.record_stage("unknown_stage", [], [], [], 1.0)
        assert "unknown_stage" in pm.stages


class TestSampler:
    def test_sampler_epoch_determinism(self):
        s1 = EpochMarketSampler(100, shuffle=True, seed=42)
        s2 = EpochMarketSampler(100, shuffle=True, seed=42)
        indices_1 = list(s1.__iter__())
        indices_2 = list(s2.__iter__())
        assert indices_1 == indices_2

    def test_sampler_no_shuffle(self):
        s = EpochMarketSampler(10, shuffle=False)
        indices = list(s.__iter__())
        assert indices == list(range(10))

    def test_sampler_epoch_changes_order(self):
        s = EpochMarketSampler(100, shuffle=True, seed=42)
        epoch_0 = list(s.__iter__())
        s.set_epoch(1)
        epoch_1 = list(s.__iter__())
        assert epoch_0 != epoch_1


class TestBenchmark:
    def test_benchmark_empty_dataset(self):
        ds = MarketDataset([])
        bm = BenchmarkSuite()
        res = bm.benchmark_dataloader(ds, batch_size=32, num_batches=10)
        assert res["samples_per_sec"] == 0.0

    def test_benchmark_with_data(self):
        win = {
            "features": np.zeros((512, 15), dtype=np.float32),
            "feature_mask": np.ones((512, 15), dtype=bool),
            "timestamps": np.arange(512, dtype=np.int64),
            "mask": np.ones((512,), dtype=bool),
            "metadata": {"symbol": "BTCUSDT"}
        }
        ds = MarketDataset([win] * 10)
        bm = BenchmarkSuite()
        res = bm.benchmark_dataloader(ds, batch_size=4, num_batches=2)
        assert res["samples_per_sec"] > 0
        assert "peak_ram_mb" in res
        assert "cpu_percent" in res


class TestDataloader:
    def test_create_dataloader_no_shuffle(self):
        win = {
            "features": np.zeros((512, 15), dtype=np.float32),
            "feature_mask": np.ones((512, 15), dtype=bool),
            "timestamps": np.arange(512, dtype=np.int64),
            "mask": np.ones((512,), dtype=bool),
            "metadata": {"symbol": "BTCUSDT"}
        }
        ds = MarketDataset([win] * 20)
        dl = create_dataloader(ds, batch_size=4, shuffle=False)
        batch = next(iter(dl))
        assert batch["features"].shape == (4, 512, 15)

    def test_create_dataloader_shuffle(self):
        win = {
            "features": np.zeros((512, 15), dtype=np.float32),
            "feature_mask": np.ones((512, 15), dtype=bool),
            "timestamps": np.arange(512, dtype=np.int64),
            "mask": np.ones((512,), dtype=bool),
            "metadata": {"symbol": "BTCUSDT"}
        }
        ds = MarketDataset([win] * 20)
        dl = create_dataloader(ds, batch_size=4, shuffle=True, seed=42)
        batch = next(iter(dl))
        assert batch["features"].shape == (4, 512, 15)
