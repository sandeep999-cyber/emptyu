"""Unit tests for Windowing Engine."""

import numpy as np
import pytest
from src.data.windowing import WindowingEngine


def test_windowing_engine():
    engine = WindowingEngine({"sequence_length": 5, "stride": 1, "drop_incomplete_windows": True, "max_gap_ms": 300000})
    features = np.arange(50).reshape(10, 5).astype(np.float32)
    feature_mask = np.ones((10, 5), dtype=bool)
    timestamps = np.arange(10, dtype=np.int64)

    windows = engine.create_windows(features, feature_mask, timestamps)
    assert len(windows) == 6
    assert windows[0]["features"].shape == (5, 5)
    assert windows[0]["timestamps"].shape == (5,)


def test_windowing_rejects_gaps():
    """Windows with gaps > max_gap_ms should be rejected; clean windows kept."""
    engine = WindowingEngine({"sequence_length": 5, "stride": 1, "max_gap_ms": 300000})
    features = np.arange(20).reshape(10, 2).astype(np.float32)
    feature_mask = np.ones((10, 2), dtype=bool)
    # Large gap between index 4 (240000) and index 5 (600000)
    timestamps = np.array([0, 60000, 120000, 180000, 240000, 600000, 660000, 720000, 780000, 840000], dtype=np.int64)

    windows = engine.create_windows(features, feature_mask, timestamps)
    # Only window [0-4] and [5-9] should survive (no internal gap)
    assert len(windows) == 2
    # Window 0 ends at 240000, window 1 starts at 600000 (wraps completely clean side)
    for w in windows:
        assert w["metadata"]["window_contiguous"] is True


def test_windowing_rejects_duplicates():
    """Windows with duplicate timestamps should be rejected."""
    engine = WindowingEngine({"sequence_length": 5, "stride": 1})
    features = np.arange(20).reshape(10, 2).astype(np.float32)
    feature_mask = np.ones((10, 2), dtype=bool)
    timestamps = np.array([0, 60000, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000], dtype=np.int64)

    with pytest.raises(ValueError, match="non-positive delta"):
        engine.create_windows(features, feature_mask, timestamps)


def test_windowing_flags_positions_before_suspicious_gap():
    """Positions whose NEXT gap exceeds ~2x the expected step are masked out."""
    engine = WindowingEngine({"sequence_length": 6, "stride": 1, "max_gap_ms": 600000})
    features = np.arange(12).reshape(6, 2).astype(np.float32)
    feature_mask = np.ones((6, 2), dtype=bool)
    # Expected step 60s; one 300s gap between index 2 and 3 (5x step).
    timestamps = np.array([0, 60000, 120000, 420000, 480000, 540000], dtype=np.int64)

    windows = engine.create_windows(features, feature_mask, timestamps)
    assert len(windows) == 1
    mask = windows[0]["mask"]
    # Index 2 is followed by the big gap -> flagged; the last position cannot be
    # judged but is observed, so it stays valid.
    assert not bool(mask[2])  # followed by the big gap -> flagged
    assert bool(mask[0])
    assert bool(mask[-1])  # last position cannot be judged but is observed
