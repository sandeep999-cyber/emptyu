"""Windowing engine cutting feature vectors into fixed-length sequence windows with gap validation."""

from typing import Any, Dict, List, Optional
import numpy as np
from src.config import config


class WindowingEngine:
    """Cuts continuous feature matrices into fixed-length sequence windows while rejecting gap-crossing sequences."""

    def __init__(self, windowing_config: Optional[Dict[str, Any]] = None):
        cfg = windowing_config or config.windowing
        self.seq_len = cfg.get("sequence_length", 512)
        self.stride = cfg.get("stride", 1)
        self.drop_incomplete = cfg.get("drop_incomplete_windows", True)
        self.max_gap_ms = cfg.get("max_gap_ms", 300000)  # 5 minutes max gap tolerance

    def create_windows(
        self,
        features: np.ndarray,
        feature_mask: np.ndarray,
        timestamps: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Cut features array into sliding windows.
        Rejects windows where timestamp delta exceeds max_gap_ms.
        Also rejects windows with duplicate or out-of-order timestamps.
        """
        num_records = len(features)
        windows = []

        if num_records == 0:
            return windows

        if num_records < self.seq_len and self.drop_incomplete:
            return windows

        # Timestamp sanity check on the full array: duplicates or non-monotonic
        if num_records > 1:
            ts_diffs_full = np.diff(timestamps)
            if np.any(ts_diffs_full <= 0):
                non_pos = ts_diffs_full <= 0
                bad_indices = np.where(non_pos)[0] + 1
                raise ValueError(
                    f"Timestamps contain {int(non_pos.sum())} non-positive delta(s) "
                    f"at indices {bad_indices[:10].tolist()} (duplicates or out-of-order). "
                    "Data must be strictly increasing with unique timestamps."
                )

        effective_len = num_records - self.seq_len + 1

        for start_idx in range(0, effective_len, self.stride):
            if not self.drop_incomplete and start_idx + self.seq_len > num_records:
                continue

            end_idx = start_idx + self.seq_len
            ts_win = timestamps[start_idx:end_idx]

            # Timestamp continuity validation
            ts_diffs = np.diff(ts_win)
            max_gap = np.max(ts_diffs) if len(ts_diffs) > 0 else 0
            min_gap = np.min(ts_diffs) if len(ts_diffs) > 0 else 0
            has_gap = np.any(ts_diffs > self.max_gap_ms)

            if has_gap:
                continue

            f_win = features[start_idx:end_idx]
            fm_win = feature_mask[start_idx:end_idx]

            # Mask tracks per-position validity:
            # A position is invalid if ALL features are unobserved
            pos_obs = fm_win.any(axis=1)
            mask_win = pos_obs

            # Positions whose NEXT gap exceeds the expected step are also flagged.
            # Next-gap flag on position i == ts_diffs[i] > expected_step*2; the last
            # position has no next timestamp, so it is never flagged.
            expected_step = int(np.median(ts_diffs)) if len(ts_diffs) > 0 else 60000
            gap_exceeds_step = np.zeros(self.seq_len, dtype=bool)
            gap_exceeds_step[:-1] = ts_diffs > expected_step * 2  # Double the expected step is suspicious
            # (Last position cannot be judged — left False, so it stays observed)
            mask_win = mask_win & ~gap_exceeds_step

            win_meta = {
                **(metadata or {}),
                "window_start_ms": int(ts_win[0]),
                "window_end_ms": int(ts_win[-1]),
                "window_span_ms": int(ts_win[-1] - ts_win[0]),
                "window_max_consecutive_gap_ms": int(max_gap),
                "window_min_consecutive_gap_ms": int(min_gap),
                "window_contiguous": bool(max_gap <= 60000),  # 1-minute step expected for aligned data
            }

            windows.append({
                "features": f_win,
                "feature_mask": fm_win,
                "timestamps": ts_win,
                "mask": mask_win,
                "metadata": win_meta,
            })

        return windows


windowing_engine = WindowingEngine()
