"""Finish pipeline: report, snapshot, scientific validation, benchmark."""
import sys, os, time, hashlib, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.config import config
from src.data.lake import lake
from src.data.manifest_builder import manifest_builder
from src.data.snapshot_manager import snapshot_manager
from src.data.reports import reports_generator
from src.data.feature_builder import feature_builder
from src.data.windowing import windowing_engine
from src.data.market_dataset import MarketDataset
from src.training.benchmark import benchmark_suite

SNAP = "2026-07-30"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MKTS = ["futures", "spot"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- Report ----
log("=== STORAGE REPORT ===")
reports_generator.save_report(
    reports_generator.generate_storage_summary(),
    config.training_dir / "storage_summary.md")
log("  saved")

# ---- Snapshot ----
log("=== SNAPSHOT ===")
sd = config.training_dir / "snapshots" / SNAP
if sd.exists():
    shutil.rmtree(str(sd))
m = manifest_builder.build_manifest_from_index(
    SNAP, train_symbols=["BTCUSDT", "ETHUSDT"],
    val_symbols=["SOLUSDT"], test_symbols=["BTCUSDT"])
snap = snapshot_manager.create_snapshot(SNAP, m, None, {"snapshot": SNAP})
manifest_builder.save_manifest(m, config.training_dir / "training_manifest_v1.json")
fp = manifest_builder.compute_fingerprint(m)
manifest_builder.save_fingerprint(fp, config.training_dir / "dataset_fingerprint.json")
log(f"  Snapshot: {snap}")
log(f"  Fingerprint: {fp['fingerprint']}")
log(f"  Files in ledger: {fp['file_count']}")

# ---- Scientific validation ----
log("=== SCIENTIFIC VALIDATION ===")
issues = []
for mk in MKTS:
    for s in SYMS:
        df = lake.market_state(s, market=mk)
        if df.empty:
            issues.append(f"{s}_{mk}_empty")
            continue
        ts = df["timestamp"]
        mono = ts.is_monotonic_increasing
        dups = int(ts.duplicated().sum())
        if not mono:
            issues.append(f"{s}_{mk}_ts_order")
        if dups:
            issues.append(f"{s}_{mk}_ts_dups")
        # OHLC sanity
        bad_hl = int((df["high"] < df["low"]).sum())
        bad_px = int((df["close"] <= 0).sum())
        if bad_hl:
            issues.append(f"{s}_{mk}_hl")
        if bad_px:
            issues.append(f"{s}_{mk}_close_pos")
        # Modality coverage (futures only for funding/OI)
        cov_info = {}
        for col in ["funding_rate", "open_interest"]:
            if col in df.columns:
                fr = df[col]
                cov = 1 - fr.isna().sum() / max(len(fr), 1)
                fi = fr.first_valid_index()
                nans_after = int(fr.loc[fi:].isna().sum()) if fi is not None else len(fr)
                cov_info[col] = (round(cov * 100, 2), nans_after)
        log(f"  {s} {mk}: mono={mono} dups={dups} high<low={bad_hl} close<=0={bad_px} {cov_info}")

if issues:
    log(f"  ISSUES FOUND: {issues}")
else:
    log("  ALL SCIENTIFIC CHECKS PASSED")

# ---- Benchmark ----
log("=== BENCHMARK ===")
try:
    df = lake.market_state("BTCUSDT", market="futures")
    feat, fm, ts = feature_builder.build_features(df)
    wins = windowing_engine.create_windows(feat, fm, ts, metadata={
        "symbol": "BTCUSDT", "market": "futures",
        "start_ts": int(ts[0]), "end_ts": int(ts[-1]),
        "snapshot_id": SNAP, "modality_config": "modalities_v1.yaml",
        "windowing_config": "windowing_v1.yaml"})
    ds = MarketDataset(wins)
    res = benchmark_suite.benchmark_dataloader(ds, batch_size=32, num_batches=50)
    for k, v in res.items():
        log(f"  {k}: {v}")
except Exception as e:
    log(f"  benchmark error: {e}")

log("=== DONE ===")
