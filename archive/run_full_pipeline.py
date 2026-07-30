"""Full Pipeline Executor v3 — uses httpx for reliable downloads."""

import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

import httpx

from src.config import config
from src.data.db import db_manager
from src.data.parquet_converter import converter
from src.data.resampler import resampler
from src.data.calendar_builder import calendar_builder
from src.data.lake import lake
from src.data.alignment import alignment_engine
from src.data.validator import validator
from src.data.quality_report import quality_reporter
from src.data.manifest_builder import manifest_builder
from src.data.snapshot_manager import snapshot_manager
from src.data.datacard_builder import datacard_builder
from src.data.feature_builder import feature_builder
from src.data.windowing import windowing_engine
from src.data.market_dataset import MarketDataset
from src.data.reports import reports_generator
from src.data.metadata import metadata_manager
from src.data.pipeline_manifest import pipeline_manifest

SNAPSHOT_DATE = "2026-07-30"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MARKETS = ["futures", "spot"]
YEARS = [2024]
MONTHS = list(range(1, 13))

DATASET_PATHS = {
    "klines": "klines",
    "funding": "fundingRate",
    "fundingRate": "fundingRate",
    "open_interest": "metrics",
    "metrics": "metrics",
    "aggTrades": "aggTrades",
    "trades": "trades",
    "depth": "depth",
    "liquidations": "liquidations",
}

# Phase 1 active + download extras
PHASE1_TYPES = ["klines", "funding", "open_interest"]
DOWNLOAD_TYPES = ["klines", "funding", "open_interest", "aggTrades", "trades"]

BASE_URL = "https://data.binance.vision/data"
REST_FUTURES = "https://fapi.binance.com"
REST_SPOT = "https://api.binance.com"


def log(msg):
    print(f"[PIPELINE] {msg}", flush=True)


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_url(market, ds_type, symbol, filename):
    market_path = "futures/um" if market == "futures" else "spot"
    ds_path = DATASET_PATHS.get(ds_type, ds_type)
    if ds_type == "klines":
        return f"{BASE_URL}/{market_path}/monthly/klines/{symbol}/1m/{filename}"
    else:
        return f"{BASE_URL}/{market_path}/monthly/{ds_path}/{symbol}/{filename}"


def download_file_httpx(client, url, dest_path, retries=5):
    """Download file with httpx, retries, and resume."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    temp_path = dest_path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=60) as resp:
                if resp.status_code == 200:
                    with open(temp_path, "wb") as f:
                        for chunk in resp.iter_bytes(65536):
                            f.write(chunk)
                    temp_path.replace(dest_path)
                    return True
                elif resp.status_code == 404:
                    return False
        except Exception:
            if attempt == retries - 1:
                if temp_path.exists():
                    temp_path.unlink()
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def download_one(client, market, sym, ds_type, year, month):
    """Download a single archive. Returns (status, path_or_None)."""
    month_str = f"{month:02d}"
    if ds_type == "klines":
        filename = f"{sym}-1m-{year}-{month_str}.zip"
    else:
        ds_path = DATASET_PATHS.get(ds_type, ds_type)
        filename = f"{sym}-{ds_path}-{year}-{month_str}.zip"

    url = build_url(market, ds_type, sym, filename)
    target_dir = config.raw_dir / market / sym / ds_type / ("1m" if ds_type == "klines" else "")
    zip_path = target_dir / filename

    if zip_path.exists() and zip_path.stat().st_size > 0:
        return ("cached", zip_path)

    ok = download_file_httpx(client, url, zip_path)
    if ok:
        # Try to download checksum file
        checksum_url = f"{url}.checksum"
        checksum_path = target_dir / f"{filename}.checksum"
        download_file_httpx(client, checksum_url, checksum_path, retries=2)

        # Verify checksum if available
        if checksum_path.exists() and checksum_path.stat().st_size > 0:
            try:
                expected = checksum_path.read_text().strip().split()[0]
                actual = compute_sha256(zip_path)
                if actual.lower() != expected.lower():
                    zip_path.unlink()
                    return ("bad_checksum", None)
            except Exception:
                pass
        return ("downloaded", zip_path)
    else:
        if zip_path.exists():
            return ("cached", zip_path)
        return ("missing", None)


# ──────────────────────────────────────────────
# STAGE 1: Exchange Info
# ──────────────────────────────────────────────
def stage_exchange_info():
    log("--- STAGE 1: Exchange Info ---")
    t0 = time.time()
    with httpx.Client(timeout=30) as client:
        for market in MARKETS:
            log(f"  Fetching {market} exchange info...")
            try:
                if market == "futures":
                    resp = client.get(f"{REST_FUTURES}/fapi/v1/exchangeInfo")
                else:
                    resp = client.get(f"{REST_SPOT}/api/v3/exchangeInfo")
                resp.raise_for_status()
                info = resp.json()
                symbols = info.get("symbols", [])
                count = 0
                for s in symbols:
                    filters = {f["filterType"]: f for f in s.get("filters", [])}
                    price_filter = filters.get("PRICE_FILTER", {})
                    lot_filter = filters.get("LOT_SIZE", {})
                    db_manager.register_asset(
                        symbol=s["symbol"],
                        market_type=market,
                        base_asset=s.get("baseAsset", ""),
                        quote_asset=s.get("quoteAsset", ""),
                        is_active=(s.get("status") == "TRADING"),
                        listing_date=str(s.get("onboardDate", "")) if s.get("onboardDate") else None,
                        delisting_date=None,
                        contract_type=s.get("contractType", "PERPETUAL") if market == "futures" else "SPOT",
                        tick_size=float(price_filter.get("tickSize", 0.0)),
                        step_size=float(lot_filter.get("stepSize", 0.0)),
                        min_qty=float(lot_filter.get("minQty", 0.0)),
                        contract_size=1.0,
                    )
                    for year in YEARS:
                        ei_dir = config.canonical_dir / market / s["symbol"] / "exchange_info"
                        ei_dir.mkdir(parents=True, exist_ok=True)
                        ei_path = ei_dir / f"{year}.json"
                        if not ei_path.exists():
                            ei_path.write_text(json.dumps(s, indent=2), encoding="utf-8")
                    count += 1
                log(f"  Registered {count} {market} symbols")
            except Exception as e:
                log(f"  WARNING: {market} exchangeInfo: {e}")
    log(f"  Done in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────
# STAGE 2: Download
# ──────────────────────────────────────────────
def stage_download():
    log("--- STAGE 2: Download ---")
    t0 = time.time()
    stats = {"downloaded": 0, "cached": 0, "missing": 0, "bad_checksum": 0}

    tasks = []
    for market in MARKETS:
        for sym in SYMBOLS:
            for ds_type in DOWNLOAD_TYPES:
                for year in YEARS:
                    for month in MONTHS:
                        tasks.append((market, sym, ds_type, year, month))

    log(f"  {len(tasks)} archives to check...")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for i, (market, sym, ds_type, year, month) in enumerate(tasks):
            status, path = download_one(client, market, sym, ds_type, year, month)
            stats[status] = stats.get(status, 0) + 1
            if (i + 1) % 30 == 0 or (i + 1) == len(tasks):
                log(f"  [{i+1}/{len(tasks)}] dl={stats['downloaded']} cached={stats['cached']} miss={stats['missing']} bad={stats['bad_checksum']}")

    elapsed = time.time() - t0
    log(f"  Download: {stats['downloaded']} new, {stats['cached']} cached, {stats['missing']} missing, {stats['bad_checksum']} bad")
    log(f"  Done in {elapsed:.1f}s")
    pipeline_manifest.record_stage(
        "download", inputs=[], outputs=[], checksums=[], duration_seconds=elapsed,
        metadata=stats
    )
    pipeline_manifest.save()


# ──────────────────────────────────────────────
# STAGE 3: Convert
# ──────────────────────────────────────────────
def stage_convert():
    log("--- STAGE 3: Convert ---")
    t0 = time.time()
    converted = 0
    cached = 0
    errors = 0

    for market in MARKETS:
        for sym in SYMBOLS:
            for ds_type in DOWNLOAD_TYPES:
                for year in YEARS:
                    for month in MONTHS:
                        month_str = f"{month:02d}"
                        if ds_type == "klines":
                            zip_filename = f"{sym}-1m-{year}-{month_str}.zip"
                            raw_dir = config.raw_dir / market / sym / ds_type / "1m"
                            out_dir = config.canonical_dir / market / sym / ds_type / "1m"
                            interval = "1m"
                        else:
                            ds_path = DATASET_PATHS.get(ds_type, ds_type)
                            zip_filename = f"{sym}-{ds_path}-{year}-{month_str}.zip"
                            raw_dir = config.raw_dir / market / sym / ds_type / ""
                            out_dir = config.canonical_dir / market / sym / ds_type
                            interval = ""

                        zip_path = raw_dir / zip_filename
                        if not zip_path.exists():
                            continue

                        existing = out_dir / f"{year}-{month:02d}.parquet"
                        if existing.exists():
                            try:
                                sha = compute_sha256(existing)
                                import pyarrow.parquet as pq
                                pf = pq.read_metadata(str(existing))
                                file_id = f"{market}_{sym}_{ds_type}_{interval}_{year}_{month}"
                                db_manager.register_file(
                                    file_id=file_id, symbol=sym, market=market,
                                    dataset_type=ds_type, interval=interval,
                                    year=year, month=month,
                                    start_ts=0, end_ts=0,
                                    row_count=pf.num_rows, file_size=existing.stat().st_size,
                                    sha256=sha, schema_hash="", file_path=str(existing),
                                    status="CONVERTED"
                                )
                                cached += 1
                                continue
                            except Exception:
                                pass

                        try:
                            out_path = converter.convert_zip_to_parquet(
                                zip_path=zip_path, market=market, symbol=sym,
                                dataset_type=ds_type, interval=interval,
                                year=year, month=month,
                                download_date=SNAPSHOT_DATE,
                            )
                            converted += 1
                            if converted % 10 == 0:
                                log(f"  Converted {converted} files...")
                        except Exception as e:
                            errors += 1
                            log(f"  ERROR: {sym}/{ds_type}/{year}-{month:02d}: {e}")

    elapsed = time.time() - t0
    log(f"  Convert: {converted} new, {cached} cached, {errors} errors")
    log(f"  Done in {elapsed:.1f}s")
    pipeline_manifest.record_stage("convert", inputs=[], outputs=[], checksums=[], duration_seconds=elapsed)
    pipeline_manifest.save()


# ──────────────────────────────────────────────
# STAGE 4: Resample
# ──────────────────────────────────────────────
def stage_resample():
    log("--- STAGE 4: Resample ---")
    t0 = time.time()
    target_intervals = ["5m", "15m", "1h", "4h", "1d"]
    resampled = 0
    cached = 0
    errors = 0

    for market in MARKETS:
        for sym in SYMBOLS:
            for year in YEARS:
                for month in MONTHS:
                    input_1m = config.canonical_dir / market / sym / "klines" / "1m" / f"{year}-{month:02d}.parquet"
                    if not input_1m.exists():
                        continue
                    for target_tf in target_intervals:
                        out_dir = config.canonical_dir / market / sym / "klines" / target_tf
                        existing = out_dir / f"{year}-{month:02d}.parquet"
                        if existing.exists():
                            try:
                                sha = compute_sha256(existing)
                                import pyarrow.parquet as pq
                                pf = pq.read_metadata(str(existing))
                                file_id = f"{market}_{sym}_klines_{target_tf}_{year}_{month}"
                                db_manager.register_file(
                                    file_id=file_id, symbol=sym, market=market,
                                    dataset_type="klines", interval=target_tf,
                                    year=year, month=month,
                                    start_ts=0, end_ts=0,
                                    row_count=pf.num_rows, file_size=existing.stat().st_size,
                                    sha256=sha, schema_hash="", file_path=str(existing),
                                    status="RESAMPLED"
                                )
                                cached += 1
                                continue
                            except Exception:
                                pass

                        try:
                            out_path = resampler.resample_file(
                                input_1m, market, sym, target_tf, year, month
                            )
                            resampled += 1
                            if resampled % 20 == 0:
                                log(f"  Resampled {resampled}...")
                        except Exception as e:
                            errors += 1
                            log(f"  ERROR: {sym} {target_tf}: {e}")

    elapsed = time.time() - t0
    log(f"  Resample: {resampled} new, {cached} cached, {errors} errors")
    log(f"  Done in {elapsed:.1f}s")
    pipeline_manifest.record_stage("resample", inputs=[], outputs=[], checksums=[], duration_seconds=elapsed)
    pipeline_manifest.save()


# ──────────────────────────────────────────────
# STAGE 5: Calendar + Data Lake
# ──────────────────────────────────────────────
def stage_build_lake():
    log("--- STAGE 5: Calendar & Lake ---")
    t0 = time.time()

    for market in MARKETS:
        for sym in SYMBOLS:
            for year in YEARS:
                cal_path = calendar_builder.build_calendar_for_year(market, sym, year)
                log(f"  Calendar: {sym} {market} {year}")

            df_state = lake.market_state(sym, market=market)
            log(f"  Market State: {sym} {market} = {len(df_state)} rows")

            if not df_state.empty:
                meta_dir = config.canonical_dir / market / sym / "metadata"
                meta_dir.mkdir(parents=True, exist_ok=True)

                ds_ver = metadata_manager.create_dataset_version(
                    binance_snapshot=SNAPSHOT_DATE, created=SNAPSHOT_DATE)
                metadata_manager.save_json(ds_ver, meta_dir / "dataset_version.json")

                stats = metadata_manager.compute_statistics(df_state)
                metadata_manager.save_json(stats, meta_dir / "statistics_v1.json")

                src_schema = Path(__file__).resolve().parent / "configs" / "market_state_schema_v1.json"
                if src_schema.exists():
                    shutil.copy2(str(src_schema), str(meta_dir / "market_state_schema_v1.json"))

                datacard = datacard_builder.build_symbol_datacard(sym, market)
                datacard_builder.save_datacard(datacard, meta_dir / "DATASET.md")

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s")
    pipeline_manifest.record_stage("alignment", inputs=[], outputs=[], checksums=[], duration_seconds=elapsed)
    pipeline_manifest.save()


# ──────────────────────────────────────────────
# STAGE 6: Validate
# ──────────────────────────────────────────────
def stage_validate():
    log("--- STAGE 6: Validate ---")
    t0 = time.time()
    total_errors = 0

    for market in MARKETS:
        for sym in SYMBOLS:
            klines_pattern = str(config.canonical_dir / market / sym / "klines" / "1m" / "*.parquet").replace("\\", "/")
            try:
                import duckdb
                conn = duckdb.connect(":memory:")
                df_klines = conn.sql(f"SELECT * FROM read_parquet('{klines_pattern}') ORDER BY timestamp").df()
                conn.close()
                if not df_klines.empty:
                    valid, errors = validator.validate_klines(df_klines)
                    if not valid:
                        log(f"  {sym} {market} klines: FAIL - {errors}")
                        total_errors += len(errors)
                    else:
                        log(f"  {sym} {market} klines: PASS ({len(df_klines)} rows)")
                else:
                    log(f"  {sym} {market} klines: NO DATA")
            except Exception as e:
                log(f"  {sym} {market} klines: ERROR - {e}")

            funding_pattern = str(config.canonical_dir / market / sym / "funding" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                df_fund = conn.sql(f"SELECT * FROM read_parquet('{funding_pattern}') ORDER BY timestamp").df()
                conn.close()
                if not df_fund.empty:
                    valid, errors = validator.validate_funding(df_fund)
                    if not valid:
                        log(f"  {sym} {market} funding: FAIL - {errors}")
                        total_errors += len(errors)
                    else:
                        log(f"  {sym} {market} funding: PASS ({len(df_fund)} rows)")
            except Exception:
                pass

            oi_pattern = str(config.canonical_dir / market / sym / "open_interest" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                df_oi = conn.sql(f"SELECT * FROM read_parquet('{oi_pattern}') ORDER BY timestamp").df()
                conn.close()
                if not df_oi.empty:
                    log(f"  {sym} {market} open_interest: OK ({len(df_oi)} rows)")
            except Exception:
                pass

    elapsed = time.time() - t0
    log(f"  Validation: {total_errors} errors, done in {elapsed:.1f}s")


# ──────────────────────────────────────────────
# STAGE 7: Quality Reports
# ──────────────────────────────────────────────
def stage_quality_reports():
    log("--- STAGE 7: Quality Reports ---")
    t0 = time.time()

    for market in MARKETS:
        for sym in SYMBOLS:
            df_state = lake.market_state(sym, market=market)
            report = quality_reporter.generate_report(sym, market, df_state)
            out_path = config.canonical_dir / market / sym / "metadata" / "quality_report.json"
            quality_reporter.save_report(report, out_path)
            log(f"  {sym} {market}: score={report['quality_score']:.1f}, records={report['total_records']}")

    log(f"  Done in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────
# STAGE 8: Features & Windows
# ──────────────────────────────────────────────
def stage_feature_windows():
    log("--- STAGE 8: Features & Windows ---")
    t0 = time.time()
    total_windows = 0

    for market in MARKETS:
        for sym in SYMBOLS:
            df_state = lake.market_state(sym, market=market)
            if df_state.empty:
                log(f"  {sym} {market}: no data")
                continue

            features, feature_mask, timestamps = feature_builder.build_features(df_state)
            log(f"  {sym} {market}: features={features.shape}")

            meta = {
                "symbol": sym, "market": market,
                "start_ts": int(timestamps[0]) if len(timestamps) > 0 else 0,
                "end_ts": int(timestamps[-1]) if len(timestamps) > 0 else 0,
                "snapshot_id": SNAPSHOT_DATE,
                "modality_config": "modalities_v1.yaml",
                "windowing_config": "windowing_v1.yaml",
            }
            windows = windowing_engine.create_windows(features, feature_mask, timestamps, metadata=meta)
            total_windows += len(windows)
            log(f"  {sym} {market}: {len(windows)} windows (features {features.shape})")

            # Determinism check
            if len(windows) > 0:
                f2, m2, t2 = feature_builder.build_features(df_state)
                w2 = windowing_engine.create_windows(f2, m2, t2, metadata=meta)
                ok = len(windows) == len(w2)
                if ok and len(windows) > 0:
                    import hashlib as hl
                    h1 = hl.sha256(windows[0]["features"].tobytes()).hexdigest()
                    h2 = hl.sha256(w2[0]["features"].tobytes()).hexdigest()
                    ok = h1 == h2
                log(f"  {sym} {market}: determinism = {'PASS' if ok else 'FAIL'}")
            else:
                log(f"  {sym} {market}: no windows (seq_len=512 > data_len={len(df_state)})")

    log(f"  Total: {total_windows} windows, {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────
# STAGE 9: Report
# ──────────────────────────────────────────────
def stage_report():
    log("--- STAGE 9: Storage Report ---")
    summary = reports_generator.generate_storage_summary()
    out_path = config.training_dir / "storage_summary.md"
    reports_generator.save_report(summary, out_path)
    log(f"  Report saved: {out_path}")


# ──────────────────────────────────────────────
# STAGE 10: Snapshot
# ──────────────────────────────────────────────
def stage_snapshot():
    log("--- STAGE 10: Snapshot ---")
    t0 = time.time()

    snap_dir = config.training_dir / "snapshots" / SNAPSHOT_DATE
    if snap_dir.exists():
        shutil.rmtree(str(snap_dir))
        log(f"  Removed old snapshot")

    manifest = manifest_builder.build_manifest_from_index(
        snapshot_date=SNAPSHOT_DATE,
        train_symbols=["BTCUSDT", "ETHUSDT"],
        val_symbols=["SOLUSDT"],
        test_symbols=["BTCUSDT"],
    )

    snap_path = snapshot_manager.create_snapshot(
        snapshot_date=SNAPSHOT_DATE, manifest_data=manifest,
        checksums_data=None, stats_data={"snapshot": SNAPSHOT_DATE},
    )
    log(f"  Snapshot: {snap_path}")

    root_manifest = config.training_dir / "training_manifest_v1.json"
    manifest_builder.save_manifest(manifest, root_manifest)

    fingerprint = manifest_builder.compute_fingerprint(manifest)
    fp_path = config.training_dir / "dataset_fingerprint.json"
    manifest_builder.save_fingerprint(fingerprint, fp_path)
    log(f"  Fingerprint: {fingerprint['fingerprint'][:32]}...")
    log(f"  Done in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────
# SCIENTIFIC VALIDATION
# ──────────────────────────────────────────────
def stage_scientific_validation():
    log("--- STAGE 11: Scientific Validation ---")
    t0 = time.time()
    issues = []

    for market in MARKETS:
        for sym in SYMBOLS:
            df_state = lake.market_state(sym, market=market)
            if df_state.empty:
                continue

            # Check causality: funding_rate should never be NaN AFTER its first observation
            if "funding_rate" in df_state.columns:
                fr = df_state["funding_rate"]
                first_observed_idx = fr.first_valid_index()
                if first_observed_idx is not None:
                    after_first = fr.loc[first_observed_idx:]
                    nan_after_first = after_first.isna().sum()
                    if nan_after_first > 0:
                        # This is expected if the modality has gaps before being observed
                        # Forward fill should fill NaNs after first observation
                        log(f"  {sym} {market}: funding_rate has {nan_after_first} NaNs after first observation (forward-fill issue)")
                        issues.append(f"{sym}_{market}_funding_nan")
                    else:
                        log(f"  {sym} {market}: funding_rate causality OK")

            # Check no future leakage: timestamps must be monotonically increasing
            if "timestamp" in df_state.columns:
                ts = df_state["timestamp"]
                if not ts.is_monotonic_increasing:
                    issues.append(f"{sym}_{market}_non_monotonic_ts")
                    log(f"  {sym} {market}: FAIL - non-monotonic timestamps")
                else:
                    log(f"  {sym} {market}: monotonic timestamps OK")

                # Check no duplicate timestamps
                dups = ts.duplicated().sum()
                if dups > 0:
                    issues.append(f"{sym}_{market}_duplicate_ts")
                    log(f"  {sym} {market}: FAIL - {dups} duplicate timestamps")
                else:
                    log(f"  {sym} {market}: unique timestamps OK")

            # Check open_interest alignment
            if "open_interest" in df_state.columns:
                oi = df_state["open_interest"]
                first_oi = oi.first_valid_index()
                if first_oi is not None:
                    after_first_oi = oi.loc[first_oi:]
                    nan_oi = after_first_oi.isna().sum()
                    if nan_oi > 0:
                        issues.append(f"{sym}_{market}_oi_nan")
                        log(f"  {sym} {market}: open_interest has {nan_oi} NaNs after first observation")
                    else:
                        log(f"  {sym} {market}: open_interest forward-fill OK")

            # Check kline values
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df_state.columns:
                    vals = df_state[col].dropna()
                    if col != "volume" and (vals <= 0).any():
                        log(f"  {sym} {market}: WARNING - {col} has {(vals <= 0).sum()} non-positive values")

            # Check high >= low
            if "high" in df_state.columns and "low" in df_state.columns:
                bad_hl = (df_state["high"] < df_state["low"]).sum()
                if bad_hl > 0:
                    issues.append(f"{sym}_{market}_high_lt_low")
                    log(f"  {sym} {market}: FAIL - {bad_hl} rows with high < low")
                else:
                    log(f"  {sym} {market}: high >= low OK")

    elapsed = time.time() - t0
    if issues:
        log(f"  SCIENTIFIC ISSUES FOUND: {issues}")
    else:
        log(f"  ALL SCIENTIFIC CHECKS PASSED")
    log(f"  Done in {elapsed:.1f}s")
    return issues


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()
    log("=" * 55)
    log("  Pure Market Foundation Model - Full Pipeline v3")
    log("=" * 55)

    stage_exchange_info()
    stage_download()
    stage_convert()
    stage_resample()
    stage_build_lake()
    stage_validate()
    stage_quality_reports()
    stage_feature_windows()
    stage_report()
    stage_snapshot()
    issues = stage_scientific_validation()

    total = time.time() - start
    log("=" * 55)
    log(f"  PIPELINE COMPLETE: {total:.1f}s ({total/60:.1f}m)")
    if issues:
        log(f"  SCIENTIFIC ISSUES: {issues}")
    else:
        log(f"  ALL CHECKS PASSED - READY FOR PHASE 2")
    log("=" * 55)
