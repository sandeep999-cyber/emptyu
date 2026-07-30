"""Remaining pipeline: OI aggregation, convert, resample, lake, validate, snapshot."""
import sys, os, time, hashlib, json, shutil, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import config
from src.data.db import db_manager
from src.data.parquet_converter import converter
from src.data.resampler import resampler
from src.data.calendar_builder import calendar_builder
from src.data.lake import lake
from src.data.validator import validator
from src.data.quality_report import quality_reporter
from src.data.manifest_builder import manifest_builder
from src.data.snapshot_manager import snapshot_manager
from src.data.datacard_builder import datacard_builder
from src.data.feature_builder import feature_builder
from src.data.windowing import windowing_engine
from src.data.reports import reports_generator
from src.data.metadata import metadata_manager
from src.data.pipeline_manifest import pipeline_manifest

SNAP = "2026-07-30"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MKTS = ["futures", "spot"]
DL = ["klines", "funding"]
TFS = ["5m", "15m", "1h", "4h", "1d"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(65536)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


# ---------- STAGE A: Aggregate daily OI zips -> monthly parquets ----------
def stage_oi_aggregate():
    log("=== OI DAILY->MONTHLY AGGREGATION ===")
    t0 = time.time()
    for sym in SYMS:
        daily_dir = config.raw_dir / "futures" / sym / "open_interest" / "daily"
        if not daily_dir.exists():
            continue
        for mo in range(1, 13):
            month_str = f"2024-{mo:02d}"
            zips = sorted(daily_dir.glob(f"{sym}-metrics-{month_str}-*.zip"))
            if not zips:
                continue
            out_dir = config.canonical_dir / "futures" / sym / "open_interest"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"2024-{mo:02d}.parquet"
            if out_path.exists():
                continue
            frames = []
            src_hashes = []
            for zp in zips:
                try:
                    with zipfile.ZipFile(zp, "r") as zf:
                        csv_name = zf.namelist()[0]
                        with zf.open(csv_name) as cf:
                            first = cf.readline().decode("utf-8")
                            cf.seek(0)
                            has_header = not first[0].isdigit()
                            cols = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value"]
                            df = pd.read_csv(cf, names=None if has_header else cols,
                                             header=0 if has_header else None,
                                             dtype_backend="numpy_nullable")
                            frames.append(df)
                            src_hashes.append(sha256f(zp))
                except Exception as e:
                    log(f"  WARN: {zp.name}: {e}")
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df = df.rename(columns={
                "create_time": "timestamp", "symbol": "base_symbol",
                "sum_open_interest": "open_interest",
                "sum_open_interest_value": "open_interest_value"})
            if "timestamp" not in df.columns:
                log(f"  WARN: no timestamp col for {sym} {month_str}, cols={list(df.columns)}")
                continue
            # Timestamps may be epoch ms (BTCUSDT) or datetime strings (ETHUSDT/SOLUSDT)
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            else:
                df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                                   .astype("int64") // 10**6)
            df.dropna(subset=["timestamp"], inplace=True)
            if df.empty:
                continue
            df["timestamp"] = df["timestamp"].astype("int64")
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
            # Keep only canonical OI columns (drop extra top-trader ratio columns)
            keep = [c for c in ["timestamp", "base_symbol", "open_interest", "open_interest_value"] if c in df.columns]
            df = df[keep]

            table = pa.Table.from_pandas(df, preserve_index=False)
            combined_hash = hashlib.sha256("|".join(src_hashes).encode()).hexdigest()
            provenance = {
                "provenance_created_by": "oi_daily_aggregator",
                "provenance_source": "binance_vision_daily_metrics",
                "provenance_source_checksum": f"sha256:{combined_hash}",
                "provenance_download_date": SNAP,
                "provenance_converter_version": "v1",
                "provenance_alignment_version": "alignment_v1.yaml",
                "provenance_schema_version": "canonical_schema_v1",
                "provenance_snapshot": SNAP,
                "provenance_daily_file_count": str(len(zips)),
            }
            meta = {**{k.encode(): str(v).encode() for k, v in provenance.items()},
                    **(table.schema.metadata or {})}
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, str(out_path), compression="snappy")

            fid = f"futures_{sym}_open_interest__2024_{mo}"
            db_manager.register_file(
                file_id=fid, symbol=sym, market="futures", dataset_type="open_interest",
                interval="", year=2024, month=mo,
                start_ts=int(df["timestamp"].min()), end_ts=int(df["timestamp"].max()),
                row_count=len(df), file_size=out_path.stat().st_size,
                sha256=sha256f(out_path),
                schema_hash=hashlib.md5(str(table.schema).encode()).hexdigest(),
                file_path=str(out_path), status="CONVERTED")
        log(f"  OI aggregated: {sym}")
    log(f"  Done ({time.time()-t0:.0f}s)")


# ---------- STAGE B: Convert klines + funding ----------
def stage_convert():
    log("=== CONVERT ===")
    t0 = time.time()
    conv = cached = errs = 0
    for mk in MKTS:
        for s in SYMS:
            for dt in DL:
                for mo in range(1, 13):
                    if dt == "klines":
                        fn = f"{s}-1m-2024-{mo:02d}.zip"
                        rd = config.raw_dir / mk / s / dt / "1m"
                        od = config.canonical_dir / mk / s / dt / "1m"
                        iv = "1m"
                    else:
                        if mk == "spot":
                            continue
                        fn = f"{s}-fundingRate-2024-{mo:02d}.zip"
                        rd = config.raw_dir / mk / s / dt / ""
                        od = config.canonical_dir / mk / s / dt
                        iv = ""
                    zp = rd / fn
                    if not zp.exists():
                        continue
                    ep = od / f"2024-{mo:02d}.parquet"
                    if ep.exists():
                        try:
                            pf = pq.read_metadata(str(ep))
                            fid = f"{mk}_{s}_{dt}_{iv}_2024_{mo}"
                            db_manager.register_file(
                                file_id=fid, symbol=s, market=mk, dataset_type=dt,
                                interval=iv, year=2024, month=mo,
                                start_ts=0, end_ts=0, row_count=pf.num_rows,
                                file_size=ep.stat().st_size, sha256=sha256f(ep),
                                schema_hash="", file_path=str(ep), status="CONVERTED")
                            cached += 1
                            continue
                        except Exception:
                            pass
                    try:
                        converter.convert_zip_to_parquet(
                            zip_path=zp, market=mk, symbol=s, dataset_type=dt,
                            interval=iv, year=2024, month=mo, download_date=SNAP)
                        conv += 1
                    except Exception as e:
                        errs += 1
                        log(f"  ERR {mk}/{s}/{dt}/{mo:02d}: {str(e)[:100]}")
    log(f"  Done: {conv} new, {cached} cached, {errs} errors ({time.time()-t0:.0f}s)")


# ---------- STAGE C: Resample ----------
def stage_resample():
    log("=== RESAMPLE ===")
    t0 = time.time()
    res = cached = errs = 0
    for mk in MKTS:
        for s in SYMS:
            for mo in range(1, 13):
                inp = config.canonical_dir / mk / s / "klines" / "1m" / f"2024-{mo:02d}.parquet"
                if not inp.exists():
                    continue
                for tf in TFS:
                    ep = config.canonical_dir / mk / s / "klines" / tf / f"2024-{mo:02d}.parquet"
                    if ep.exists():
                        try:
                            pf = pq.read_metadata(str(ep))
                            fid = f"{mk}_{s}_klines_{tf}_2024_{mo}"
                            db_manager.register_file(
                                file_id=fid, symbol=s, market=mk, dataset_type="klines",
                                interval=tf, year=2024, month=mo,
                                start_ts=0, end_ts=0, row_count=pf.num_rows,
                                file_size=ep.stat().st_size, sha256=sha256f(ep),
                                schema_hash="", file_path=str(ep), status="RESAMPLED")
                            cached += 1
                            continue
                        except Exception:
                            pass
                    try:
                        resampler.resample_file(inp, mk, s, tf, 2024, mo)
                        res += 1
                    except Exception as e:
                        errs += 1
                        log(f"  ERR {s} {tf} {mo:02d}: {str(e)[:100]}")
    log(f"  Done: {res} new, {cached} cached, {errs} errors ({time.time()-t0:.0f}s)")


# ---------- STAGE D: Calendar + Lake + Metadata ----------
def stage_lake():
    log("=== CALENDAR + LAKE ===")
    t0 = time.time()
    for mk in MKTS:
        for s in SYMS:
            calendar_builder.build_calendar_for_year(mk, s, 2024)
            df = lake.market_state(s, market=mk)
            log(f"  {s} {mk}: {len(df)} aligned rows")
            if not df.empty:
                md = config.canonical_dir / mk / s / "metadata"
                md.mkdir(parents=True, exist_ok=True)
                metadata_manager.save_json(
                    metadata_manager.create_dataset_version(binance_snapshot=SNAP, created=SNAP),
                    md / "dataset_version.json")
                metadata_manager.save_json(metadata_manager.compute_statistics(df), md / "statistics_v1.json")
                src = Path(__file__).resolve().parent / "configs" / "market_state_schema_v1.json"
                if src.exists():
                    shutil.copy2(str(src), str(md / "market_state_schema_v1.json"))
                datacard_builder.save_datacard(datacard_builder.build_symbol_datacard(s, mk), md / "DATASET.md")
    log(f"  Done ({time.time()-t0:.0f}s)")


# ---------- STAGE E: Validate ----------
def stage_validate():
    log("=== VALIDATE ===")
    t0 = time.time()
    import duckdb
    for mk in MKTS:
        for s in SYMS:
            kp = str(config.canonical_dir / mk / s / "klines" / "1m" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                df = conn.sql(f"SELECT * FROM read_parquet('{kp}') ORDER BY timestamp").df()
                conn.close()
                if not df.empty:
                    v, errs = validator.validate_klines(df)
                    status = "PASS" if v else f"FAIL {errs[:2]}"
                    log(f"  {s} {mk} klines: {status} ({len(df)} rows)")
            except Exception as e:
                log(f"  {s} {mk} klines: ERROR {str(e)[:80]}")
            for dt_name in ["funding", "open_interest"]:
                pp = str(config.canonical_dir / mk / s / dt_name / "*.parquet").replace("\\", "/")
                try:
                    conn = duckdb.connect(":memory:")
                    n = conn.sql(f"SELECT count(*) FROM read_parquet('{pp}')").fetchone()[0]
                    conn.close()
                    log(f"  {s} {mk} {dt_name}: {n} rows")
                except Exception:
                    pass
    log(f"  Done ({time.time()-t0:.0f}s)")


# ---------- STAGE F: Quality reports ----------
def stage_quality():
    log("=== QUALITY REPORTS ===")
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            rpt = quality_reporter.generate_report(s, mk, df)
            quality_reporter.save_report(rpt, config.canonical_dir / mk / s / "metadata" / "quality_report.json")
            log(f"  {s} {mk}: score={rpt['quality_score']:.1f} records={rpt['total_records']}")


# ---------- STAGE G: Features + Windows + determinism ----------
def stage_features():
    log("=== FEATURES + WINDOWS ===")
    t0 = time.time()
    tw = 0
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            if df.empty:
                continue
            feat, fm, ts = feature_builder.build_features(df)
            meta = {"symbol": s, "market": mk,
                    "start_ts": int(ts[0]) if len(ts) else 0,
                    "end_ts": int(ts[-1]) if len(ts) else 0,
                    "snapshot_id": SNAP, "modality_config": "modalities_v1.yaml",
                    "windowing_config": "windowing_v1.yaml"}
            wins = windowing_engine.create_windows(feat, fm, ts, metadata=meta)
            tw += len(wins)
            det = "N/A"
            if wins:
                f2, m2, t2 = feature_builder.build_features(df)
                w2 = windowing_engine.create_windows(f2, m2, t2, metadata=meta)
                ok = len(wins) == len(w2)
                if ok:
                    h1 = hashlib.sha256(wins[0]["features"].tobytes()).hexdigest()
                    h2 = hashlib.sha256(w2[0]["features"].tobytes()).hexdigest()
                    ok = h1 == h2
                det = "PASS" if ok else "FAIL"
            log(f"  {s} {mk}: {len(wins)} windows, det={det}")
    log(f"  Total windows: {tw} ({time.time()-t0:.0f}s)")


# ---------- STAGE H: Report + Snapshot ----------
def stage_snapshot():
    log("=== REPORT + SNAPSHOT ===")
    reports_generator.save_report(
        reports_generator.generate_storage_summary(),
        config.training_dir / "storage_summary.md")
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
    log(f"  Fingerprint: {fp['fingerprint'][:32]}...")


# ---------- STAGE I: Scientific validation ----------
def stage_scientific():
    log("=== SCIENTIFIC VALIDATION ===")
    issues = []
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            if df.empty:
                continue
            ts = df["timestamp"]
            if not ts.is_monotonic_increasing:
                issues.append(f"{s}_{mk}_ts_order")
            dups = int(ts.duplicated().sum())
            if dups:
                issues.append(f"{s}_{mk}_ts_dups")
            for col in ["funding_rate", "open_interest"]:
                if col in df.columns:
                    fr = df[col]
                    fi = fr.first_valid_index()
                    nans_after = int(fr.loc[fi:].isna().sum()) if fi is not None else len(fr)
                    cov = 1 - fr.isna().sum() / max(len(fr), 1)
                    log(f"  {s} {mk} {col}: coverage={cov*100:.1f}% nans_after_first={nans_after}")
            if "high" in df.columns and "low" in df.columns:
                bad = int((df["high"] < df["low"]).sum())
                if bad:
                    issues.append(f"{s}_{mk}_hl")
                nan_ohlc = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
                if nan_ohlc:
                    issues.append(f"{s}_{mk}_ohlc_nan")
            log(f"  {s} {mk}: ts_monotonic={ts.is_monotonic_increasing} dups={dups}")
    if issues:
        log(f"  ISSUES: {issues}")
    else:
        log("  ALL SCIENTIFIC CHECKS PASSED")
    return issues


if __name__ == "__main__":
    start = time.time()
    log("=" * 50)
    log("  LOCAL PIPELINE (post-download)")
    log("=" * 50)
    stage_oi_aggregate()
    stage_convert()
    stage_resample()
    stage_lake()
    stage_validate()
    stage_quality()
    stage_features()
    stage_snapshot()
    issues = stage_scientific()
    total = time.time() - start
    log("=" * 50)
    log(f"  COMPLETE: {total:.0f}s ({total/60:.1f}m) issues={issues}")
    log("=" * 50)
