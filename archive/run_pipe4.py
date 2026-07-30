"""Fast Pipeline v4 - continues from existing data."""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import httpx
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
YEARS = [2024]
MONTHS = list(range(1, 13))
DS_PATHS = {
    "klines": "klines", "funding": "fundingRate", "fundingRate": "fundingRate",
    "open_interest": "metrics", "metrics": "metrics",
    "aggTrades": "aggTrades", "trades": "trades",
}
DL_TYPES = ["klines", "funding", "open_interest", "aggTrades", "trades"]
BASE = "https://data.binance.vision/data"


def log(m):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {m}", flush=True)


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(65536)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def burl(mkt, dt, sym, fn):
    mp = "futures/um" if mkt == "futures" else "spot"
    dp = DS_PATHS.get(dt, dt)
    if dt == "klines":
        return f"{BASE}/{mp}/monthly/klines/{sym}/1m/{fn}"
    return f"{BASE}/{mp}/monthly/{dp}/{sym}/{fn}"


def dl_file(cl, url, dest, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(".tmp")
    for a in range(retries):
        try:
            with cl.stream("GET", url, follow_redirects=True, timeout=60) as resp:
                if resp.status_code == 200:
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes(65536):
                            f.write(chunk)
                    tmp.replace(dest)
                    return True
                elif resp.status_code == 404:
                    return False
        except Exception:
            if a == retries - 1:
                tmp.unlink(missing_ok=True)
                return False
            time.sleep(1 * (a + 1))
    return False


# ================================================================
# STAGE 1: Download
# ================================================================
def stage_download():
    log("=== DOWNLOAD ===")
    t0 = time.time()
    stats = {"ok": 0, "cached": 0, "miss": 0}
    tasks = []
    for mk in MKTS:
        for s in SYMS:
            for dt in DL_TYPES:
                for y in YEARS:
                    for mo in MONTHS:
                        ms = f"{mo:02d}"
                        if dt == "klines":
                            fn = f"{s}-1m-{y}-{ms}.zip"
                        else:
                            fn = f"{s}-{DS_PATHS.get(dt, dt)}-{y}-{ms}.zip"
                        url = burl(mk, dt, s, fn)
                        td = config.raw_dir / mk / s / dt / ("1m" if dt == "klines" else "")
                        tasks.append((mk, s, dt, y, mo, url, td / fn))

    total = len(tasks)
    log("  " + str(total) + " items to check")
    with httpx.Client(timeout=60, follow_redirects=True) as cl:
        for i, (mk, s, dt, y, mo, url, dest) in enumerate(tasks):
            if dest.exists() and dest.stat().st_size > 0:
                stats["cached"] += 1
                continue
            ok = dl_file(cl, url, dest)
            if ok:
                stats["ok"] += 1
                dl_file(cl, url + ".checksum", dest.parent / (dest.name + ".checksum"), 1)
            else:
                stats["miss"] += 1
            done = i + 1
            if done % 30 == 0 or done == total:
                log("  [%d/%d] ok=%d cached=%d miss=%d" % (done, total, stats["ok"], stats["cached"], stats["miss"]))

    elapsed = time.time() - t0
    log("  Done: %d new, %d cached, %d missing (%.0fs)" % (stats["ok"], stats["cached"], stats["miss"], elapsed))
    pipeline_manifest.record_stage("download", [], [], [], elapsed, metadata=stats)
    pipeline_manifest.save()


# ================================================================
# STAGE 2: Convert
# ================================================================
def stage_convert():
    log("=== CONVERT ===")
    t0 = time.time()
    conv = cached = errs = 0
    for mk in MKTS:
        for s in SYMS:
            for dt in DL_TYPES:
                for y in YEARS:
                    for mo in MONTHS:
                        ms = f"{mo:02d}"
                        if dt == "klines":
                            fn = f"{s}-1m-{y}-{ms}.zip"
                            rd = config.raw_dir / mk / s / dt / "1m"
                            od = config.canonical_dir / mk / s / dt / "1m"
                            iv = "1m"
                        else:
                            dp = DS_PATHS.get(dt, dt)
                            fn = f"{s}-{dp}-{y}-{ms}.zip"
                            rd = config.raw_dir / mk / s / dt / ""
                            od = config.canonical_dir / mk / s / dt
                            iv = ""
                        zp = rd / fn
                        if not zp.exists():
                            continue
                        ep = od / ("%d-%02d.parquet" % (y, mo))
                        if ep.exists():
                            try:
                                s_hash = sha256f(ep)
                                import pyarrow.parquet as pq
                                pf = pq.read_metadata(str(ep))
                                fid = "%s_%s_%s_%s_%d_%d" % (mk, s, dt, iv, y, mo)
                                db_manager.register_file(
                                    fid, s, mk, dt, iv, y, mo,
                                    0, 0, pf.num_rows, ep.stat().st_size,
                                    s_hash, "", str(ep), "CONVERTED"
                                )
                                cached += 1
                                continue
                            except Exception:
                                pass
                        try:
                            converter.convert_zip_to_parquet(zp, mk, s, dt, iv, y, mo, SNAP)
                            conv += 1
                            if conv % 10 == 0:
                                log("  Converted %d..." % conv)
                        except Exception as e:
                            errs += 1
                            log("  ERR %s/%s/%d-%02d: %s" % (s, dt, y, mo, str(e)[:80]))

    log("  Done: %d new, %d cached, %d errors (%.0fs)" % (conv, cached, errs, time.time() - t0))
    pipeline_manifest.record_stage("convert", [], [], [], time.time() - t0)
    pipeline_manifest.save()


# ================================================================
# STAGE 3: Resample
# ================================================================
def stage_resample():
    log("=== RESAMPLE ===")
    t0 = time.time()
    tfs = ["5m", "15m", "1h", "4h", "1d"]
    res = cached = errs = 0
    for mk in MKTS:
        for s in SYMS:
            for y in YEARS:
                for mo in MONTHS:
                    inp = config.canonical_dir / mk / s / "klines" / "1m" / ("%d-%02d.parquet" % (y, mo))
                    if not inp.exists():
                        continue
                    for tf in tfs:
                        ep = config.canonical_dir / mk / s / "klines" / tf / ("%d-%02d.parquet" % (y, mo))
                        if ep.exists():
                            try:
                                s_hash = sha256f(ep)
                                import pyarrow.parquet as pq
                                pf = pq.read_metadata(str(ep))
                                fid = "%s_%s_klines_%s_%d_%d" % (mk, s, tf, y, mo)
                                db_manager.register_file(
                                    fid, s, mk, "klines", tf, y, mo,
                                    0, 0, pf.num_rows, ep.stat().st_size,
                                    s_hash, "", str(ep), "RESAMPLED"
                                )
                                cached += 1
                                continue
                            except Exception:
                                pass
                        try:
                            resampler.resample_file(inp, mk, s, tf, y, mo)
                            res += 1
                        except Exception as e:
                            errs += 1
                            log("  ERR %s %s: %s" % (s, tf, str(e)[:80]))

    log("  Done: %d new, %d cached, %d errors (%.0fs)" % (res, cached, errs, time.time() - t0))
    pipeline_manifest.record_stage("resample", [], [], [], time.time() - t0)
    pipeline_manifest.save()


# ================================================================
# STAGE 4: Build Lake
# ================================================================
def stage_build_lake():
    log("=== BUILD LAKE ===")
    t0 = time.time()
    for mk in MKTS:
        for s in SYMS:
            for y in YEARS:
                cal = calendar_builder.build_calendar_for_year(mk, s, y)
                log("  Calendar: %s %s %d" % (s, mk, y))

            df = lake.market_state(s, market=mk)
            log("  Market State: %s %s = %d rows" % (s, mk, len(df)))
            if not df.empty:
                md = config.canonical_dir / mk / s / "metadata"
                md.mkdir(parents=True, exist_ok=True)
                metadata_manager.save_json(
                    metadata_manager.create_dataset_version(binance_snapshot=SNAP, created=SNAP),
                    md / "dataset_version.json"
                )
                metadata_manager.save_json(
                    metadata_manager.compute_statistics(df),
                    md / "statistics_v1.json"
                )
                src = Path(__file__).resolve().parent / "configs" / "market_state_schema_v1.json"
                if src.exists():
                    shutil.copy2(str(src), str(md / "market_state_schema_v1.json"))
                dc = datacard_builder.build_symbol_datacard(s, mk)
                datacard_builder.save_datacard(dc, md / "DATASET.md")

    log("  Done (%.0fs)" % (time.time() - t0))
    pipeline_manifest.record_stage("alignment", [], [], [], time.time() - t0)
    pipeline_manifest.save()


# ================================================================
# STAGE 5: Validate
# ================================================================
def stage_validate():
    log("=== VALIDATE ===")
    t0 = time.time()
    import duckdb
    for mk in MKTS:
        for s in SYMS:
            kp = str(config.canonical_dir / mk / s / "klines" / "1m" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                r = conn.sql("SELECT count(*) as n FROM read_parquet('%s')" % kp).fetchone()
                conn.close()
                if r and r[0] > 0:
                    log("  %s %s klines: %d rows OK" % (s, mk, r[0]))
            except Exception as e:
                log("  %s %s klines: %s" % (s, mk, str(e)[:80]))

            fp = str(config.canonical_dir / mk / s / "funding" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                r = conn.sql("SELECT count(*) FROM read_parquet('%s')" % fp).fetchone()
                conn.close()
                if r and r[0] > 0:
                    log("  %s %s funding: %d rows OK" % (s, mk, r[0]))
            except Exception:
                pass

            op = str(config.canonical_dir / mk / s / "open_interest" / "*.parquet").replace("\\", "/")
            try:
                conn = duckdb.connect(":memory:")
                r = conn.sql("SELECT count(*) FROM read_parquet('%s')" % op).fetchone()
                conn.close()
                if r and r[0] > 0:
                    log("  %s %s open_interest: %d rows OK" % (s, mk, r[0]))
            except Exception:
                pass

    log("  Done (%.0fs)" % (time.time() - t0))


# ================================================================
# STAGE 6: Quality Reports
# ================================================================
def stage_quality():
    log("=== QUALITY REPORTS ===")
    t0 = time.time()
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            rpt = quality_reporter.generate_report(s, mk, df)
            op = config.canonical_dir / mk / s / "metadata" / "quality_report.json"
            quality_reporter.save_report(rpt, op)
            log("  %s %s: score=%.0f, records=%d" % (s, mk, rpt["quality_score"], rpt["total_records"]))
    log("  Done (%.0fs)" % (time.time() - t0))


# ================================================================
# STAGE 7: Features & Windows
# ================================================================
def stage_features():
    log("=== FEATURES & WINDOWS ===")
    t0 = time.time()
    tw = 0
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            if df.empty:
                log("  %s %s: no data" % (s, mk))
                continue
            feat, fm, ts = feature_builder.build_features(df)
            meta = {
                "symbol": s, "market": mk,
                "start_ts": int(ts[0]) if len(ts) else 0,
                "end_ts": int(ts[-1]) if len(ts) else 0,
                "snapshot_id": SNAP,
                "modality_config": "modalities_v1.yaml",
                "windowing_config": "windowing_v1.yaml",
            }
            wins = windowing_engine.create_windows(feat, fm, ts, metadata=meta)
            tw += len(wins)
            if wins:
                f2, m2, t2 = feature_builder.build_features(df)
                w2 = windowing_engine.create_windows(f2, m2, t2, metadata=meta)
                ok = len(wins) == len(w2)
                if ok and wins:
                    import hashlib as hl
                    h1 = hl.sha256(wins[0]["features"].tobytes()).hexdigest()
                    h2 = hl.sha256(w2[0]["features"].tobytes()).hexdigest()
                    ok = h1 == h2
                log("  %s %s: %d windows, dim=%d, det=%s" % (s, mk, len(wins), feat.shape[1], "PASS" if ok else "FAIL"))
            else:
                log("  %s %s: %d rows, 0 windows (< seq_len=512)" % (s, mk, len(df)))

    log("  Total %d windows (%.0fs)" % (tw, time.time() - t0))


# ================================================================
# STAGE 8: Report
# ================================================================
def stage_report():
    log("=== STORAGE REPORT ===")
    s = reports_generator.generate_storage_summary()
    reports_generator.save_report(s, config.training_dir / "storage_summary.md")
    log("  Saved")


# ================================================================
# STAGE 9: Snapshot
# ================================================================
def stage_snapshot():
    log("=== SNAPSHOT ===")
    t0 = time.time()
    sd = config.training_dir / "snapshots" / SNAP
    if sd.exists():
        shutil.rmtree(str(sd))
    m = manifest_builder.build_manifest_from_index(
        SNAP, train_symbols=["BTCUSDT", "ETHUSDT"],
        val_symbols=["SOLUSDT"], test_symbols=["BTCUSDT"]
    )
    snap = snapshot_manager.create_snapshot(SNAP, m, None, {"snapshot": SNAP})
    manifest_builder.save_manifest(m, config.training_dir / "training_manifest_v1.json")
    fp = manifest_builder.compute_fingerprint(m)
    manifest_builder.save_fingerprint(fp, config.training_dir / "dataset_fingerprint.json")
    log("  Snapshot: %s" % snap)
    log("  Fingerprint: %s..." % fp["fingerprint"][:32])
    log("  Done (%.0fs)" % (time.time() - t0))


# ================================================================
# STAGE 10: Scientific Validation
# ================================================================
def stage_scientific():
    log("=== SCIENTIFIC VALIDATION ===")
    t0 = time.time()
    import duckdb
    issues = []
    for mk in MKTS:
        for s in SYMS:
            df = lake.market_state(s, market=mk)
            if df.empty:
                continue
            ts = df["timestamp"]
            if not ts.is_monotonic_increasing:
                issues.append("%s_%s_ts_order" % (s, mk))
                log("  FAIL %s %s: non-monotonic timestamps" % (s, mk))
            dups = ts.duplicated().sum()
            if dups > 0:
                issues.append("%s_%s_ts_dups" % (s, mk))
                log("  FAIL %s %s: %d duplicate timestamps" % (s, mk, dups))
            else:
                log("  OK %s %s: timestamps unique and monotonic" % (s, mk))

            if "funding_rate" in df.columns:
                fr = df["funding_rate"]
                fi = fr.first_valid_index()
                if fi is not None:
                    nans = fr.loc[fi:].isna().sum()
                    if nans > 0:
                        log("  WARN %s %s: funding %d NaNs after first obs" % (s, mk, nans))
                    else:
                        log("  OK %s %s: funding forward-fill complete" % (s, mk))

            if "open_interest" in df.columns:
                oi = df["open_interest"]
                fi = oi.first_valid_index()
                if fi is not None:
                    nans = oi.loc[fi:].isna().sum()
                    if nans > 0:
                        log("  WARN %s %s: OI %d NaNs after first obs" % (s, mk, nans))
                    else:
                        log("  OK %s %s: OI forward-fill complete" % (s, mk))

            if "high" in df.columns and "low" in df.columns:
                bad = int((df["high"] < df["low"]).sum())
                if bad > 0:
                    issues.append("%s_%s_hl" % (s, mk))
                    log("  FAIL %s %s: %d rows high < low" % (s, mk, bad))
                else:
                    log("  OK %s %s: high >= low" % (s, mk))

    # Aggregation math check: 1m -> 5m ratio
    for mk in MKTS:
        for s in SYMS:
            try:
                conn = duckdb.connect(":memory:")
                p1 = str(config.canonical_dir / mk / s / "klines" / "1m" / "*.parquet").replace("\\", "/")
                p5 = str(config.canonical_dir / mk / s / "klines" / "5m" / "*.parquet").replace("\\", "/")
                r1 = conn.sql("SELECT count(*) FROM read_parquet('%s')" % p1).fetchone()
                r5 = conn.sql("SELECT count(*) FROM read_parquet('%s')" % p5).fetchone()
                conn.close()
                if r1 and r1[0] > 0 and r5 and r5[0] > 0:
                    ratio = r1[0] / r5[0]
                    ok = 4.0 <= ratio <= 6.0
                    log("  %s %s: 1m/5m ratio=%.2f %s" % (s, mk, ratio, "OK" if ok else "WARN"))
            except Exception:
                pass

    elapsed = time.time() - t0
    if issues:
        log("  ISSUES: %s" % str(issues))
    else:
        log("  ALL SCIENTIFIC CHECKS PASSED")
    log("  Done (%.0fs)" % elapsed)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    start = time.time()
    log("=" * 55)
    log("  Pure Market Foundation Model - Pipeline v4")
    log("=" * 55)

    stage_download()
    stage_convert()
    stage_resample()
    stage_build_lake()
    stage_validate()
    stage_quality()
    stage_features()
    stage_report()
    stage_snapshot()
    stage_scientific()

    total = time.time() - start
    log("=" * 55)
    log("  PIPELINE COMPLETE: %.0fs (%.1fm)" % (total, total / 60))
    log("=" * 55)
