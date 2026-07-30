"""Focused downloader: Phase 1 active modalities only, throttle-safe."""
import sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import httpx
from src.config import config

DS = {"klines": "klines", "funding": "fundingRate", "open_interest": "metrics"}
# Phase 1 active types only (modalities_v1.yaml: aggTrades/trades/depth/liquidations disabled)
DL = ["klines", "funding", "open_interest"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MKTS = ["futures", "spot"]
BASE = "https://data.binance.vision/data"


def burl(mk, dt, s, fn):
    mp = "futures/um" if mk == "futures" else "spot"
    dp = DS.get(dt, dt)
    if dt == "klines":
        return f"{BASE}/{mp}/monthly/klines/{s}/1m/{fn}"
    return f"{BASE}/{mp}/monthly/{dp}/{s}/{fn}"


def dl_one(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    tmp = dest.with_suffix(".tmp")
    for attempt in range(2):
        try:
            with httpx.Client(timeout=45, follow_redirects=True) as cl:
                with cl.stream("GET", url) as r:
                    if r.status_code == 404:
                        return "miss"
                    if r.status_code != 200:
                        time.sleep(2)
                        continue
                    with open(tmp, "wb") as f:
                        for c in r.iter_bytes(65536):
                            f.write(c)
                    tmp.replace(dest)
                    return "ok"
        except Exception:
            tmp.unlink(missing_ok=True)
            time.sleep(2)
    return "err"


tasks = []
for mk in MKTS:
    for s in SYMS:
        for dt in DL:
            # spot has no funding/OI on Binance Vision
            if mk == "spot" and dt in ("funding", "open_interest"):
                continue
            for mo in range(1, 13):
                fn = f"{s}-1m-2024-{mo:02d}.zip" if dt == "klines" else f"{s}-{DS.get(dt,dt)}-2024-{mo:02d}.zip"
                url = burl(mk, dt, s, fn)
                td = config.raw_dir / mk / s / dt / ("1m" if dt == "klines" else "")
                tasks.append((url, td / fn))

total = len(tasks)
stats = {}
print(f"=== FOCUSED DOWNLOAD: {total} items (Phase 1 modalities only) ===", flush=True)
t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    futs = {pool.submit(dl_one, url, dest): dest for url, dest in tasks}
    for fut in as_completed(futs):
        r = fut.result()
        stats[r] = stats.get(r, 0) + 1
        done += 1
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1)
        eta = (total - done) / max(rate, 0.01)
        if done % 5 == 0 or done == total:
            print(f"  [{done}/{total}] ok={stats.get('ok',0)} cached={stats.get('cached',0)} "
                  f"miss={stats.get('miss',0)} err={stats.get('err',0)} ETA={eta:.0f}s", flush=True)

elapsed = time.time() - t0
raw_files = list(Path("D:/emptyu/storage/raw").rglob("*.zip"))
raw_size = sum(f.stat().st_size for f in raw_files)
print(f"DONE in {elapsed:.0f}s: {stats}", flush=True)
print(f"Total raw: {len(raw_files)} files, {raw_size/(1024*1024*1024):.2f} GB", flush=True)
