"""Parallel downloader for Binance Vision archives."""
import sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import httpx
from src.config import config

DS = {"klines":"klines","funding":"fundingRate","open_interest":"metrics","aggTrades":"aggTrades","trades":"trades"}
DL = ["klines","funding","open_interest","aggTrades","trades"]
SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT"]
MKTS = ["futures","spot"]
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
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as cl:
            with cl.stream("GET", url) as r:
                if r.status_code == 404:
                    return "miss"
                if r.status_code != 200:
                    return "err_%d" % r.status_code
                with open(tmp, "wb") as f:
                    for c in r.iter_bytes(65536):
                        f.write(c)
                tmp.replace(dest)
                return "ok"
    except Exception:
        tmp.unlink(missing_ok=True)
        return "err"


print("=== PARALLEL DOWNLOAD ===", flush=True)
t0 = time.time()
tasks = []
for mk in MKTS:
    for s in SYMS:
        for dt in DL:
            for mo in range(1, 13):
                fn = f"{s}-1m-2024-{mo:02d}.zip" if dt == "klines" else f"{s}-{DS.get(dt,dt)}-2024-{mo:02d}.zip"
                url = burl(mk, dt, s, fn)
                td = config.raw_dir / mk / s / dt / ("1m" if dt == "klines" else "")
                tasks.append((url, td / fn))

total = len(tasks)
print(f"  {total} items, 8 parallel workers", flush=True)
stats = {"ok": 0, "cached": 0, "miss": 0, "err": 0}
done_count = 0

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(dl_one, url, dest): (url, dest) for url, dest in tasks}
    for fut in as_completed(futures):
        result = fut.result()
        stats[result] = stats.get(result, 0) + 1
        done_count += 1
        if done_count % 30 == 0 or done_count == total:
            print(f"  [{done_count}/{total}] ok={stats.get('ok',0)} c={stats.get('cached',0)} miss={stats.get('miss',0)} err={stats.get('err',0)}", flush=True)

elapsed = time.time() - t0
raw_files = list(Path("D:/emptyu/storage/raw").rglob("*.zip"))
raw_size = sum(f.stat().st_size for f in raw_files)
print(f"  DONE: {stats.get('ok',0)} new, {stats.get('cached',0)} cached, {stats.get('miss',0)} missing, {stats.get('err',0)} errors", flush=True)
print(f"  Total raw: {len(raw_files)} files, {raw_size/(1024*1024*1024):.2f} GB ({elapsed:.0f}s)", flush=True)
