"""Robust downloader with per-file timeout."""
import sys, os, time, hashlib, signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import httpx
from src.config import config

DS = {"klines":"klines","funding":"fundingRate","open_interest":"metrics","aggTrades":"aggTrades","trades":"trades"}
DL = ["klines","funding","open_interest","aggTrades","trades"]
SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT"]
MKTS = ["futures","spot"]
BASE = "https://data.binance.vision/data"

def burl(mk,dt,s,fn):
    mp = "futures/um" if mk=="futures" else "spot"
    dp = DS.get(dt,dt)
    if dt=="klines":
        return f"{BASE}/{mp}/monthly/klines/{s}/1m/{fn}"
    return f"{BASE}/{mp}/monthly/{dp}/{s}/{fn}"

def dl_one(args):
    url, dest = args
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
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return "err"

print("=== ROBUST DOWNLOAD ===", flush=True)
t0 = time.time()
tasks = []
for mk in MKTS:
    for s in SYMS:
        for dt in DL:
            for mo in range(1, 13):
                fn = f"{s}-1m-2024-{mo:02d}.zip" if dt=="klines" else f"{s}-{DS.get(dt,dt)}-2024-{mo:02d}.zip"
                url = burl(mk, dt, s, fn)
                td = config.raw_dir / mk / s / dt / ("1m" if dt=="klines" else "")
                tasks.append((url, td / fn))

total = len(tasks)
print(f"  {total} items", flush=True)
stats = {"ok":0, "cached":0, "miss":0, "err":0}

# Process sequentially with per-file timeout
for i, (url, dest) in enumerate(tasks):
    if dest.exists() and dest.stat().st_size > 0:
        stats["cached"] += 1
        continue
    with ThreadPoolExecutor(1) as pool:
        try:
            fut = pool.submit(dl_one, (url, dest))
            result = fut.result(timeout=45)
            if result == "ok":
                stats["ok"] += 1
            elif result == "miss":
                stats["miss"] += 1
            else:
                stats["err"] += 1
        except FuturesTimeout:
            stats["err"] += 1
        except Exception:
            stats["err"] += 1

    done = i + 1
    if done % 30 == 0 or done == total:
        print(f"  [{done}/{total}] ok={stats['ok']} c={stats['cached']} miss={stats['miss']} err={stats['err']}", flush=True)

elapsed = time.time() - t0
print(f"  DONE: {stats['ok']} new, {stats['cached']} cached, {stats['miss']} missing, {stats['err']} errors ({elapsed:.0f}s)", flush=True)

# Count raw files
raw_count = sum(1 for _ in Path("D:/emptyu/storage/raw").rglob("*.zip"))
raw_size = sum(f.stat().st_size for f in Path("D:/emptyu/storage/raw").rglob("*.zip"))
print(f"  Total raw: {raw_count} files, {raw_size/(1024*1024*1024):.2f} GB", flush=True)
