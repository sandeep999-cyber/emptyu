"""Download daily open-interest metrics for 2024, futures symbols."""
import sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import httpx
from src.config import config

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BASE = "https://data.binance.vision/data"

import datetime as dt
dates = []
d = dt.date(2024, 1, 1)
while d <= dt.date(2024, 12, 31):
    dates.append(d)
    d += dt.timedelta(days=1)

def url_for(sym, date):
    fn = f"{sym}-metrics-{date.isoformat()}.zip"
    return f"{BASE}/futures/um/daily/metrics/{sym}/{fn}", fn

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
                        time.sleep(1)
                        continue
                    with open(tmp, "wb") as f:
                        for c in r.iter_bytes(65536):
                            f.write(c)
                    tmp.replace(dest)
                    return "ok"
        except Exception:
            tmp.unlink(missing_ok=True)
            time.sleep(1)
    return "err"

tasks = []
for sym in SYMS:
    for date in dates:
        url, fn = url_for(sym, date)
        dest = config.raw_dir / "futures" / sym / "open_interest" / "daily" / fn
        tasks.append((url, dest))

total = len(tasks)
print(f"=== DAILY OI DOWNLOAD: {total} items ===", flush=True)
t0 = time.time()
stats = {}
done = 0
with ThreadPoolExecutor(max_workers=6) as pool:
    futs = {pool.submit(dl_one, url, dest): dest for url, dest in tasks}
    for fut in as_completed(futs):
        r = fut.result()
        stats[r] = stats.get(r, 0) + 1
        done += 1
        if done % 100 == 0 or done == total:
            elapsed = time.time() - t0
            eta = (total - done) / max(done / max(elapsed, 1), 0.01)
            print(f"  [{done}/{total}] ok={stats.get('ok',0)} cached={stats.get('cached',0)} "
                  f"miss={stats.get('miss',0)} err={stats.get('err',0)} ETA={eta:.0f}s", flush=True)

print(f"DONE in {time.time()-t0:.0f}s: {stats}", flush=True)
