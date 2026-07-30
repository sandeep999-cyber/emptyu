"""Test direct download speed."""
import httpx
import time

url = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip"
dest = "D:/emptyu/test_download.zip"

t0 = time.time()
with httpx.Client(timeout=120, follow_redirects=True) as cl:
    with cl.stream("GET", url) as resp:
        print("Status:", resp.status_code)
        print("Content-Length:", resp.headers.get("content-length", "unknown"))
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(65536):
                f.write(chunk)
                total += len(chunk)
        print("Downloaded:", total, "bytes in", time.time() - t0, "seconds")
        print("Speed:", total / (1024*1024) / max(0.1, time.time()-t0), "MB/s")
