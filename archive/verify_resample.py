"""Verify resample aggregation math against 1m source data."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"
import duckdb

conn = duckdb.connect(":memory:")
base = "D:/emptyu/storage/canonical"
checks = []

for mk, sym in [("futures", "BTCUSDT"), ("spot", "ETHUSDT")]:
    p1 = f"{base}/{mk}/{sym}/klines/1m/2024-03.parquet"
    for tf, div in [("5m", 5), ("1h", 60), ("1d", 1440)]:
        pT = f"{base}/{mk}/{sym}/klines/{tf}/2024-03.parquet"
        # Recompute aggregation from 1m and compare to stored resampled file
        q = f"""
        WITH expected AS (
            SELECT
                CAST(FLOOR(CAST(timestamp AS DOUBLE) / ({div}*60000)) * ({div}*60000) AS BIGINT) AS ts,
                FIRST(open ORDER BY timestamp) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close ORDER BY timestamp) AS close,
                SUM(volume) AS volume
            FROM read_parquet('{p1}')
            GROUP BY 1
        ),
        actual AS (
            SELECT timestamp AS ts, open, high, low, close, volume
            FROM read_parquet('{pT}')
        ),
        joined AS (
            SELECT e.ts,
                   e.open - a.open AS d_open, e.high - a.high AS d_high,
                   e.low - a.low AS d_low, e.close - a.close AS d_close,
                   e.volume - a.volume AS d_vol
            FROM expected e JOIN actual a ON e.ts = a.ts
        )
        SELECT count(*) AS n,
               sum(abs(d_open)) AS e_open, sum(abs(d_high)) AS e_high,
               sum(abs(d_low)) AS e_low, sum(abs(d_close)) AS e_close,
               sum(abs(d_vol)) AS e_vol
        FROM joined
        """
        n, eo, eh, el, ec, ev = conn.execute(q).fetchone()
        ok = (eo or 0) == 0 and (eh or 0) == 0 and (el or 0) == 0 and (ec or 0) == 0 and (ev or 0) < 1e-6
        status = "PASS" if ok else "FAIL"
        checks.append(ok)
        print(f"  {mk} {sym} {tf}: {n} candles compared -> {status} "
              f"(err open={eo}, high={eh}, low={el}, close={ec}, vol={ev})", flush=True)

print("ALL AGGREGATION MATH:", "PASS" if all(checks) else "FAIL", flush=True)
