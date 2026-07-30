"""Test Binance Vision URL patterns."""
import httpx

urls = [
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
    "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/metrics/BTCUSDT/BTCUSDT-metrics-2024-01.zip",
    "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
    "https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/liquidations/BTCUSDT/BTCUSDT-liquidations-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/depth/BTCUSDT/BTCUSDT-depth-2024-01.zip",
]

for url in urls:
    try:
        r = httpx.head(url, follow_redirects=True, timeout=15)
        short = url.split("/")[-1]
        print(f"  {r.status_code}  {short}")
    except Exception as e:
        short = url.split("/")[-1]
        print(f"  ERR  {short}: {e}")
