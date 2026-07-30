"""Test downloader with debug output."""
import asyncio
import sys
import aiohttp
from pathlib import Path
sys.path.insert(0, ".")

async def test():
    from src.config import config
    from src.data.binance_vision import downloader
    
    # Test the URL directly
    market = "futures"
    symbol = "BTCUSDT"
    dataset_type = "klines"
    year, month = 2024, 1
    
    filename = f"{symbol}-1m-{year}-{month:02d}.zip"
    url = downloader._build_url(market, dataset_type, symbol, filename, "monthly")
    print(f"URL: {url}")
    
    # Test with aiohttp directly
    async with aiohttp.ClientSession() as session:
        print(f"Attempting GET...")
        async with session.get(url) as response:
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}")
            if response.status == 200:
                target_dir = downloader.raw_dir / market / symbol / dataset_type / "1m"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / filename
                print(f"Downloading to: {target_path}")
                with open(target_path, "wb") as f:
                    total = 0
                    while True:
                        chunk = await response.content.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                print(f"Downloaded {total} bytes")
                print(f"File exists: {target_path.exists()}, Size: {target_path.stat().st_size}")
    
    # Now test the downloader's method
    print("\nTesting downloader.download_monthly_archive...")
    result = await downloader.download_monthly_archive("futures", "BTCUSDT", "klines", 2024, 1)
    print(f"Result: {result}")

asyncio.run(test())
