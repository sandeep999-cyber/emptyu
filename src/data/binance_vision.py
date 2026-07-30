"""Async Downloader for Binance Vision Historical Archives."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiohttp
from src.config import config

# Mapping from pipeline dataset_type to Binance Vision path component
BINANCE_DATASET_PATHS = {
    "klines": "klines",
    "funding": "fundingRate",
    "fundingRate": "fundingRate",
    "open_interest": "metrics",
    "metrics": "metrics",
    "aggTrades": "aggTrades",
    "trades": "trades",
    "depth": "depth",
    "liquidations": "liquidations",
}


class BinanceVisionDownloader:
    """Downloader for Binance Vision data (data.binance.vision)."""

    def __init__(self, raw_dir: Optional[Path] = None):
        self.base_url = config.download.get("base_url", "https://data.binance.vision/data")
        self.raw_dir = raw_dir or config.raw_dir
        self.concurrency = config.download.get("concurrency_limit", 8)
        self.retry_attempts = config.download.get("retry_attempts", 5)

    def _resolve_dataset_path(self, dataset_type: str) -> str:
        """Map internal dataset_type to Binance Vision path component."""
        return BINANCE_DATASET_PATHS.get(dataset_type, dataset_type)

    def _dataset_is_monthly(self, dataset_type: str) -> bool:
        """Return whether the dataset uses monthly or daily archives."""
        monthly_types = {"klines", "funding", "fundingRate", "aggTrades", "trades"}
        return dataset_type in monthly_types

    def _build_url(
        self,
        market: str,
        dataset_type: str,
        symbol: str,
        filename: str,
        period: str = "monthly"
    ) -> str:
        """Construct Binance Vision archive URL."""
        market_path = "futures/um" if market == "futures" else "spot"
        ds_path = self._resolve_dataset_path(dataset_type)
        if dataset_type == "klines":
            return f"{self.base_url}/{market_path}/{period}/klines/{symbol}/1m/{filename}"
        else:
            return f"{self.base_url}/{market_path}/{period}/{ds_path}/{symbol}/{filename}"

    def compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest_path: Path
    ) -> bool:
        """Download file asynchronously with resume support and integrity check."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True

        temp_path = dest_path.with_suffix(".tmp")
        for attempt in range(self.retry_attempts):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(temp_path, "wb") as f:
                            while chunk := await response.content.read(65536):
                                f.write(chunk)
                        temp_path.replace(dest_path)
                        return True
                    elif response.status == 404:
                        return False
                    else:
                        # Any other status (403, 500, 502, etc.) is an error
                        if attempt == self.retry_attempts - 1:
                            return False
            except Exception:
                if attempt == self.retry_attempts - 1:
                    if temp_path.exists():
                        temp_path.unlink()
                    return False
                await asyncio.sleep(1.0 * (attempt + 1))
        return False

    async def verify_checksum(self, zip_path: Path, checksum_path: Path) -> bool:
        """Verify zip SHA256 against Binance .checksum file."""
        if not checksum_path.exists():
            return False
        try:
            expected = checksum_path.read_text().strip().split()[0]
            actual = self.compute_sha256(zip_path)
            return actual.lower() == expected.lower()
        except Exception:
            return False

    async def download_monthly_archive(
        self,
        market: str,
        symbol: str,
        dataset_type: str,
        year: int,
        month: int
    ) -> Optional[Path]:
        """Download monthly archive ZIP and .checksum, then verify integrity."""
        month_str = f"{month:02d}"
        if dataset_type == "klines":
            filename = f"{symbol}-1m-{year}-{month_str}.zip"
        else:
            ds_path = self._resolve_dataset_path(dataset_type)
            filename = f"{symbol}-{ds_path}-{year}-{month_str}.zip"

        url = self._build_url(market, dataset_type, symbol, filename, "monthly")
        checksum_url = f"{url}.checksum"

        target_dir = self.raw_dir / market / symbol / dataset_type / ("1m" if dataset_type == "klines" else "")
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / filename
        checksum_path = target_dir / f"{filename}.checksum"

        async with aiohttp.ClientSession() as session:
            success = await self.download_file(session, url, zip_path)
            if success:
                await self.download_file(session, checksum_url, checksum_path)
                # Verify checksum if available
                if checksum_path.exists() and checksum_path.stat().st_size > 0:
                    if not await self.verify_checksum(zip_path, checksum_path):
                        zip_path.unlink()
                        return None
                return zip_path
            return None


downloader = BinanceVisionDownloader()
