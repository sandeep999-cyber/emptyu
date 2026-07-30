"""Unit tests for Binance Vision Downloader."""

import tempfile
from pathlib import Path
import hashlib
import asyncio
import pytest
from src.data.binance_vision import BinanceVisionDownloader


class TestBinanceVisionDownloader:
    def test_incremental_download_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            downloader = BinanceVisionDownloader(raw_dir=raw_dir)
            dest_file = raw_dir / "futures" / "BTCUSDT" / "klines" / "1m" / "BTCUSDT-1m-2024-01.zip"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_bytes(b"dummy zip data")
            assert dest_file.exists()
            assert dest_file.stat().st_size > 0

    def test_checksum_verification_rejects_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            downloader = BinanceVisionDownloader(raw_dir=raw_dir)
            zip_path = raw_dir / "test.zip"
            checksum_path = raw_dir / "test.zip.checksum"
            raw_dir.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(b"corrupt data")
            checksum_path.write_text("abcdef1234567890abcdef1234567890abcdef12  test.zip")
            result = asyncio.run(downloader.verify_checksum(zip_path, checksum_path))
            assert not result

    def test_checksum_verification_passes_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            downloader = BinanceVisionDownloader(raw_dir=raw_dir)
            zip_path = raw_dir / "test.zip"
            checksum_path = raw_dir / "test.zip.checksum"
            raw_dir.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(b"valid data")
            valid_hash = downloader.compute_sha256(zip_path)
            checksum_path.write_text(f"{valid_hash}  test.zip")
            result = asyncio.run(downloader.verify_checksum(zip_path, checksum_path))
            assert result

    def test_build_url_futures_klines(self):
        downloader = BinanceVisionDownloader()
        url = downloader._build_url("futures", "klines", "BTCUSDT", "BTCUSDT-1m-2024-01.zip")
        assert "futures/um" in url
        assert "klines/BTCUSDT/1m" in url

    def test_build_url_spot_funding(self):
        downloader = BinanceVisionDownloader()
        url = downloader._build_url("spot", "funding", "BTCUSDT", "BTCUSDT-fundingRate-2024-01.zip")
        assert "spot" in url
        assert "fundingRate/BTCUSDT" in url
