"""Binance REST API client for exchangeInfo, funding rates, and open interest."""

from typing import Any, Dict, List, Optional
import httpx
from src.config import config


class BinanceRESTClient:
    """REST Client for Binance Futures & Spot API."""

    def __init__(self):
        self.futures_url = config.download.get("rest_url_futures", "https://fapi.binance.com")
        self.spot_url = config.download.get("rest_url_spot", "https://api.binance.com")
        self.timeout = config.download.get("timeout_seconds", 30)

    def fetch_futures_exchange_info(self) -> Dict[str, Any]:
        """Fetch USD-M Futures exchangeInfo including contract specs and listing statuses."""
        url = f"{self.futures_url}/fapi/v1/exchangeInfo"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def fetch_spot_exchange_info(self) -> Dict[str, Any]:
        """Fetch Spot exchangeInfo."""
        url = f"{self.spot_url}/api/v3/exchangeInfo"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def get_all_symbols(self, market: str = "futures") -> List[Dict[str, Any]]:
        """Extract all active and delisted symbols from exchangeInfo."""
        if market == "futures":
            info = self.fetch_futures_exchange_info()
            symbols = info.get("symbols", [])
            results = []
            for s in symbols:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", {})
                results.append({
                    "symbol": s["symbol"],
                    "market_type": "futures",
                    "base_asset": s["baseAsset"],
                    "quote_asset": s["quoteAsset"],
                    "is_active": (s.get("status") == "TRADING"),
                    "listing_date": str(s.get("onboardDate", "")) if s.get("onboardDate") else None,
                    "delisting_date": None,
                    "contract_type": s.get("contractType", "PERPETUAL"),
                    "tick_size": float(price_filter.get("tickSize", 0.0)),
                    "step_size": float(lot_filter.get("stepSize", 0.0)),
                    "min_qty": float(lot_filter.get("minQty", 0.0)),
                    "contract_size": 1.0
                })
            return results
        else:
            info = self.fetch_spot_exchange_info()
            symbols = info.get("symbols", [])
            results = []
            for s in symbols:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", {})
                results.append({
                    "symbol": s["symbol"],
                    "market_type": "spot",
                    "base_asset": s["baseAsset"],
                    "quote_asset": s["quoteAsset"],
                    "is_active": (s.get("status") == "TRADING"),
                    "listing_date": str(s.get("onboardDate", "")) if s.get("onboardDate") else None,
                    "delisting_date": None,
                    "contract_type": "SPOT",
                    "tick_size": float(price_filter.get("tickSize", 0.0)),
                    "step_size": float(lot_filter.get("stepSize", 0.0)),
                    "min_qty": float(lot_filter.get("minQty", 0.0)),
                    "contract_size": 1.0
                })
            return results

    def fetch_funding_history(
        self, symbol: str, start_time: Optional[int] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Fetch funding rate history via REST fallback."""
        url = f"{self.futures_url}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def fetch_open_interest_hist(
        self, symbol: str, period: str = "5m", start_time: Optional[int] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Fetch historical open interest metrics (5m snapshots)."""
        url = f"{self.futures_url}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()


rest_client = BinanceRESTClient()
