"""Financial Modeling Prep (FMP) API client.

Authoritative source for OHLCV price data.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from operator1.constants import FMP_BASE_URL
from operator1.http_utils import cached_get, inject_api_key, HTTPError

logger = logging.getLogger(__name__)


class FMPAPIError(Exception):
    """Raised on FMP-specific API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"FMP API error on {endpoint}: {detail}")


class FMPClient:
    """Thin wrapper around the FMP REST API.

    Parameters
    ----------
    api_key:
        FMP API key (from secrets).
    base_url:
        Override for testing.
    """

    def __init__(self, api_key: str, base_url: str = FMP_BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build a full URL with API key injected."""
        raw = f"{self._base_url}{path}"
        return inject_api_key(raw, self._api_key)

    def _get(self, path: str) -> Any:
        """Execute a cached GET and raise on failure."""
        url = self._url(path)
        try:
            return cached_get(url)
        except HTTPError as exc:
            raise FMPAPIError(path, str(exc)) from exc

    # ------------------------------------------------------------------
    # Quote verification (Sec 3)
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch a real-time quote for *symbol* to verify it exists.

        Raises ``FMPAPIError`` if symbol is invalid or API key is bad.
        """
        data = self._get(f"/quote/{symbol}")
        if isinstance(data, list):
            if not data:
                raise FMPAPIError(
                    f"/quote/{symbol}",
                    f"Invalid symbol or empty response for '{symbol}'",
                )
            return data[0]
        if isinstance(data, dict) and data.get("Error Message"):
            raise FMPAPIError(f"/quote/{symbol}", data["Error Message"])
        return data

    # ------------------------------------------------------------------
    # Daily OHLCV (Sec 6) -- authoritative price source
    # ------------------------------------------------------------------

    def get_daily_ohlcv(
        self,
        symbol: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch 2-year daily OHLCV for *symbol*.

        Parameters
        ----------
        symbol:
            FMP ticker symbol.
        from_date, to_date:
            Date range (inclusive).  Passed as query parameters.

        Returns
        -------
        DataFrame with columns: date, open, high, low, close, volume.
        Sorted ascending by date.
        """
        path = f"/historical-price-full/{symbol}"
        params_parts: list[str] = []
        if from_date:
            params_parts.append(f"from={from_date.isoformat()}")
        if to_date:
            params_parts.append(f"to={to_date.isoformat()}")

        if params_parts:
            path = f"{path}?{'&'.join(params_parts)}"

        data = self._get(path)

        # FMP wraps the series in a "historical" key
        records: list[dict] = []
        if isinstance(data, dict):
            records = data.get("historical", [])
        elif isinstance(data, list):
            records = data

        if not records:
            raise FMPAPIError(
                path,
                f"No OHLCV data returned for '{symbol}'. "
                "Check symbol validity and date range.",
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # Ensure canonical columns
        col_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "adjClose": "adjusted_close",
            "vwap": "vwap",
        }
        rename = {k: v for k, v in col_map.items() if k in df.columns and k != v}
        if rename:
            df = df.rename(columns=rename)

        return df

    # ------------------------------------------------------------------
    # Intraday (optional, Sec 6)
    # ------------------------------------------------------------------

    def get_intraday(
        self,
        symbol: str,
        interval: str = "1hour",
    ) -> pd.DataFrame:
        """Fetch intraday data for *symbol* (optional, for UI zoom).

        Parameters
        ----------
        interval:
            One of ``1min``, ``5min``, ``15min``, ``30min``, ``1hour``, ``4hour``.
        """
        data = self._get(f"/historical-chart/{interval}/{symbol}")
        if not isinstance(data, list):
            data = data.get("data", [])
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df
