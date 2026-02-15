"""Eulerpool API client.

Provides access to equity profiles, quotes, financial statements,
peers, supply chain, executives, and search.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.constants import EULERPOOL_BASE_URL
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)


class EulerportAPIError(Exception):
    """Raised on Eulerpool-specific API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"Eulerpool API error on {endpoint}: {detail}")


class EulerportClient:
    """Thin wrapper around the Eulerpool REST API.

    Parameters
    ----------
    api_key:
        Eulerpool API key (from secrets).
    base_url:
        Override for testing.
    """

    def __init__(self, api_key: str, base_url: str = EULERPOOL_BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        """Execute a cached GET and raise on failure."""
        url = f"{self._base_url}{path}"
        try:
            return cached_get(url, headers=self._headers)
        except HTTPError as exc:
            raise EulerportAPIError(path, str(exc)) from exc

    @staticmethod
    def _to_dataframe(records: list[dict], date_col: str | None = None) -> pd.DataFrame:
        """Convert a list of dicts to a DataFrame, optionally parsing a date column."""
        df = pd.DataFrame(records)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Profile (Sec 3, 8)
    # ------------------------------------------------------------------

    def get_profile(self, isin: str) -> dict[str, Any]:
        """Fetch equity profile for *isin*.

        Returns a dict with keys: isin, ticker, exchange, currency,
        country, sector, industry, sub_industry (if available), name.
        """
        data = self._get(f"/api/1/equity/profile/{isin}")
        if isinstance(data, list):
            if not data:
                raise EulerportAPIError(
                    f"/api/1/equity/profile/{isin}", "Empty response"
                )
            data = data[0]

        # Normalise to expected keys
        return {
            "isin": data.get("isin", isin),
            "ticker": data.get("ticker") or data.get("symbol"),
            "name": data.get("name") or data.get("companyName"),
            "exchange": data.get("exchange") or data.get("exchangeShortName"),
            "currency": data.get("currency"),
            "country": data.get("country"),
            "sector": data.get("sector"),
            "industry": data.get("industry"),
            "sub_industry": data.get("subIndustry") or data.get("sub_industry"),
        }

    # ------------------------------------------------------------------
    # Quotes (Sec 6, 8)
    # ------------------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Fetch daily quote series for *identifier* (ISIN or ticker).

        Returns DataFrame with columns: date, open, high, low, close,
        volume, adjusted_close, vwap, market_cap, shares_outstanding.
        """
        data = self._get(f"/api/1/equity/quotes/{identifier}")
        if not isinstance(data, list):
            data = data.get("quotes", data.get("data", []))
        return self._to_dataframe(data, date_col="date")

    # ------------------------------------------------------------------
    # Financial statements (Sec 6, 8)
    # ------------------------------------------------------------------

    def get_income_statement(self, isin: str) -> pd.DataFrame:
        """Fetch periodic income statement for *isin*."""
        data = self._get(f"/api/1/equity/incomestatement/{isin}")
        if not isinstance(data, list):
            data = data.get("data", [])
        return self._to_dataframe(data, date_col="report_date")

    def get_balance_sheet(self, isin: str) -> pd.DataFrame:
        """Fetch periodic balance sheet for *isin*."""
        data = self._get(f"/api/1/equity/balancesheet/{isin}")
        if not isinstance(data, list):
            data = data.get("data", [])
        return self._to_dataframe(data, date_col="report_date")

    def get_cashflow_statement(self, isin: str) -> pd.DataFrame:
        """Fetch periodic cash-flow statement for *isin*."""
        data = self._get(f"/api/1/equity/cashflowstatement/{isin}")
        if not isinstance(data, list):
            data = data.get("data", [])
        return self._to_dataframe(data, date_col="report_date")

    # ------------------------------------------------------------------
    # Target-only extras (Sec 6)
    # ------------------------------------------------------------------

    def get_peers(self, isin: str) -> list[str]:
        """Return list of peer ISINs for *isin*."""
        data = self._get(f"/api/1/equity/peers/{isin}")
        if isinstance(data, list):
            return data
        return data.get("peers", data.get("data", []))

    def get_supply_chain(self, isin: str) -> list[dict[str, Any]]:
        """Return supply-chain relationships for *isin*."""
        data = self._get(f"/api/1/equity/supply-chain/{isin}")
        if isinstance(data, list):
            return data
        return data.get("data", [])

    def get_executives(self, isin: str) -> list[dict[str, Any]]:
        """Return executives list for *isin*."""
        data = self._get(f"/api/1/equity/executives/{isin}")
        if isinstance(data, list):
            return data
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Search (Sec 5)
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search Eulerpool for entities matching *query*.

        Returns list of dicts with: name, ticker, isin, exchange, country.
        """
        data = self._get(f"/api/1/equity/search?query={query}")
        if not isinstance(data, list):
            data = data.get("results", data.get("data", []))
        results: list[dict[str, Any]] = []
        for item in data:
            results.append({
                "name": item.get("name") or item.get("companyName"),
                "ticker": item.get("ticker") or item.get("symbol"),
                "isin": item.get("isin"),
                "exchange": item.get("exchange") or item.get("exchangeShortName"),
                "country": item.get("country"),
                "sector": item.get("sector"),
                "market_cap": item.get("marketCap") or item.get("market_cap"),
            })
        return results
