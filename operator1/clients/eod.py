"""EOD Historical Data API client.

Drop-in alternative to the Eulerpool client.  Provides the same public
interface (get_profile, get_quotes, get_income_statement, ...) so the
rest of the pipeline can use either provider transparently.

API docs: https://eodhd.com/financial-apis/
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.constants import EOD_BASE_URL
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal ISIN -> ticker.exchange cache (in-memory, per-session)
# ---------------------------------------------------------------------------
_isin_symbol_cache: dict[str, str] = {}


class EODAPIError(Exception):
    """Raised on EOD-specific API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"EOD API error on {endpoint}: {detail}")


class EODClient:
    """Thin wrapper around the EOD Historical Data REST API.

    Mirrors the ``EulerportClient`` interface so the pipeline can swap
    providers without touching downstream code.

    Parameters
    ----------
    api_key:
        EOD Historical Data API token.
    base_url:
        Override for testing.
    """

    def __init__(self, api_key: str, base_url: str = EOD_BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, extra_params: dict[str, str] | None = None) -> Any:
        """Execute a cached GET with the API token injected."""
        url = f"{self._base_url}{path}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_token={self._api_key}&fmt=json"
        if extra_params:
            for k, v in extra_params.items():
                url = f"{url}&{k}={v}"
        try:
            return cached_get(url)
        except HTTPError as exc:
            raise EODAPIError(path, str(exc)) from exc

    @staticmethod
    def _to_dataframe(records: list[dict], date_col: str | None = None) -> pd.DataFrame:
        """Convert a list of dicts to a DataFrame, optionally parsing a date column."""
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # ISIN resolution
    # ------------------------------------------------------------------

    def _resolve_isin(self, isin: str) -> str:
        """Resolve an ISIN to ``TICKER.EXCHANGE`` via the EOD search API.

        Results are cached in-memory for the session so repeated calls
        for the same ISIN don't burn API quota.
        """
        if isin in _isin_symbol_cache:
            return _isin_symbol_cache[isin]

        candidates = self._get(f"/search/{isin}", extra_params={"type": "stock"})
        if not isinstance(candidates, list) or not candidates:
            # Fallback: try using the ISIN directly as a code
            logger.warning(
                "EOD search returned no results for ISIN %s -- using raw ISIN as code",
                isin,
            )
            _isin_symbol_cache[isin] = isin
            return isin

        # Prefer exact ISIN match
        for c in candidates:
            if (c.get("ISIN") or "").upper() == isin.upper():
                code = f"{c['Code']}.{c['Exchange']}"
                _isin_symbol_cache[isin] = code
                logger.debug("Resolved ISIN %s -> %s", isin, code)
                return code

        # Fall back to the first result
        first = candidates[0]
        code = f"{first.get('Code', isin)}.{first.get('Exchange', 'US')}"
        _isin_symbol_cache[isin] = code
        logger.debug("Resolved ISIN %s -> %s (first result)", isin, code)
        return code

    # ------------------------------------------------------------------
    # Profile (mirrors EulerportClient.get_profile)
    # ------------------------------------------------------------------

    def get_profile(self, isin: str) -> dict[str, Any]:
        """Fetch equity profile for *isin*.

        Returns a dict with the same keys as EulerportClient: isin,
        ticker, exchange, currency, country, sector, industry,
        sub_industry, name.
        """
        code = self._resolve_isin(isin)
        data = self._get(f"/fundamentals/{code}")

        if not isinstance(data, dict):
            raise EODAPIError(
                f"/fundamentals/{code}", "Unexpected response format"
            )

        general = data.get("General", {})

        return {
            "isin": general.get("ISIN", isin),
            "ticker": general.get("Code") or code.split(".")[0],
            "name": general.get("Name") or general.get("CompanyName"),
            "exchange": general.get("Exchange"),
            "currency": general.get("CurrencyCode") or general.get("Currency"),
            "country": general.get("CountryISO") or general.get("Country"),
            "sector": general.get("Sector"),
            "industry": general.get("Industry"),
            "sub_industry": general.get("GicSubIndustry"),
        }

    # ------------------------------------------------------------------
    # Quotes (mirrors EulerportClient.get_quotes)
    # ------------------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Fetch daily quote series for *identifier* (ISIN or ticker).

        Returns DataFrame with columns: date, open, high, low, close,
        volume, adjusted_close.
        """
        code = self._resolve_isin(identifier)
        data = self._get(f"/eod/{code}")

        if not isinstance(data, list):
            data = []

        df = self._to_dataframe(data, date_col="date")
        if df.empty:
            return df

        # Normalise column names to match Eulerpool schema
        col_map = {
            "adjusted_close": "adjusted_close",
            "close": "close",
            "high": "high",
            "low": "low",
            "open": "open",
            "volume": "volume",
            "date": "date",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        return df

    # ------------------------------------------------------------------
    # Financial statements (mirrors EulerportClient)
    # ------------------------------------------------------------------

    def _get_financial(self, isin: str, statement_key: str) -> pd.DataFrame:
        """Generic fetcher for financial statement data from fundamentals."""
        code = self._resolve_isin(isin)
        data = self._get(f"/fundamentals/{code}")

        if not isinstance(data, dict):
            return pd.DataFrame()

        financials = data.get("Financials", {})
        statement_data = financials.get(statement_key, {})

        # EOD returns statements as {period_key: {field: value, ...}}
        # We need to flatten into a list of dicts
        if isinstance(statement_data, dict):
            # Could be nested by period type (quarterly/yearly)
            records: list[dict[str, Any]] = []
            for period_type in ("quarterly", "yearly"):
                period_data = statement_data.get(period_type, {})
                if isinstance(period_data, dict):
                    for _date_key, row in period_data.items():
                        if isinstance(row, dict):
                            row_copy = dict(row)
                            row_copy.setdefault("report_date", row_copy.get("date"))
                            records.append(row_copy)
            if not records:
                # Maybe it's a flat dict of {date: {...}}
                for _date_key, row in statement_data.items():
                    if isinstance(row, dict):
                        row_copy = dict(row)
                        row_copy.setdefault("report_date", row_copy.get("date"))
                        records.append(row_copy)
            return self._to_dataframe(records, date_col="report_date")

        return pd.DataFrame()

    def get_income_statement(self, isin: str) -> pd.DataFrame:
        """Fetch periodic income statement for *isin*."""
        return self._get_financial(isin, "Income_Statement")

    def get_balance_sheet(self, isin: str) -> pd.DataFrame:
        """Fetch periodic balance sheet for *isin*."""
        return self._get_financial(isin, "Balance_Sheet")

    def get_cashflow_statement(self, isin: str) -> pd.DataFrame:
        """Fetch periodic cash-flow statement for *isin*."""
        return self._get_financial(isin, "Cash_Flow")

    # ------------------------------------------------------------------
    # Target-only extras (mirrors EulerportClient)
    # ------------------------------------------------------------------

    def get_peers(self, isin: str) -> list[str]:
        """Return list of peer ISINs for *isin*.

        EOD doesn't have a direct peers endpoint, so we extract them
        from the fundamentals ``Holders`` / ``General`` section, or
        fall back to an empty list.
        """
        code = self._resolve_isin(isin)
        try:
            data = self._get(f"/fundamentals/{code}")
        except EODAPIError:
            return []

        if not isinstance(data, dict):
            return []

        # Try to extract peers from ETF or General sections
        general = data.get("General", {})
        peers_raw = general.get("Peers", [])
        if isinstance(peers_raw, list):
            return peers_raw

        # Fallback: no direct peer data available from EOD
        logger.debug("No peers data available from EOD for %s", isin)
        return []

    def get_supply_chain(self, isin: str) -> list[dict[str, Any]]:
        """Return supply-chain relationships for *isin*.

        EOD doesn't have a dedicated supply-chain endpoint.
        Returns an empty list as a graceful degradation.
        """
        logger.debug("Supply chain data not available via EOD for %s", isin)
        return []

    def get_executives(self, isin: str) -> list[dict[str, Any]]:
        """Return executives list for *isin*."""
        code = self._resolve_isin(isin)
        try:
            data = self._get(f"/fundamentals/{code}")
        except EODAPIError:
            return []

        if not isinstance(data, dict):
            return []

        # Extract officer data from General section
        general = data.get("General", {})
        officers = general.get("Officers", {})

        if isinstance(officers, dict):
            return list(officers.values())
        if isinstance(officers, list):
            return officers

        return []

    # ------------------------------------------------------------------
    # Search (mirrors EulerportClient.search)
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search EOD for entities matching *query*.

        Returns list of dicts with: name, ticker, isin, exchange,
        country, sector, market_cap.
        """
        data = self._get(f"/search/{query}", extra_params={"type": "stock"})
        if not isinstance(data, list):
            data = []

        results: list[dict[str, Any]] = []
        for item in data:
            results.append({
                "name": item.get("Name"),
                "ticker": item.get("Code"),
                "isin": item.get("ISIN"),
                "exchange": item.get("Exchange"),
                "country": item.get("Country"),
                "sector": None,  # search endpoint doesn't return sector
                "market_cap": None,  # search endpoint doesn't return market_cap
            })
        return results
