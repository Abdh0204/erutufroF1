"""EOD Historical Data API client.

Drop-in alternative to the Eulerpool client.  Provides the same public
interface (get_profile, get_quotes, get_income_statement, ...) so the
rest of the pipeline can use either provider transparently.

API docs: https://eodhd.com/financial-apis/

Key differences from Eulerpool handled internally:
- EODHD uses ``TICKER.EXCHANGE`` identifiers instead of ISINs.
  ISINs are resolved via the ``/search/`` endpoint with session caching.
- Financial statements live under a single ``/fundamentals/`` response,
  so we cache the full response per-entity to avoid redundant API calls.
- Column names differ from Eulerpool/FMP; normalisation is applied so
  the cache builder's ``_build_column_rename_map`` can do the rest.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.constants import EOD_BASE_URL
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory session caches
# ---------------------------------------------------------------------------
_isin_symbol_cache: dict[str, str] = {}
_fundamentals_cache: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# EOD -> canonical column name mapping
# ---------------------------------------------------------------------------
# Maps EODHD field names to the camelCase names already handled by
# cache_builder._build_column_rename_map.  This ensures the financial
# statement DataFrames produced by EODClient can flow through the
# existing pipeline without modification.
_EOD_FIELD_MAP: dict[str, str] = {
    # Income Statement
    "totalRevenue": "totalRevenue",
    "grossProfit": "grossProfit",
    "operatingIncome": "operatingIncome",
    "ebitda": "ebitda",
    "netIncome": "netIncome",
    "interestExpense": "interestExpense",
    "incomeTaxExpense": "incomeTaxExpense",
    # Balance Sheet -- EOD variants
    "totalAssets": "totalAssets",
    "totalLiab": "totalLiabilities",
    "totalCurrentLiabilities": "totalCurrentLiabilities",
    "totalCurrentAssets": "totalCurrentAssets",
    "totalStockholderEquity": "totalStockholdersEquity",
    "shortTermDebt": "shortTermDebt",
    "longTermDebt": "longTermDebt",
    "longTermDebtTotal": "longTermDebt",
    "cash": "cashAndCashEquivalents",
    "cashAndShortTermInvestments": "cashAndShortTermInvestments",
    "cashAndEquivalents": "cashAndCashEquivalents",
    "netReceivables": "netReceivables",
    "shortLongTermDebt": "shortTermDebt",
    # Cash Flow -- EOD variants
    "totalCashFromOperatingActivities": "operatingCashFlow",
    "capitalExpenditures": "capitalExpenditure",
    "totalCashflowsFromInvestingActivities": "investingActivitiesCashflow",
    "totalCashFromFinancingActivities": "financingActivitiesCashflow",
    "dividendsPaid": "dividendsPaid",
    "paymentOfDividends": "dividendsPaid",
    # Common fields already matching
    "sharesOutstanding": "sharesOutstanding",
    "marketCapitalization": "marketCap",
}


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
        """Execute a cached GET with the API token injected.

        The ``api_token`` and ``fmt=json`` parameters are appended
        automatically.  Additional query parameters can be passed via
        *extra_params*.
        """
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

    def _get_fundamentals(self, code: str) -> dict[str, Any]:
        """Fetch fundamentals for *code*, with per-session in-memory caching.

        The EODHD ``/fundamentals/`` endpoint returns profile, financials,
        officers, and peers in a single response.  Multiple methods
        (get_profile, get_income_statement, get_peers, ...) all need this
        data, so we cache the parsed response per ``code`` to avoid
        redundant HTTP round-trips and disk-cache lookups.
        """
        if code in _fundamentals_cache:
            return _fundamentals_cache[code]

        data = self._get(f"/fundamentals/{code}")
        if not isinstance(data, dict):
            data = {}

        _fundamentals_cache[code] = data
        return data

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

    @staticmethod
    def _normalise_eod_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename EOD-specific column names to match the canonical names
        that ``cache_builder._build_column_rename_map`` already handles.

        This is the critical bridge that allows EOD data to flow through
        the same pipeline as Eulerpool data.
        """
        rename = {k: v for k, v in _EOD_FIELD_MAP.items()
                  if k in df.columns and k != v}
        if rename:
            df = df.rename(columns=rename)
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

        try:
            candidates = self._get(f"/search/{isin}", extra_params={"type": "stock"})
        except EODAPIError:
            logger.warning("EOD search failed for %s -- using raw ISIN as code", isin)
            _isin_symbol_cache[isin] = isin
            return isin

        if not isinstance(candidates, list) or not candidates:
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
        data = self._get_fundamentals(code)

        if not data:
            raise EODAPIError(
                f"/fundamentals/{code}", "Empty or non-dict response"
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
        volume, adjusted_close, market_cap, shares_outstanding.

        Note: ``vwap`` is not available from EOD's daily endpoint.
        ``market_cap`` and ``shares_outstanding`` are sourced from the
        fundamentals ``General`` section and applied as static values
        (they don't change daily but having them is better than leaving
        them null for downstream survival mode and valuation calculations).
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

        # Enrich with market_cap and shares_outstanding from fundamentals.
        # These are static snapshots (not daily) but are needed by the
        # cache builder for country_protection (market_cap > 0.1% GDP)
        # and valuation ratios (pe_ratio, ps_ratio, enterprise_value).
        try:
            fund_data = self._get_fundamentals(code)
            general = fund_data.get("General", {}) if fund_data else {}
            highlights = fund_data.get("Highlights", {}) if fund_data else {}

            mkt_cap = highlights.get("MarketCapitalization") or general.get(
                "MarketCapitalization"
            )
            shares = general.get("SharesOutstanding") or highlights.get(
                "SharesOutstanding"
            )

            if mkt_cap is not None:
                df["market_cap"] = float(mkt_cap)
            if shares is not None:
                df["shares_outstanding"] = float(shares)
        except Exception as exc:
            logger.debug(
                "Could not enrich quotes with market_cap/shares: %s", exc
            )

        return df

    # ------------------------------------------------------------------
    # Financial statements (mirrors EulerportClient)
    # ------------------------------------------------------------------

    def _get_financial(self, isin: str, statement_key: str) -> pd.DataFrame:
        """Generic fetcher for financial statement data from fundamentals.

        The EODHD ``/fundamentals/`` response nests financial data as::

            Financials -> <statement_key> -> quarterly -> { date_key: {row} }
            Financials -> <statement_key> -> yearly    -> { date_key: {row} }

        We flatten both period types into a single list of records,
        normalise column names via ``_normalise_eod_columns``, and
        return a DataFrame sorted by ``report_date``.
        """
        code = self._resolve_isin(isin)
        data = self._get_fundamentals(code)

        if not data:
            return pd.DataFrame()

        financials = data.get("Financials", {})
        statement_data = financials.get(statement_key, {})

        if not isinstance(statement_data, dict):
            return pd.DataFrame()

        records: list[dict[str, Any]] = []

        # Try nested quarterly/yearly structure first
        for period_type in ("quarterly", "yearly"):
            period_data = statement_data.get(period_type, {})
            if isinstance(period_data, dict):
                for _date_key, row in period_data.items():
                    if isinstance(row, dict):
                        row_copy = dict(row)
                        # Ensure report_date is set from the ``date`` field
                        row_copy.setdefault("report_date", row_copy.get("date"))
                        records.append(row_copy)

        # Fallback: flat dict of {date: {...}} (some endpoints)
        if not records:
            for _date_key, row in statement_data.items():
                if isinstance(row, dict):
                    row_copy = dict(row)
                    row_copy.setdefault("report_date", row_copy.get("date"))
                    records.append(row_copy)

        df = self._to_dataframe(records, date_col="report_date")
        if not df.empty:
            df = self._normalise_eod_columns(df)
        return df

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

        EOD doesn't have a dedicated peers endpoint.  The ``General``
        section of fundamentals may contain a ``Peers`` list; if not,
        returns an empty list (the pipeline's peer-fallback logic in
        ``entity_discovery`` handles this gracefully).
        """
        code = self._resolve_isin(isin)
        try:
            data = self._get_fundamentals(code)
        except EODAPIError:
            return []

        if not data:
            return []

        general = data.get("General", {})
        peers_raw = general.get("Peers", [])
        if isinstance(peers_raw, list):
            return peers_raw

        logger.debug("No peers data available from EOD for %s", isin)
        return []

    def get_supply_chain(self, isin: str) -> list[dict[str, Any]]:
        """Return supply-chain relationships for *isin*.

        EOD doesn't have a dedicated supply-chain endpoint.
        Returns an empty list -- the pipeline handles missing supply
        chain data gracefully.
        """
        logger.debug("Supply chain data not available via EOD for %s", isin)
        return []

    def get_executives(self, isin: str) -> list[dict[str, Any]]:
        """Return executives list for *isin*.

        Extracted from the ``General.Officers`` section of the
        fundamentals response.
        """
        code = self._resolve_isin(isin)
        try:
            data = self._get_fundamentals(code)
        except EODAPIError:
            return []

        if not data:
            return []

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
        country, sector, market_cap -- matching the schema returned
        by ``EulerportClient.search``.
        """
        try:
            data = self._get(f"/search/{query}", extra_params={"type": "stock"})
        except EODAPIError:
            return []

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
