"""World Bank API client.

Fetches macroeconomic indicator data and country code mappings.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.constants import WORLD_BANK_BASE_URL, WB_CACHE_DIR
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# In-memory cache for the country mapping (ISO-2 -> ISO-3).
_country_map: dict[str, str] | None = None


class WorldBankAPIError(Exception):
    """Raised on World Bank API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"World Bank API error on {endpoint}: {detail}")


class WorldBankClient:
    """Thin wrapper around the World Bank REST API (v2, JSON format).

    Parameters
    ----------
    base_url:
        Override for testing.
    api_key:
        Optional gateway API key (most endpoints are open).
    """

    def __init__(
        self,
        base_url: str = WORLD_BANK_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Cached GET against the World Bank API."""
        url = f"{self._base_url}{path}"
        merged_params = {"format": "json", "per_page": "500"}
        if params:
            merged_params.update(params)
        try:
            return cached_get(url, params=merged_params, cache_dir=WB_CACHE_DIR)
        except HTTPError as exc:
            raise WorldBankAPIError(path, str(exc)) from exc

    # ------------------------------------------------------------------
    # Country mapping (Sec 4)
    # ------------------------------------------------------------------

    def get_countries(self) -> pd.DataFrame:
        """Fetch the full World Bank country list.

        Returns DataFrame with columns: iso2, iso3, name, region, income_level.
        Result is cached to disk after first call.
        """
        data = self._get("/country")
        # World Bank v2 returns [metadata, records]
        records = data[1] if isinstance(data, list) and len(data) > 1 else data
        if not isinstance(records, list):
            records = []

        rows: list[dict[str, Any]] = []
        for item in records:
            rows.append({
                "iso2": item.get("iso2Code"),
                "iso3": item.get("id"),
                "name": item.get("name"),
                "region": (item.get("region") or {}).get("value"),
                "income_level": (item.get("incomeLevel") or {}).get("value"),
            })

        return pd.DataFrame(rows)

    def iso2_to_iso3(self, country_iso2: str) -> str:
        """Convert a 2-letter country code to the World Bank 3-letter code.

        Raises ``WorldBankAPIError`` if the mapping cannot be found.
        """
        global _country_map
        if _country_map is None:
            df = self.get_countries()
            _country_map = dict(zip(df["iso2"].str.upper(), df["iso3"].str.upper()))

        iso3 = _country_map.get(country_iso2.upper())
        if iso3 is None:
            raise WorldBankAPIError(
                "/country",
                f"No ISO-3 mapping found for ISO-2 code '{country_iso2}'",
            )
        return iso3

    # ------------------------------------------------------------------
    # Indicator data (Sec 11)
    # ------------------------------------------------------------------

    def get_indicator(
        self,
        country_iso3: str,
        indicator_code: str,
        from_year: int | None = None,
        to_year: int | None = None,
    ) -> pd.DataFrame:
        """Fetch a single indicator series for a country.

        Parameters
        ----------
        country_iso3:
            3-letter country code (e.g. ``USA``).
        indicator_code:
            World Bank indicator ID (e.g. ``FP.CPI.TOTL.ZG``).
        from_year, to_year:
            Year range (inclusive).

        Returns
        -------
        DataFrame with columns: year (int), value (float or NaN).
        """
        path = f"/country/{country_iso3}/indicator/{indicator_code}"
        params: dict[str, str] = {}
        if from_year is not None:
            params["date"] = f"{from_year}:{to_year or from_year}"

        data = self._get(path, params=params)

        # World Bank v2 paginates: [metadata, records]
        records = data[1] if isinstance(data, list) and len(data) > 1 else data
        if records is None:
            records = []
        if not isinstance(records, list):
            records = []

        rows: list[dict[str, Any]] = []
        for item in records:
            try:
                year = int(item.get("date", 0))
            except (ValueError, TypeError):
                continue
            value = item.get("value")
            if value is not None:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = float("nan")
            else:
                value = float("nan")
            rows.append({"year": year, "value": value})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("year").reset_index(drop=True)
        return df
