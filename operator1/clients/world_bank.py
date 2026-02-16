"""World Bank API client.

Fetches macroeconomic indicator data, country code mappings, and
Documents & Reports search via the WDS API (Phase E3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from operator1.constants import WORLD_BANK_BASE_URL, WB_WDS_BASE_URL, WB_CACHE_DIR
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

    # ------------------------------------------------------------------
    # Phase E3: World Bank Documents & Reports API (WDS)
    # ------------------------------------------------------------------

    def search_documents(
        self,
        country: str,
        *,
        sector_keywords: str | None = None,
        max_results: int = 10,
        newer_than: str | None = None,
        doc_types: list[str] | None = None,
        language: str = "English",
        timeout_s: int = 30,
    ) -> list[dict[str, Any]]:
        """Search the World Bank Documents & Reports API (WDS).

        Fetches a small, recent set of documents for a country to provide
        qualitative context (policy, macro stress, sector conditions).
        This is an additive source -- it does not change the core
        quantitative cache logic.

        Base endpoint: ``https://search.worldbank.org/api/v3/wds``

        Parameters
        ----------
        country:
            Country name or ISO code to search for.
        sector_keywords:
            Optional sector/industry keywords to narrow results.
        max_results:
            Maximum number of documents to return.
        newer_than:
            ISO date string (``YYYY-MM-DD``).  Only return documents
            newer than this date (uses ``strdate`` parameter).
        doc_types:
            Filter by document types (e.g. ``["Economic and Sector Work",
            "Country Economic Memorandum"]``).
        language:
            Language filter (default ``"English"``).
        timeout_s:
            Request timeout in seconds.

        Returns
        -------
        List of document metadata dicts with keys:
            ``title``, ``abstract``, ``doc_date``, ``doc_type``,
            ``url``, ``country``, ``topic``.
        Returns empty list on any failure.
        """
        try:
            # Build search query
            qterm_parts = [country]
            if sector_keywords:
                qterm_parts.append(sector_keywords)
            qterm = " ".join(qterm_parts)

            params: dict[str, str] = {
                "format": "json",
                "qterm": qterm,
                "rows": str(min(max_results, 50)),
                "os": "0",
                "fl": "display_title,abstracts,docdt,docty,url,count,topic",
                "lang_exact": language,
                "sort": "docdt",
                "order": "desc",
            }

            if newer_than:
                params["strdate"] = newer_than

            if doc_types:
                params["docty_exact"] = ",".join(doc_types)

            # Filter by country using count_exact
            if len(country) == 2:
                # ISO-2 code: convert to country name via our mapping
                try:
                    iso3 = self.iso2_to_iso3(country)
                    params["count_exact"] = iso3
                except WorldBankAPIError:
                    pass  # fall back to qterm search
            elif len(country) == 3 and country.isalpha():
                params["count_exact"] = country.upper()

            resp = requests.get(
                WB_WDS_BASE_URL,
                params=params,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()

            # Parse response
            documents: list[dict[str, Any]] = []

            # WDS response has a "documents" key with doc entries
            docs_data = data.get("documents", {})
            for doc_key, doc_info in docs_data.items():
                if doc_key == "facets" or not isinstance(doc_info, dict):
                    continue

                title = doc_info.get("display_title", "")
                if not title:
                    title = doc_info.get("title", {})
                    if isinstance(title, dict):
                        title = list(title.values())[0] if title else ""

                abstract = doc_info.get("abstracts", "")
                if isinstance(abstract, dict):
                    abstract = list(abstract.values())[0] if abstract else ""

                doc_date = doc_info.get("docdt", "")
                doc_type = doc_info.get("docty", "")
                url = doc_info.get("url", "")
                country_val = doc_info.get("count", "")
                topic = doc_info.get("topic", "")

                if isinstance(topic, dict):
                    topic = ", ".join(topic.values()) if topic else ""
                if isinstance(country_val, dict):
                    country_val = ", ".join(country_val.values()) if country_val else ""

                documents.append({
                    "title": str(title)[:500],
                    "abstract": str(abstract)[:1000],
                    "doc_date": str(doc_date),
                    "doc_type": str(doc_type),
                    "url": str(url),
                    "country": str(country_val),
                    "topic": str(topic),
                })

                if len(documents) >= max_results:
                    break

            logger.info(
                "WDS search for '%s': found %d documents",
                qterm, len(documents),
            )
            return documents

        except Exception as exc:
            logger.warning("WDS document search failed: %s", exc)
            return []

    def get_country_context(
        self,
        country: str,
        sector: str | None = None,
        *,
        max_docs: int = 5,
        years_back: int = 3,
    ) -> dict[str, Any]:
        """Fetch lightweight country-specific qualitative context.

        Combines WDS document metadata into a structured context dict
        suitable for feeding into the linked/macro context layer.

        Parameters
        ----------
        country:
            Country name or ISO code.
        sector:
            Optional sector keyword for more targeted search.
        max_docs:
            Maximum documents to retrieve.
        years_back:
            Only fetch documents from the last N years.

        Returns
        -------
        Dict with ``documents``, ``document_count``, ``topic_distribution``,
        and ``doc_type_distribution`` keys.
        """
        from datetime import date, timedelta

        newer_than = (date.today() - timedelta(days=365 * years_back)).isoformat()

        docs = self.search_documents(
            country,
            sector_keywords=sector,
            max_results=max_docs,
            newer_than=newer_than,
        )

        # Compute lightweight text features
        topic_counts: dict[str, int] = {}
        doctype_counts: dict[str, int] = {}

        for doc in docs:
            for topic in str(doc.get("topic", "")).split(","):
                topic = topic.strip()
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            dt = str(doc.get("doc_type", "")).strip()
            if dt:
                doctype_counts[dt] = doctype_counts.get(dt, 0) + 1

        return {
            "documents": docs,
            "document_count": len(docs),
            "topic_distribution": topic_counts,
            "doc_type_distribution": doctype_counts,
            "search_country": country,
            "search_sector": sector,
        }
