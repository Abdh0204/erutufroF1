"""Step 0.2 -- World Bank macro data mapping and fetching.

Loads the canonical variable-to-indicator mapping from config, converts
the target country code, and fetches all macro indicators for the
required date range.  Results are cached to disk to avoid re-fetching.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.config_loader import load_config, get_global_config
from operator1.constants import DATE_START, DATE_END, WB_CACHE_DIR
from operator1.clients.world_bank import WorldBankClient, WorldBankAPIError
from operator1.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)

# Extra buffer years -- fetch 1 year before DATE_START so as-of alignment
# never runs out of data at the start of the window.
_BUFFER_YEARS = 1


@dataclass
class MacroDataset:
    """Container for fetched World Bank macro data."""

    country_iso3: str
    indicators: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    gemini_suggestions: dict[str, str] = field(default_factory=dict)


def _indicator_cache_path(indicator_code: str) -> Path:
    """Return the disk cache path for a single indicator's raw data."""
    return Path(WB_CACHE_DIR) / f"{indicator_code.replace('.', '_')}.parquet"


def _is_cached(indicator_code: str) -> bool:
    """Check whether an indicator's data is already on disk."""
    return _indicator_cache_path(indicator_code).exists()


def _read_cached_indicator(indicator_code: str) -> pd.DataFrame:
    """Read a previously cached indicator DataFrame."""
    return pd.read_parquet(_indicator_cache_path(indicator_code))


def _write_cached_indicator(indicator_code: str, df: pd.DataFrame) -> None:
    """Persist an indicator DataFrame to disk."""
    path = _indicator_cache_path(indicator_code)
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path, index=False)


def fetch_macro_data(
    country_iso2: str,
    wb_client: WorldBankClient,
    gemini_client: GeminiClient | None = None,
    force_rebuild: bool | None = None,
    sector: str = "",
) -> MacroDataset:
    """Fetch all canonical World Bank indicators for a country.

    Parameters
    ----------
    country_iso2:
        2-letter ISO country code from the verified target profile.
    wb_client:
        Initialised World Bank API client.
    gemini_client:
        Optional Gemini client for advisory mapping suggestions.
    force_rebuild:
        Override for FORCE_REBUILD config flag.
    sector:
        Target sector hint for Gemini suggestions.

    Returns
    -------
    MacroDataset
        Contains fetched indicator DataFrames, list of missing indicators,
        and optional Gemini mapping suggestions.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    # ------------------------------------------------------------------
    # 1. Convert country code
    # ------------------------------------------------------------------
    logger.info("Converting country ISO-2 '%s' to ISO-3 ...", country_iso2)
    try:
        country_iso3 = wb_client.iso2_to_iso3(country_iso2)
    except WorldBankAPIError as exc:
        logger.error("Country code conversion failed: %s", exc)
        return MacroDataset(country_iso3="UNKNOWN", missing=["ALL"])

    logger.info("Country ISO-3: %s", country_iso3)

    # ------------------------------------------------------------------
    # 2. Load indicator mapping from config
    # ------------------------------------------------------------------
    indicator_map: dict[str, str] = load_config("world_bank_indicator_map")
    logger.info(
        "Loaded %d canonical indicators from config", len(indicator_map),
    )

    # ------------------------------------------------------------------
    # 3. Compute year range (with buffer)
    # ------------------------------------------------------------------
    from_year = DATE_START.year - _BUFFER_YEARS
    to_year = DATE_END.year

    # ------------------------------------------------------------------
    # 4. Fetch each indicator
    # ------------------------------------------------------------------
    dataset = MacroDataset(country_iso3=country_iso3)

    for canonical_name, indicator_code in indicator_map.items():
        # Check disk cache first
        if not force_rebuild and _is_cached(indicator_code):
            logger.debug("Cache HIT for indicator %s (%s)", canonical_name, indicator_code)
            df = _read_cached_indicator(indicator_code)
            dataset.indicators[canonical_name] = df
            continue

        logger.info(
            "Fetching indicator %s (%s) for %s [%d-%d] ...",
            canonical_name, indicator_code, country_iso3, from_year, to_year,
        )
        try:
            df = wb_client.get_indicator(
                country_iso3, indicator_code,
                from_year=from_year, to_year=to_year,
            )
            if df.empty:
                logger.warning(
                    "No data returned for %s (%s) -- marking as missing",
                    canonical_name, indicator_code,
                )
                dataset.missing.append(canonical_name)
                continue

            _write_cached_indicator(indicator_code, df)
            dataset.indicators[canonical_name] = df
            logger.info(
                "  -> %d data points for %s", len(df), canonical_name,
            )

        except WorldBankAPIError as exc:
            logger.warning(
                "Failed to fetch %s (%s): %s -- marking as missing",
                canonical_name, indicator_code, exc,
            )
            dataset.missing.append(canonical_name)

    logger.info(
        "Macro fetch complete: %d fetched, %d missing",
        len(dataset.indicators), len(dataset.missing),
    )

    # ------------------------------------------------------------------
    # 5. Optional: Gemini mapping suggestions (advisory only)
    # ------------------------------------------------------------------
    if gemini_client is not None:
        logger.info("Requesting Gemini mapping suggestions (advisory only) ...")
        suggestions = gemini_client.propose_world_bank_mappings(
            country=country_iso2, sector=sector,
        )
        if suggestions:
            dataset.gemini_suggestions = suggestions
            logger.info(
                "Gemini suggested %d mappings (logged, not applied at runtime)",
                len(suggestions),
            )

    return dataset


def save_macro_metadata(
    dataset: MacroDataset,
    output_path: str = "cache/macro_metadata.json",
) -> None:
    """Persist macro fetch summary to a JSON file for metadata.json."""
    meta = {
        "country_iso3": dataset.country_iso3,
        "fetched_indicators": list(dataset.indicators.keys()),
        "missing_indicators": dataset.missing,
        "gemini_suggestions": dataset.gemini_suggestions,
        "data_points_per_indicator": {
            name: len(df) for name, df in dataset.indicators.items()
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Macro metadata saved to %s", output_path)
