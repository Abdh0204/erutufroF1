"""Step C / C.1 -- Bulk data extraction for target and linked entities.

Fetches profiles, quotes, financial statements, and OHLCV data for all
entities.  Raw data is persisted to ``cache/raw/{isin}/`` as Parquet files.
Every API call is logged via the http_utils request log.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.clients.eulerpool import EulerportAPIError
from operator1.clients.eod import EODAPIError
from operator1.clients.equity_provider import EquityProvider
from operator1.clients.fmp import FMPClient, FMPAPIError
from operator1.config_loader import get_global_config
from operator1.constants import CACHE_DIR, RAW_CACHE_DIR, DATE_START, DATE_END
from operator1.steps.verify_identifiers import VerifiedTarget

logger = logging.getLogger(__name__)


@dataclass
class EntityData:
    """Raw data container for a single entity."""

    isin: str
    profile: dict[str, Any] = field(default_factory=dict)
    quotes: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow_statement: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Target-only extras
    peers: list[str] = field(default_factory=list)
    supply_chain: list[dict[str, Any]] = field(default_factory=list)
    executives: list[dict[str, Any]] = field(default_factory=list)

    # FMP OHLCV (target only)
    fmp_ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ExtractionResult:
    """Container for all extracted data."""

    target: EntityData = field(default_factory=lambda: EntityData(isin=""))
    linked: dict[str, EntityData] = field(default_factory=dict)  # isin -> EntityData
    errors: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _entity_cache_dir(isin: str) -> Path:
    """Return the raw cache directory for an entity."""
    return Path(RAW_CACHE_DIR) / isin.replace("/", "_")


def _cache_file(isin: str, name: str) -> Path:
    return _entity_cache_dir(isin) / f"{name}.parquet"


def _json_cache_file(isin: str, name: str) -> Path:
    return _entity_cache_dir(isin) / f"{name}.json"


def _is_cached(isin: str, name: str, is_json: bool = False) -> bool:
    path = _json_cache_file(isin, name) if is_json else _cache_file(isin, name)
    return path.exists()


def _write_df(isin: str, name: str, df: pd.DataFrame) -> None:
    path = _cache_file(isin, name)
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path, index=False)


def _read_df(isin: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_file(isin, name))


def _write_json(isin: str, name: str, data: Any) -> None:
    path = _json_cache_file(isin, name)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _read_json(isin: str, name: str) -> Any:
    with open(_json_cache_file(isin, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-entity extraction
# ---------------------------------------------------------------------------

def _fetch_entity_data(
    isin: str,
    eulerpool_client: EquityProvider,
    is_target: bool = False,
    force_rebuild: bool = False,
) -> EntityData:
    """Fetch all Eulerpool data for a single entity.

    Parameters
    ----------
    isin:
        Entity ISIN.
    eulerpool_client:
        Initialised Eulerpool client.
    is_target:
        If True, also fetch peers, supply chain, and executives.
    force_rebuild:
        If True, bypass disk caches.
    """
    entity = EntityData(isin=isin)

    # Profile
    if not force_rebuild and _is_cached(isin, "profile", is_json=True):
        entity.profile = _read_json(isin, "profile")
    else:
        try:
            entity.profile = eulerpool_client.get_profile(isin)
            _write_json(isin, "profile", entity.profile)
        except (EulerportAPIError, EODAPIError) as exc:
            logger.warning("Profile fetch failed for %s: %s", isin, exc)

    # Quotes
    if not force_rebuild and _is_cached(isin, "quotes"):
        entity.quotes = _read_df(isin, "quotes")
    else:
        try:
            entity.quotes = eulerpool_client.get_quotes(isin)
            if not entity.quotes.empty:
                _write_df(isin, "quotes", entity.quotes)
        except (EulerportAPIError, EODAPIError) as exc:
            logger.warning("Quotes fetch failed for %s: %s", isin, exc)

    # Income statement
    if not force_rebuild and _is_cached(isin, "income_statement"):
        entity.income_statement = _read_df(isin, "income_statement")
    else:
        try:
            entity.income_statement = eulerpool_client.get_income_statement(isin)
            if not entity.income_statement.empty:
                _write_df(isin, "income_statement", entity.income_statement)
        except (EulerportAPIError, EODAPIError) as exc:
            logger.warning("Income statement fetch failed for %s: %s", isin, exc)

    # Balance sheet
    if not force_rebuild and _is_cached(isin, "balance_sheet"):
        entity.balance_sheet = _read_df(isin, "balance_sheet")
    else:
        try:
            entity.balance_sheet = eulerpool_client.get_balance_sheet(isin)
            if not entity.balance_sheet.empty:
                _write_df(isin, "balance_sheet", entity.balance_sheet)
        except (EulerportAPIError, EODAPIError) as exc:
            logger.warning("Balance sheet fetch failed for %s: %s", isin, exc)

    # Cash-flow statement
    if not force_rebuild and _is_cached(isin, "cashflow_statement"):
        entity.cashflow_statement = _read_df(isin, "cashflow_statement")
    else:
        try:
            entity.cashflow_statement = eulerpool_client.get_cashflow_statement(isin)
            if not entity.cashflow_statement.empty:
                _write_df(isin, "cashflow_statement", entity.cashflow_statement)
        except (EulerportAPIError, EODAPIError) as exc:
            logger.warning("Cash-flow statement fetch failed for %s: %s", isin, exc)

    # Target-only extras
    if is_target:
        # Peers
        if not force_rebuild and _is_cached(isin, "peers", is_json=True):
            entity.peers = _read_json(isin, "peers")
        else:
            try:
                entity.peers = eulerpool_client.get_peers(isin)
                _write_json(isin, "peers", entity.peers)
            except (EulerportAPIError, EODAPIError) as exc:
                logger.warning("Peers fetch failed for %s: %s", isin, exc)

        # Supply chain
        if not force_rebuild and _is_cached(isin, "supply_chain", is_json=True):
            entity.supply_chain = _read_json(isin, "supply_chain")
        else:
            try:
                entity.supply_chain = eulerpool_client.get_supply_chain(isin)
                _write_json(isin, "supply_chain", entity.supply_chain)
            except (EulerportAPIError, EODAPIError) as exc:
                logger.warning("Supply chain fetch failed for %s: %s", isin, exc)

        # Executives
        if not force_rebuild and _is_cached(isin, "executives", is_json=True):
            entity.executives = _read_json(isin, "executives")
        else:
            try:
                entity.executives = eulerpool_client.get_executives(isin)
                _write_json(isin, "executives", entity.executives)
            except (EulerportAPIError, EODAPIError) as exc:
                logger.warning("Executives fetch failed for %s: %s", isin, exc)

    return entity


def _fetch_fmp_ohlcv(
    fmp_symbol: str,
    fmp_client: FMPClient,
    force_rebuild: bool = False,
    cache_isin: str = "",
) -> pd.DataFrame:
    """Fetch FMP daily OHLCV for the target (authoritative price source).

    Parameters
    ----------
    fmp_symbol:
        FMP ticker symbol.
    fmp_client:
        Initialised FMP client.
    force_rebuild:
        Bypass disk cache.
    cache_isin:
        ISIN to use for the cache directory.
    """
    cache_name = "fmp_ohlcv"
    if cache_isin and not force_rebuild and _is_cached(cache_isin, cache_name):
        logger.debug("Cache HIT for FMP OHLCV (%s)", fmp_symbol)
        return _read_df(cache_isin, cache_name)

    logger.info(
        "Fetching FMP daily OHLCV for %s [%s - %s] ...",
        fmp_symbol, DATE_START, DATE_END,
    )
    try:
        df = fmp_client.get_daily_ohlcv(
            fmp_symbol, from_date=DATE_START, to_date=DATE_END,
        )
        if cache_isin and not df.empty:
            _write_df(cache_isin, cache_name, df)
        logger.info("FMP OHLCV: %d rows for %s", len(df), fmp_symbol)
        return df
    except FMPAPIError as exc:
        logger.error("FMP OHLCV fetch failed for %s: %s", fmp_symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main extraction orchestrator
# ---------------------------------------------------------------------------

def extract_all_data(
    target: VerifiedTarget,
    linked_isins: list[str],
    eulerpool_client: EquityProvider,
    fmp_client: FMPClient,
    force_rebuild: bool | None = None,
) -> ExtractionResult:
    """Fetch all raw data for the target and linked entities.

    Parameters
    ----------
    target:
        Verified target from Step 0.1.
    linked_isins:
        List of ISINs for linked entities (from discovery).
    eulerpool_client:
        Initialised Eulerpool client.
    fmp_client:
        Initialised FMP client.
    force_rebuild:
        Override FORCE_REBUILD config.

    Returns
    -------
    ExtractionResult
        Contains raw data for target and all linked entities.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    result = ExtractionResult()

    # ------------------------------------------------------------------
    # 1. Target entity -- Eulerpool data
    # ------------------------------------------------------------------
    logger.info("Extracting target data for %s (%s) ...", target.name, target.isin)
    result.target = _fetch_entity_data(
        target.isin, eulerpool_client, is_target=True, force_rebuild=force_rebuild,
    )

    # ------------------------------------------------------------------
    # 2. Target entity -- FMP OHLCV (authoritative)
    # ------------------------------------------------------------------
    result.target.fmp_ohlcv = _fetch_fmp_ohlcv(
        target.fmp_symbol, fmp_client,
        force_rebuild=force_rebuild, cache_isin=target.isin,
    )
    if result.target.fmp_ohlcv.empty:
        result.errors.append({
            "entity": target.isin,
            "module": "fmp_ohlcv",
            "error": "No OHLCV data -- downstream price-dependent modules will halt",
        })

    # ------------------------------------------------------------------
    # 3. Linked entities -- Eulerpool data (batched)
    # ------------------------------------------------------------------
    total = len(linked_isins)
    for idx, isin in enumerate(linked_isins, 1):
        logger.info(
            "Extracting linked entity %d/%d: %s ...", idx, total, isin,
        )
        try:
            entity_data = _fetch_entity_data(
                isin, eulerpool_client, is_target=False, force_rebuild=force_rebuild,
            )
            result.linked[isin] = entity_data
        except Exception as exc:
            logger.error("Extraction failed for linked entity %s: %s", isin, exc)
            result.errors.append({
                "entity": isin,
                "module": "eulerpool_linked",
                "error": str(exc),
            })

    # Summary
    logger.info(
        "Extraction complete: target=%s, linked=%d, errors=%d",
        target.isin, len(result.linked), len(result.errors),
    )

    return result


def save_extraction_metadata(
    result: ExtractionResult,
    target: VerifiedTarget,
    output_path: str = "cache/extraction_metadata.json",
) -> None:
    """Persist extraction summary metadata to disk."""
    meta = {
        "target_isin": target.isin,
        "target_fmp_symbol": target.fmp_symbol,
        "target_data": {
            "profile": bool(result.target.profile),
            "quotes_rows": len(result.target.quotes),
            "income_statement_rows": len(result.target.income_statement),
            "balance_sheet_rows": len(result.target.balance_sheet),
            "cashflow_statement_rows": len(result.target.cashflow_statement),
            "fmp_ohlcv_rows": len(result.target.fmp_ohlcv),
            "peers_count": len(result.target.peers),
            "supply_chain_count": len(result.target.supply_chain),
            "executives_count": len(result.target.executives),
        },
        "linked_entities": {
            isin: {
                "profile": bool(e.profile),
                "quotes_rows": len(e.quotes),
                "income_statement_rows": len(e.income_statement),
                "balance_sheet_rows": len(e.balance_sheet),
                "cashflow_statement_rows": len(e.cashflow_statement),
            }
            for isin, e in result.linked.items()
        },
        "errors": result.errors,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Extraction metadata saved to %s", output_path)
