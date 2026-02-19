"""Step 0 / 0.1 -- Verify target identifiers and extract metadata.

Validates the user-provided ISIN (Eulerpool) and FMP symbol, then
extracts the target company's country and profile fields needed by
every downstream module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from operator1.clients.eulerpool import EulerportAPIError
from operator1.clients.eod import EODAPIError
from operator1.clients.equity_provider import EquityProvider
from operator1.clients.fmp import FMPClient, FMPAPIError

logger = logging.getLogger(__name__)


@dataclass
class VerifiedTarget:
    """Container for a successfully verified target company.

    All fields are populated during the verification step and
    remain immutable for the rest of the pipeline.
    """

    isin: str
    ticker: str
    name: str
    country: str          # ISO-2 from Eulerpool profile
    sector: str
    industry: str
    sub_industry: str | None
    fmp_symbol: str
    currency: str
    exchange: str
    raw_profile: dict[str, Any] = field(default_factory=dict, repr=False)


class VerificationError(Exception):
    """Raised when identifier verification fails.

    The pipeline should halt immediately when this is raised.
    """

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"[{source}] Verification failed: {detail}")


def verify_identifiers(
    target_isin: str,
    fmp_symbol: str,
    eulerpool_client: EquityProvider,
    fmp_client: FMPClient,
) -> VerifiedTarget:
    """Verify both identifiers and return a ``VerifiedTarget``.

    Parameters
    ----------
    target_isin:
        Eulerpool ISIN for the target company.
    fmp_symbol:
        FMP ticker symbol (authoritative OHLCV source).
    eulerpool_client:
        Initialised Eulerpool API client.
    fmp_client:
        Initialised FMP API client.

    Returns
    -------
    VerifiedTarget
        Populated dataclass with all verified metadata.

    Raises
    ------
    VerificationError
        If the ISIN is invalid, the FMP symbol is invalid, or either
        API key is rejected.
    """
    # ------------------------------------------------------------------
    # 1. Verify ISIN via Eulerpool profile
    # ------------------------------------------------------------------
    logger.info("Verifying ISIN %s via Eulerpool ...", target_isin)
    try:
        profile = eulerpool_client.get_profile(target_isin)
    except (EulerportAPIError, EODAPIError) as exc:
        raise VerificationError(
            "Eulerpool",
            f"Invalid ISIN '{target_isin}' or bad API key: {exc}",
        ) from exc

    country = profile.get("country")
    if not country:
        raise VerificationError(
            "Eulerpool",
            f"Profile for ISIN '{target_isin}' has no country field. "
            "Cannot proceed without country information.",
        )

    logger.info(
        "Eulerpool OK -- %s (%s), country=%s, sector=%s",
        profile.get("name"), profile.get("ticker"), country, profile.get("sector"),
    )

    # ------------------------------------------------------------------
    # 2. Verify FMP symbol
    # ------------------------------------------------------------------
    logger.info("Verifying FMP symbol %s ...", fmp_symbol)
    try:
        fmp_quote = fmp_client.get_quote(fmp_symbol)
    except FMPAPIError as exc:
        raise VerificationError(
            "FMP",
            f"Invalid symbol '{fmp_symbol}' or bad API key: {exc}",
        ) from exc

    logger.info(
        "FMP OK -- %s, price=%.2f",
        fmp_quote.get("name", fmp_symbol),
        fmp_quote.get("price", 0),
    )

    # ------------------------------------------------------------------
    # 3. Build VerifiedTarget
    # ------------------------------------------------------------------
    return VerifiedTarget(
        isin=profile.get("isin", target_isin),
        ticker=profile.get("ticker", ""),
        name=profile.get("name", ""),
        country=country,
        sector=profile.get("sector", ""),
        industry=profile.get("industry", ""),
        sub_industry=profile.get("sub_industry"),
        fmp_symbol=fmp_symbol,
        currency=profile.get("currency", ""),
        exchange=profile.get("exchange", ""),
        raw_profile=profile,
    )
