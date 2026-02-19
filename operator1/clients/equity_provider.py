"""Equity data provider abstraction.

Defines a ``Protocol`` that both ``EulerportClient`` and ``EODClient``
satisfy, plus a factory function that picks the right backend based on
available API keys.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class EquityProvider(Protocol):
    """Structural interface that any equity-data backend must satisfy.

    Both ``EulerportClient`` and ``EODClient`` implement these methods
    so the rest of the pipeline can treat them interchangeably.
    """

    def get_profile(self, isin: str) -> dict[str, Any]: ...
    def get_quotes(self, identifier: str) -> pd.DataFrame: ...
    def get_income_statement(self, isin: str) -> pd.DataFrame: ...
    def get_balance_sheet(self, isin: str) -> pd.DataFrame: ...
    def get_cashflow_statement(self, isin: str) -> pd.DataFrame: ...
    def get_peers(self, isin: str) -> list[str]: ...
    def get_supply_chain(self, isin: str) -> list[dict[str, Any]]: ...
    def get_executives(self, isin: str) -> list[dict[str, Any]]: ...
    def search(self, query: str) -> list[dict[str, Any]]: ...


def create_equity_provider(secrets: dict[str, str]) -> EquityProvider:
    """Instantiate the appropriate equity data client.

    Selection priority:

    1. If ``EULERPOOL_API_KEY`` is present, use :class:`EulerportClient`.
    2. If ``EOD_API_KEY`` is present, use :class:`EODClient`.
    3. Raise ``SystemExit`` if neither is available.

    Parameters
    ----------
    secrets:
        Dictionary of loaded API secrets (from ``load_secrets``).

    Returns
    -------
    EquityProvider
        An initialised client satisfying the ``EquityProvider`` protocol.
    """
    eulerpool_key = secrets.get("EULERPOOL_API_KEY")
    eod_key = secrets.get("EOD_API_KEY")

    if eulerpool_key:
        from operator1.clients.eulerpool import EulerportClient

        logger.info("Using Eulerpool as equity data provider")
        return EulerportClient(api_key=eulerpool_key)

    if eod_key:
        from operator1.clients.eod import EODClient

        logger.info("Using EOD Historical Data as equity data provider")
        return EODClient(api_key=eod_key)

    raise SystemExit(
        "No equity data provider configured. "
        "Set either EULERPOOL_API_KEY or EOD_API_KEY in your environment / .env file."
    )
