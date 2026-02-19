"""Peer Percentile Ranking -- rank target vs linked entity peers.

For each tier variable on each trading day, computes the target's
percentile rank within its peer group. A rank of 50 means the target
is at the median; 90 means it outperforms 90% of peers.

Columns are injected into the daily cache so temporal models learn
relative positioning, not just absolute levels.

Top-level entry point:
    ``compute_peer_ranking(cache, linked_caches)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from operator1.features.linked_aggregates import AGGREGATE_VARIABLES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Variables to rank (subset of AGGREGATE_VARIABLES that make sense for ranking)
RANK_VARIABLES: tuple[str, ...] = (
    "current_ratio",
    "debt_to_equity_abs",
    "free_cash_flow",
    "fcf_yield",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "pe_ratio_calc",
    "ev_to_ebitda",
    "return_1d",
    "volatility_21d",
    "drawdown_252d",
    "market_cap",
)

# Variables where lower is better (inverted ranking)
_LOWER_IS_BETTER: set[str] = {
    "debt_to_equity_abs",
    "volatility_21d",
    "drawdown_252d",
    "pe_ratio_calc",
    "ev_to_ebitda",
}

# Label thresholds for composite rank
_RANK_LABELS: list[tuple[float, str]] = [
    (20.0, "Laggard"),
    (40.0, "Below Average"),
    (60.0, "Average"),
    (80.0, "Above Average"),
    (100.1, "Leader"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PeerRankingResult:
    """Summary of peer percentile ranking."""

    n_peers: int = 0
    n_variables_ranked: int = 0
    latest_composite_rank: float = float("nan")
    latest_label: str = "Unknown"
    variable_ranks: dict[str, float] = field(default_factory=dict)
    columns_added: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_label(rank: float) -> str:
    """Map a percentile rank to a categorical label."""
    if np.isnan(rank):
        return "Unknown"
    for threshold, label in _RANK_LABELS:
        if rank < threshold:
            return label
    return "Leader"


def _compute_daily_percentile(
    target_series: pd.Series,
    peer_series_list: list[pd.Series],
    invert: bool = False,
) -> pd.Series:
    """Compute daily percentile rank of target within peer group.

    Parameters
    ----------
    target_series:
        Daily values for the target company.
    peer_series_list:
        List of daily value Series for each peer (aligned to same index).
    invert:
        If True, lower values get higher rank (e.g. debt ratios).

    Returns
    -------
    pd.Series of percentile ranks (0-100).
    """
    if not peer_series_list:
        return pd.Series(np.nan, index=target_series.index)

    # Build a matrix: target + all peers, columns are entities
    all_series = [target_series] + peer_series_list
    matrix = pd.concat(all_series, axis=1)

    # For each row, compute where target ranks
    result = pd.Series(np.nan, index=target_series.index)

    for idx in target_series.index:
        row = matrix.loc[idx].dropna()
        if len(row) < 2:  # Need at least target + 1 peer
            continue

        target_val = row.iloc[0]
        if np.isnan(target_val):
            continue

        all_vals = row.values
        if invert:
            # Lower is better: count how many are WORSE (higher)
            rank = float(np.sum(all_vals >= target_val)) / len(all_vals) * 100
        else:
            # Higher is better: count how many are WORSE (lower)
            rank = float(np.sum(all_vals <= target_val)) / len(all_vals) * 100

        result[idx] = rank

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_peer_ranking(
    cache: pd.DataFrame,
    linked_caches: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, PeerRankingResult]:
    """Compute peer percentile ranking and inject into cache.

    Parameters
    ----------
    cache:
        Target company daily cache (DatetimeIndex).
    linked_caches:
        Dict of ISIN -> daily cache DataFrame for linked entities.
        From CacheResult.linked_daily.

    Returns
    -------
    (cache, result)
        Cache with peer_pctile_* and peer_rank_* columns, and summary.
    """
    logger.info("Computing peer percentile ranking...")

    result = PeerRankingResult()

    if not linked_caches:
        logger.warning("No linked entity caches available for peer ranking")
        # Add empty columns for consistency
        cache["peer_rank_composite"] = np.nan
        cache["peer_rank_label"] = "Unknown"
        cache["is_missing_peer_rank"] = 1
        result.columns_added = ["peer_rank_composite", "peer_rank_label", "is_missing_peer_rank"]
        return cache, result

    result.n_peers = len(linked_caches)

    # Collect peer series for each variable
    ranked_count = 0
    pctile_series_list: list[pd.Series] = []

    for var in RANK_VARIABLES:
        if var not in cache.columns:
            continue

        # Collect peer values for this variable, aligned to target index
        peer_series: list[pd.Series] = []
        for isin, peer_cache in linked_caches.items():
            if var in peer_cache.columns:
                # Reindex peer to target's date index
                aligned = peer_cache[var].reindex(cache.index)
                peer_series.append(aligned)

        if not peer_series:
            continue

        invert = var in _LOWER_IS_BETTER
        pctile = _compute_daily_percentile(cache[var], peer_series, invert=invert)

        col_name = f"peer_pctile_{var}"
        cache[col_name] = pctile
        result.columns_added.append(col_name)

        # Track for composite
        pctile_series_list.append(pctile)
        ranked_count += 1

        # Latest value for summary
        valid_pctile = pctile.dropna()
        if len(valid_pctile) > 0:
            result.variable_ranks[var] = float(valid_pctile.iloc[-1])

    result.n_variables_ranked = ranked_count

    # Composite rank: average of all percentile ranks
    if pctile_series_list:
        composite = pd.concat(pctile_series_list, axis=1).mean(axis=1)
    else:
        composite = pd.Series(np.nan, index=cache.index)

    cache["peer_rank_composite"] = composite
    cache["peer_rank_label"] = composite.apply(_rank_label)
    cache["is_missing_peer_rank"] = composite.isna().astype(int)
    result.columns_added.extend(["peer_rank_composite", "peer_rank_label", "is_missing_peer_rank"])

    # Z-score vs peers (for each variable)
    for var in RANK_VARIABLES:
        if var not in cache.columns:
            continue
        peer_vals = []
        for isin, peer_cache in linked_caches.items():
            if var in peer_cache.columns:
                peer_vals.append(peer_cache[var].reindex(cache.index))
        if peer_vals:
            peer_matrix = pd.concat(peer_vals, axis=1)
            peer_mean = peer_matrix.mean(axis=1)
            peer_std = peer_matrix.std(axis=1).replace(0, np.nan)
            zscore = (cache[var] - peer_mean) / peer_std
            col_name = f"peer_zscore_{var}"
            cache[col_name] = zscore
            result.columns_added.append(col_name)

    # Summary
    valid_composite = composite.dropna()
    if len(valid_composite) > 0:
        result.latest_composite_rank = float(valid_composite.iloc[-1])
        result.latest_label = _rank_label(result.latest_composite_rank)

    logger.info(
        "Peer ranking: %d peers, %d variables ranked, composite=%.1f (%s)",
        result.n_peers,
        result.n_variables_ranked,
        result.latest_composite_rank,
        result.latest_label,
    )

    return cache, result
