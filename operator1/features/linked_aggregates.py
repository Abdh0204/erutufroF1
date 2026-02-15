"""T3.4 -- Linked entity aggregates (daily per relationship group).

For each day and each relationship group (competitors, suppliers,
customers, financial_institutions, sector_peers, industry_peers),
compute mean and median of key variables across group members.
Also compute relative strength and valuation premium measures
comparing the target to its peers.

Groups with zero members produce null columns with ``is_missing_*``
flags.  Output is persisted to ``cache/linked_aggregates_daily.parquet``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# Key variables to aggregate across linked entities.
# These should exist in the entity daily caches after derived-variable
# computation.
AGGREGATE_VARIABLES: tuple[str, ...] = (
    "return_1d",
    "volatility_21d",
    "drawdown_252d",
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
    "enterprise_value",
    "market_cap",
)

# Relationship groups to process
RELATIONSHIP_GROUPS: tuple[str, ...] = (
    "competitors",
    "suppliers",
    "customers",
    "financial_institutions",
    "sector_peers",
    "industry_peers",
)


# ---------------------------------------------------------------------------
# Per-group aggregation
# ---------------------------------------------------------------------------


def _aggregate_group(
    group_name: str,
    entity_frames: list[pd.DataFrame],
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute mean and median aggregates for a single relationship group.

    Parameters
    ----------
    group_name:
        E.g. ``"competitors"``.
    entity_frames:
        List of daily DataFrames for entities in this group.
    daily_index:
        The business-day index to align to.

    Returns
    -------
    pd.DataFrame
        Indexed by ``daily_index`` with columns like
        ``competitors_avg_return_1d``, ``competitors_median_volatility_21d``.
    """
    result = pd.DataFrame(index=daily_index)

    if not entity_frames:
        # No members -- produce null columns with missing flags
        for var in AGGREGATE_VARIABLES:
            avg_col = f"{group_name}_avg_{var}"
            med_col = f"{group_name}_median_{var}"
            result[avg_col] = np.nan
            result[med_col] = np.nan
            result[f"is_missing_{avg_col}"] = 1
            result[f"is_missing_{med_col}"] = 1
        logger.debug("Group '%s' has 0 members -- null columns", group_name)
        return result

    # Stack all entity frames for this group
    aligned: list[pd.DataFrame] = []
    for df in entity_frames:
        # Reindex to the common daily index
        reindexed = df.reindex(daily_index)
        aligned.append(reindexed)

    for var in AGGREGATE_VARIABLES:
        # Collect the variable from each entity
        series_list: list[pd.Series] = []
        for df in aligned:
            if var in df.columns:
                series_list.append(df[var])

        if not series_list:
            avg_col = f"{group_name}_avg_{var}"
            med_col = f"{group_name}_median_{var}"
            result[avg_col] = np.nan
            result[med_col] = np.nan
            result[f"is_missing_{avg_col}"] = 1
            result[f"is_missing_{med_col}"] = 1
            continue

        combined = pd.concat(series_list, axis=1)

        avg_col = f"{group_name}_avg_{var}"
        med_col = f"{group_name}_median_{var}"

        result[avg_col] = combined.mean(axis=1)
        result[med_col] = combined.median(axis=1)
        result[f"is_missing_{avg_col}"] = result[avg_col].isna().astype(int)
        result[f"is_missing_{med_col}"] = result[med_col].isna().astype(int)

    logger.info(
        "Group '%s': aggregated %d variables across %d entities",
        group_name, len(AGGREGATE_VARIABLES), len(entity_frames),
    )

    return result


# ---------------------------------------------------------------------------
# Relative measures
# ---------------------------------------------------------------------------


def _compute_relative_measures(
    target_daily: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute relative strength and valuation premium vs peer groups.

    Variables:
    - ``rel_strength_vs_sector``: target return_1d / sector_peers_avg_return_1d
    - ``valuation_premium_vs_industry``: target pe_ratio_calc / industry_peers_avg_pe_ratio_calc

    These use safe division (null when denominator is zero/null).
    """
    result = pd.DataFrame(index=target_daily.index)

    # Relative strength vs sector peers
    sector_avg_ret = aggregates.get(
        "sector_peers_avg_return_1d",
        pd.Series(np.nan, index=target_daily.index),
    )
    target_ret = target_daily.get(
        "return_1d",
        pd.Series(np.nan, index=target_daily.index),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = target_ret / sector_avg_ret
        rel = rel.replace([np.inf, -np.inf], np.nan)
    result["rel_strength_vs_sector"] = rel
    result["is_missing_rel_strength_vs_sector"] = rel.isna().astype(int)

    # Valuation premium vs industry peers
    industry_avg_pe = aggregates.get(
        "industry_peers_avg_pe_ratio_calc",
        pd.Series(np.nan, index=target_daily.index),
    )
    target_pe = target_daily.get(
        "pe_ratio_calc",
        pd.Series(np.nan, index=target_daily.index),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        premium = target_pe / industry_avg_pe
        premium = premium.replace([np.inf, -np.inf], np.nan)
    result["valuation_premium_vs_industry"] = premium
    result["is_missing_valuation_premium_vs_industry"] = premium.isna().astype(int)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_linked_aggregates(
    target_daily: pd.DataFrame,
    linked_daily: dict[str, pd.DataFrame],
    entity_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute all linked entity aggregates and relative measures.

    Parameters
    ----------
    target_daily:
        Target company daily cache (with derived variables).
    linked_daily:
        Dict of ``{isin: daily_df}`` for each linked entity.
    entity_groups:
        Dict of ``{relationship_group: [isin, ...]}`` mapping each
        group to its member ISINs.  This comes from
        ``DiscoveryResult.linked`` in entity_discovery.

    Returns
    -------
    pd.DataFrame
        Daily-indexed DataFrame with all aggregate columns and
        relative measures.
    """
    daily_index = target_daily.index

    # Build per-group frames
    all_groups: list[pd.DataFrame] = []

    for group_name in RELATIONSHIP_GROUPS:
        member_isins = entity_groups.get(group_name, [])
        member_frames = [
            linked_daily[isin]
            for isin in member_isins
            if isin in linked_daily
        ]
        group_agg = _aggregate_group(group_name, member_frames, daily_index)
        all_groups.append(group_agg)

    # Combine all group aggregates
    aggregates = pd.concat(all_groups, axis=1)

    # Compute relative measures
    rel_measures = _compute_relative_measures(target_daily, aggregates)
    aggregates = pd.concat([aggregates, rel_measures], axis=1)

    logger.info(
        "Linked aggregates computed: %d columns across %d groups",
        len(aggregates.columns), len(RELATIONSHIP_GROUPS),
    )

    return aggregates


def persist_linked_aggregates(
    aggregates: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """Write linked aggregates to disk.

    Parameters
    ----------
    aggregates:
        Output of ``compute_linked_aggregates()``.
    output_path:
        Override output path.

    Returns
    -------
    str
        Path to the written Parquet file.
    """
    if output_path is None:
        output_path = str(Path(CACHE_DIR) / "linked_aggregates_daily.parquet")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aggregates.to_parquet(output_path)
    logger.info(
        "Linked aggregates saved: %s (%d rows, %d cols)",
        output_path, len(aggregates), len(aggregates.columns),
    )
    return output_path
