"""T3.5 -- Macro data alignment (World Bank yearly -> daily).

Takes the yearly World Bank indicator data fetched by
``macro_mapping.py`` and aligns each canonical variable to daily
frequency using as-of logic (latest year <= day's year).

Also computes:
- ``inflation_rate_daily_equivalent = inflation_rate_yoy / 365``
- ``real_return_1d = return_1d - inflation_rate_daily_equivalent``

All variables get companion ``is_missing_<var>`` flags.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.steps.macro_mapping import MacroDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# As-of yearly -> daily alignment
# ---------------------------------------------------------------------------


def _align_yearly_to_daily(
    yearly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    variable_name: str,
) -> pd.Series:
    """Align a yearly indicator to a daily date index via as-of logic.

    For each day ``t``, the value from the latest year where
    ``year <= t.year`` is used.

    Parameters
    ----------
    yearly_df:
        DataFrame with columns ``year`` and ``value``.
    daily_index:
        Business-day DatetimeIndex.
    variable_name:
        Name of the macro variable (for logging).

    Returns
    -------
    pd.Series
        Daily-aligned series indexed by ``daily_index``.
    """
    if yearly_df.empty:
        logger.debug("No data for %s -- returning NaN series", variable_name)
        return pd.Series(np.nan, index=daily_index, name=variable_name)

    df = yearly_df.copy()

    # Ensure year column is integer
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    elif "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    else:
        logger.warning(
            "No year/date column in %s data -- returning NaN", variable_name,
        )
        return pd.Series(np.nan, index=daily_index, name=variable_name)

    # Ensure value column exists
    if "value" not in df.columns:
        # Try to find a numeric column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "year"]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "value"})
        else:
            logger.warning(
                "No value column in %s data -- returning NaN", variable_name,
            )
            return pd.Series(np.nan, index=daily_index, name=variable_name)

    df = df.dropna(subset=["year", "value"]).sort_values("year")

    # Build a year -> value lookup
    year_values = dict(zip(df["year"].astype(int), df["value"]))

    if not year_values:
        return pd.Series(np.nan, index=daily_index, name=variable_name)

    # For each day, find the latest year <= day.year
    years_available = sorted(year_values.keys())
    daily_years = daily_index.year

    result = pd.Series(np.nan, index=daily_index, name=variable_name, dtype=float)

    for yr in years_available:
        mask = daily_years >= yr
        result = result.where(~mask, other=year_values[yr])

    # Actually we need as-of: for each day, use the latest available year <= day.year
    # The loop above overwrites with later years, so the final result is correct:
    # the last year that satisfies year <= day.year wins.

    return result


# ---------------------------------------------------------------------------
# Derived macro variables
# ---------------------------------------------------------------------------


def _compute_inflation_daily(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Compute inflation_rate_daily_equivalent and real_return_1d.

    ``inflation_rate_daily_equivalent = inflation_rate_yoy / 365``
    ``real_return_1d = return_1d - inflation_rate_daily_equivalent``
    """
    if "inflation_rate_yoy" in aligned.columns:
        aligned["inflation_rate_daily_equivalent"] = (
            aligned["inflation_rate_yoy"] / 365.0
        )
        aligned["is_missing_inflation_rate_daily_equivalent"] = (
            aligned["inflation_rate_daily_equivalent"].isna().astype(int)
        )
    else:
        aligned["inflation_rate_daily_equivalent"] = np.nan
        aligned["is_missing_inflation_rate_daily_equivalent"] = 1

    return aligned


def _compute_real_return(
    macro_aligned: pd.DataFrame,
    target_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute real_return_1d = return_1d - inflation_rate_daily_equivalent.

    This requires both the macro-aligned frame and the target daily cache
    (which has return_1d from derived variables).
    """
    return_1d = target_daily.get(
        "return_1d",
        pd.Series(np.nan, index=macro_aligned.index),
    )
    inflation_daily = macro_aligned.get(
        "inflation_rate_daily_equivalent",
        pd.Series(np.nan, index=macro_aligned.index),
    )

    macro_aligned["real_return_1d"] = return_1d - inflation_daily
    macro_aligned["is_missing_real_return_1d"] = (
        macro_aligned["real_return_1d"].isna().astype(int)
    )

    return macro_aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_macro_to_daily(
    macro_dataset: MacroDataset,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align all World Bank macro indicators to daily frequency.

    Parameters
    ----------
    macro_dataset:
        Output of ``fetch_macro_data()`` containing yearly indicator
        DataFrames.
    daily_index:
        Business-day DatetimeIndex (from the target daily cache).

    Returns
    -------
    pd.DataFrame
        Daily-indexed DataFrame with all macro variables and their
        ``is_missing_*`` companion flags.
    """
    result = pd.DataFrame(index=daily_index)

    for canonical_name, yearly_df in macro_dataset.indicators.items():
        logger.info("Aligning macro variable: %s ...", canonical_name)
        aligned_series = _align_yearly_to_daily(
            yearly_df, daily_index, canonical_name,
        )
        result[canonical_name] = aligned_series
        result[f"is_missing_{canonical_name}"] = aligned_series.isna().astype(int)

    # Mark explicitly missing indicators
    for missing_name in macro_dataset.missing:
        if missing_name not in result.columns:
            result[missing_name] = np.nan
            result[f"is_missing_{missing_name}"] = 1

    # Compute derived macro variables
    result = _compute_inflation_daily(result)

    logger.info(
        "Macro alignment complete: %d variables aligned to %d days, %d missing",
        len(macro_dataset.indicators),
        len(daily_index),
        len(macro_dataset.missing),
    )

    return result


def merge_macro_onto_target(
    target_daily: pd.DataFrame,
    macro_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Merge macro-aligned data onto the target daily cache.

    Also computes ``real_return_1d``.

    Parameters
    ----------
    target_daily:
        Target daily cache (with derived variables computed).
    macro_aligned:
        Output of ``align_macro_to_daily()``.

    Returns
    -------
    pd.DataFrame
        Target daily cache augmented with macro columns and
        ``real_return_1d``.
    """
    # Compute real return before merging
    macro_aligned = _compute_real_return(macro_aligned, target_daily)

    # Merge -- avoid duplicate columns
    new_cols = [c for c in macro_aligned.columns if c not in target_daily.columns]
    if new_cols:
        merged = target_daily.join(macro_aligned[new_cols])
    else:
        merged = target_daily.copy()

    logger.info(
        "Macro data merged onto target: %d new columns",
        len(new_cols),
    )

    return merged
