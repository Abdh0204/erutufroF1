"""Macro Quadrant Mapping -- classify macro environment into 4 quadrants.

Uses GDP growth and inflation from World Bank data to classify each
trading day into one of four macro quadrants:

- **Goldilocks**: growth above trend, inflation below target
- **Reflation**: growth above trend, inflation above target
- **Stagflation**: growth below trend, inflation above target
- **Deflation**: growth below trend, inflation below target

Columns are injected into the daily cache so temporal models
(forecasting, forward pass, burn-out) automatically learn macro context.

Top-level entry point:
    ``compute_macro_quadrant(cache, macro_data)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default inflation target (central bank consensus)
_DEFAULT_INFLATION_TARGET: float = 2.0

# Stability window for measuring quadrant persistence
_STABILITY_WINDOW: int = 21

# Quadrant labels
QUADRANT_GOLDILOCKS = "goldilocks"
QUADRANT_REFLATION = "reflation"
QUADRANT_STAGFLATION = "stagflation"
QUADRANT_DEFLATION = "deflation"

# Numeric encoding
QUADRANT_NUMERIC = {
    QUADRANT_GOLDILOCKS: 0,
    QUADRANT_REFLATION: 1,
    QUADRANT_STAGFLATION: 2,
    QUADRANT_DEFLATION: 3,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MacroQuadrantResult:
    """Summary of macro quadrant classification."""

    current_quadrant: str = "unknown"
    quadrant_distribution: dict[str, float] = field(default_factory=dict)
    n_transitions: int = 0
    growth_trend: float = float("nan")
    inflation_target: float = _DEFAULT_INFLATION_TARGET
    columns_added: list[str] = field(default_factory=list)
    n_days_classified: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_yearly_value(
    macro_data: Any,
    indicator_name: str,
) -> pd.Series | None:
    """Extract a yearly indicator as a Series indexed by year.

    Parameters
    ----------
    macro_data:
        MacroDataset object with .indicators dict.
    indicator_name:
        Canonical name (e.g. 'gdp_growth', 'inflation_rate_yoy').

    Returns
    -------
    pd.Series indexed by year, or None if unavailable.
    """
    if macro_data is None:
        return None

    indicators = getattr(macro_data, "indicators", None)
    if indicators is None or not isinstance(indicators, dict):
        return None

    df = indicators.get(indicator_name)
    if df is None or df.empty:
        return None

    # DataFrame should have 'year' and 'value' columns
    if "year" not in df.columns or "value" not in df.columns:
        return None

    result = df.set_index("year")["value"].dropna().sort_index()
    return result if not result.empty else None


def _align_yearly_to_daily(
    yearly: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align yearly data to daily index via as-of logic.

    For each day, use the value from the latest year <= day's year.
    """
    if yearly is None or yearly.empty:
        return pd.Series(np.nan, index=daily_index)

    result = pd.Series(np.nan, index=daily_index)
    years_available = sorted(yearly.index)

    for day in daily_index:
        day_year = day.year
        # Find latest year <= day_year
        candidates = [y for y in years_available if y <= day_year]
        if candidates:
            result[day] = yearly[candidates[-1]]

    return result


def _classify_quadrant(
    growth_vs_trend: float,
    inflation_vs_target: float,
) -> str:
    """Classify a single observation into a macro quadrant."""
    if np.isnan(growth_vs_trend) or np.isnan(inflation_vs_target):
        return "unknown"

    high_growth = growth_vs_trend >= 0
    high_inflation = inflation_vs_target >= 0

    if high_growth and not high_inflation:
        return QUADRANT_GOLDILOCKS
    elif high_growth and high_inflation:
        return QUADRANT_REFLATION
    elif not high_growth and high_inflation:
        return QUADRANT_STAGFLATION
    else:
        return QUADRANT_DEFLATION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_macro_quadrant(
    cache: pd.DataFrame,
    macro_data: Any = None,
    *,
    inflation_target: float = _DEFAULT_INFLATION_TARGET,
) -> tuple[pd.DataFrame, MacroQuadrantResult]:
    """Classify each trading day into a macro quadrant and inject into cache.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex).
    macro_data:
        MacroDataset from Step 2 (World Bank data). Contains .indicators
        dict with 'gdp_growth' and 'inflation_rate_yoy'.
    inflation_target:
        Central bank inflation target (default 2.0%).

    Returns
    -------
    (cache, result)
        Cache with macro_* columns, and MacroQuadrantResult summary.
    """
    logger.info("Computing macro quadrant classification...")

    result = MacroQuadrantResult(inflation_target=inflation_target)

    # Extract GDP growth and inflation
    gdp_growth = _extract_yearly_value(macro_data, "gdp_growth")
    inflation = _extract_yearly_value(macro_data, "inflation_rate_yoy")

    # Compute GDP trend (median of available years)
    growth_trend = float("nan")
    if gdp_growth is not None and len(gdp_growth) >= 2:
        growth_trend = float(gdp_growth.median())
        result.growth_trend = growth_trend

    # Align to daily
    daily_gdp = _align_yearly_to_daily(gdp_growth, cache.index)
    daily_inflation = _align_yearly_to_daily(inflation, cache.index)

    # Compute deviations
    growth_vs_trend = daily_gdp - growth_trend if not np.isnan(growth_trend) else pd.Series(np.nan, index=cache.index)
    inflation_vs_target = daily_inflation - inflation_target

    cache["macro_growth_vs_trend"] = growth_vs_trend
    cache["macro_inflation_vs_target"] = inflation_vs_target
    result.columns_added.extend(["macro_growth_vs_trend", "macro_inflation_vs_target"])

    # Classify each day
    quadrants = pd.Series(
        [_classify_quadrant(g, i) for g, i in zip(growth_vs_trend, inflation_vs_target)],
        index=cache.index,
    )

    cache["macro_quadrant"] = quadrants
    cache["macro_quadrant_numeric"] = quadrants.map(
        lambda q: QUADRANT_NUMERIC.get(q, -1)
    ).astype(float)
    cache["macro_quadrant_numeric"] = cache["macro_quadrant_numeric"].replace(-1, np.nan)
    result.columns_added.extend(["macro_quadrant", "macro_quadrant_numeric"])

    # Stability: how many of last N days are in the same quadrant
    def _rolling_stability(series: pd.Series, window: int) -> pd.Series:
        """Count how many of the last `window` days share the current label."""
        stability = pd.Series(np.nan, index=series.index)
        for i in range(len(series)):
            if i < window - 1:
                w = series.iloc[:i + 1]
            else:
                w = series.iloc[i - window + 1:i + 1]
            current = series.iloc[i]
            if current == "unknown":
                stability.iloc[i] = np.nan
            else:
                stability.iloc[i] = float((w == current).sum()) / len(w) * 100
        return stability

    cache["macro_quadrant_stability_21d"] = _rolling_stability(quadrants, _STABILITY_WINDOW)
    result.columns_added.append("macro_quadrant_stability_21d")

    # Missing flag
    cache["is_missing_macro_quadrant"] = (quadrants == "unknown").astype(int)
    result.columns_added.append("is_missing_macro_quadrant")

    # Summary stats
    valid = quadrants[quadrants != "unknown"]
    result.n_days_classified = len(valid)

    if len(valid) > 0:
        result.current_quadrant = str(quadrants.iloc[-1])
        dist = valid.value_counts(normalize=True)
        result.quadrant_distribution = {str(k): round(float(v), 4) for k, v in dist.items()}

        # Count transitions
        transitions = (valid != valid.shift(1)).sum() - 1  # first element doesn't count
        result.n_transitions = max(0, int(transitions))

    logger.info(
        "Macro quadrant: %s, %d days classified, %d transitions, distribution=%s",
        result.current_quadrant,
        result.n_days_classified,
        result.n_transitions,
        result.quadrant_distribution,
    )

    return cache, result
