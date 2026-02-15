"""T4.2 -- Vanity percentage computation.

Measures wasteful spending as a percentage of revenue, composed of
four independently testable components:

1. **Executive compensation excess**: executive comp > 5% of net income
   -> excess amount.
2. **SG&A bloat**: company SG&A ratio > 1.2x industry median SG&A ratio
   -> excess amount.
3. **Buyback waste**: buybacks while ``fcf_ttm < 0`` -> full buyback
   amount.
4. **Marketing excess during survival**: marketing spend > 10% revenue
   during survival mode -> excess amount.

``vanity_percentage = clip(total_vanity_spend / revenue * 100, 0, 100)``

Returns null (not 0) when revenue is null/zero.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import EPSILON

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component 1: Executive compensation excess
# ---------------------------------------------------------------------------


def _exec_comp_excess(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
) -> pd.Series:
    """Compute excess executive compensation.

    If ``exec_compensation > threshold_pct% * net_income``, the excess
    above the threshold is returned.  Otherwise 0.

    Returns NaN where either input is missing.
    """
    exec_comp = df.get("exec_compensation", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))

    threshold = net_income.abs() * (threshold_pct / 100.0)
    excess = exec_comp - threshold
    # Only count positive excess; null inputs -> null
    result = excess.clip(lower=0)
    result = result.where(exec_comp.notna() & net_income.notna(), other=np.nan)

    return result


# ---------------------------------------------------------------------------
# Component 2: SG&A bloat
# ---------------------------------------------------------------------------


def _sga_bloat(
    df: pd.DataFrame,
    bloat_factor: float = 1.2,
) -> pd.Series:
    """Compute SG&A bloat excess.

    If the company's SG&A-to-revenue ratio exceeds ``bloat_factor`` times
    the industry median SG&A ratio, the excess spend is returned.

    The industry median SG&A ratio can come from linked aggregates
    (``industry_peers_median_sga_ratio``) or a static column
    (``industry_median_sga_ratio``).

    Returns NaN where inputs are missing.
    """
    sga = df.get("sga_expense", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    industry_sga_ratio = df.get(
        "industry_median_sga_ratio",
        df.get(
            "industry_peers_median_sga_ratio",
            pd.Series(np.nan, index=df.index),
        ),
    )

    # Company SGA ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        company_ratio = sga / revenue
        company_ratio = company_ratio.replace([np.inf, -np.inf], np.nan)

    # Threshold = bloat_factor * industry median ratio
    threshold_ratio = bloat_factor * industry_sga_ratio

    # Excess = (company_ratio - threshold_ratio) * revenue, clipped to 0
    excess_ratio = (company_ratio - threshold_ratio).clip(lower=0)
    excess_amount = excess_ratio * revenue

    # Null where inputs are missing
    inputs_ok = sga.notna() & revenue.notna() & industry_sga_ratio.notna()
    result = excess_amount.where(inputs_ok, other=np.nan)

    return result


# ---------------------------------------------------------------------------
# Component 3: Buyback waste
# ---------------------------------------------------------------------------


def _buyback_waste(df: pd.DataFrame) -> pd.Series:
    """Compute buyback waste.

    If the company is buying back stock (``share_buybacks > 0``) while
    ``free_cash_flow_ttm_asof < 0``, the full buyback amount is waste.

    Returns NaN where inputs are missing.
    """
    buybacks = df.get("share_buybacks", pd.Series(np.nan, index=df.index))
    fcf_ttm = df.get(
        "free_cash_flow_ttm_asof",
        df.get("free_cash_flow", pd.Series(np.nan, index=df.index)),
    )

    # Buybacks are waste only when FCF TTM is negative
    waste = buybacks.where(
        buybacks.notna() & fcf_ttm.notna() & (buybacks > 0) & (fcf_ttm < 0),
        other=0.0,
    )
    # Preserve NaN where both inputs are missing
    waste = waste.where(buybacks.notna() | fcf_ttm.notna(), other=np.nan)

    return waste


# ---------------------------------------------------------------------------
# Component 4: Marketing excess during survival
# ---------------------------------------------------------------------------


def _marketing_excess(
    df: pd.DataFrame,
    threshold_pct: float = 10.0,
) -> pd.Series:
    """Compute marketing excess during survival mode.

    If ``marketing_spend > threshold_pct% * revenue`` AND the company
    is in survival mode, the excess above the threshold is waste.

    Returns NaN where inputs are missing.
    """
    marketing = df.get("marketing_expense", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    survival = df.get(
        "company_survival_mode_flag",
        pd.Series(0, index=df.index),
    )

    threshold = revenue * (threshold_pct / 100.0)
    excess = (marketing - threshold).clip(lower=0)

    # Only count during survival mode
    result = excess.where(survival == 1, other=0.0)
    # Null where inputs are missing
    result = result.where(
        marketing.notna() & revenue.notna(), other=np.nan,
    )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_vanity_percentage(
    df: pd.DataFrame,
    exec_comp_threshold: float = 5.0,
    sga_bloat_factor: float = 1.2,
    marketing_threshold: float = 10.0,
) -> pd.DataFrame:
    """Compute vanity percentage and attach to the feature table.

    Parameters
    ----------
    df:
        Daily feature table with derived variables and survival flags.
    exec_comp_threshold:
        Percentage of net income above which exec comp is excessive.
    sga_bloat_factor:
        Multiplier above industry median SG&A ratio.
    marketing_threshold:
        Percentage of revenue above which marketing is excessive
        during survival mode.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``vanity_exec_comp_excess``
        - ``vanity_sga_bloat``
        - ``vanity_buyback_waste``
        - ``vanity_marketing_excess``
        - ``vanity_total_spend``
        - ``vanity_percentage``
        - ``is_missing_vanity_percentage``
    """
    result = df.copy()

    # Compute each component
    result["vanity_exec_comp_excess"] = _exec_comp_excess(
        result, exec_comp_threshold,
    )
    result["vanity_sga_bloat"] = _sga_bloat(result, sga_bloat_factor)
    result["vanity_buyback_waste"] = _buyback_waste(result)
    result["vanity_marketing_excess"] = _marketing_excess(
        result, marketing_threshold,
    )

    # Total vanity spend -- sum of components, treating NaN as 0
    # (a missing component doesn't invalidate the total)
    components = [
        "vanity_exec_comp_excess",
        "vanity_sga_bloat",
        "vanity_buyback_waste",
        "vanity_marketing_excess",
    ]
    total = pd.Series(0.0, index=result.index)
    any_component = pd.Series(False, index=result.index)
    for comp in components:
        vals = result[comp]
        any_component = any_component | vals.notna()
        total = total + vals.fillna(0)

    # If no component had data, total is NaN
    result["vanity_total_spend"] = total.where(any_component, other=np.nan)

    # Vanity percentage = total_spend / revenue * 100, clipped 0-100
    revenue = result.get("revenue", pd.Series(np.nan, index=result.index))
    denom_ok = revenue.notna() & (revenue.abs() > EPSILON)

    with np.errstate(divide="ignore", invalid="ignore"):
        vp = (result["vanity_total_spend"] / revenue * 100)
        vp = vp.replace([np.inf, -np.inf], np.nan)

    vp = vp.clip(lower=0, upper=100)
    # Return null (not 0) when revenue is null/zero
    vp = vp.where(denom_ok, other=np.nan)

    result["vanity_percentage"] = vp
    result["is_missing_vanity_percentage"] = vp.isna().astype(int)

    triggered = (vp.notna() & (vp > 0)).sum()
    logger.info(
        "Vanity percentage: %d / %d days with non-zero vanity (max=%.1f%%)",
        triggered, len(vp),
        vp.max() if vp.notna().any() else 0.0,
    )

    return result
