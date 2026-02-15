"""T4.1 -- Survival mode detection (Step F).

Computes three daily boolean flags:

- **company_survival_mode_flag**: triggered when ANY of the following
  are true for a given day ``t``:
    - ``current_ratio < 1.0``
    - ``debt_to_equity_abs > 3.0``
    - ``fcf_yield < 0``
    - ``drawdown_252d < -0.40``

- **country_survival_mode_flag**: triggered when ANY config-driven
  macro threshold is breached (credit spread, unemployment, yield
  curve, FX volatility).  Since many of these macro proxies may not
  be available from World Bank data, the flag conservatively defaults
  to 0 when the data is missing.

- **country_protected_flag**: triggered when ANY of:
    - Target sector appears in ``strategic_sectors`` config list
    - Market cap > ``market_cap_gdp_threshold`` * GDP
    - Emergency rate cut > threshold within lookback window

All thresholds are loaded from ``config/country_protection_rules.yml``
-- nothing is hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Company survival
# ---------------------------------------------------------------------------

# Default thresholds (overridable via config if needed in the future)
_COMPANY_THRESHOLDS = {
    "current_ratio_lt": 1.0,
    "debt_to_equity_abs_gt": 3.0,
    "fcf_yield_lt": 0.0,
    "drawdown_252d_lt": -0.40,
}


def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """Compute daily company survival mode flag.

    Parameters
    ----------
    df:
        Daily feature table with derived variables already computed.
    thresholds:
        Override default thresholds (mainly for testing).

    Returns
    -------
    pd.Series
        Integer series: 1 = survival mode active, 0 = normal.
    """
    t = thresholds or _COMPANY_THRESHOLDS

    conditions: list[pd.Series] = []

    # Current ratio < 1.0
    if "current_ratio" in df.columns:
        cr = df["current_ratio"]
        conditions.append(cr.notna() & (cr < t.get("current_ratio_lt", 1.0)))

    # Debt-to-equity (absolute) > 3.0
    if "debt_to_equity_abs" in df.columns:
        de = df["debt_to_equity_abs"]
        conditions.append(de.notna() & (de > t.get("debt_to_equity_abs_gt", 3.0)))

    # FCF yield < 0
    if "fcf_yield" in df.columns:
        fy = df["fcf_yield"]
        conditions.append(fy.notna() & (fy < t.get("fcf_yield_lt", 0.0)))

    # Drawdown < -40%
    if "drawdown_252d" in df.columns:
        dd = df["drawdown_252d"]
        conditions.append(dd.notna() & (dd < t.get("drawdown_252d_lt", -0.40)))

    if not conditions:
        logger.warning("No company survival trigger columns found -- defaulting to 0")
        return pd.Series(0, index=df.index, name="company_survival_mode_flag")

    # ANY condition true triggers survival mode
    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "company_survival_mode_flag"

    triggered = flag.sum()
    logger.info(
        "Company survival flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


# ---------------------------------------------------------------------------
# Country survival
# ---------------------------------------------------------------------------


def compute_country_survival_flag(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute daily country survival mode flag.

    Uses config-driven thresholds from ``country_protection_rules.yml``.
    Since many macro proxies (credit spread, yield curve, FX vol) may
    not be available from World Bank data, the flag defaults to 0 when
    data is missing.

    Parameters
    ----------
    df:
        Daily feature table with macro variables aligned.
    config:
        Override config dict (loads from YAML if None).

    Returns
    -------
    pd.Series
        Integer series: 1 = country survival, 0 = normal.
    """
    if config is None:
        config = load_config("country_protection_rules")

    cs_cfg = config.get("country_survival", {})
    conditions: list[pd.Series] = []

    # Credit spread > threshold
    credit_spread_threshold = cs_cfg.get("credit_spread_pct", 5.0)
    if "credit_spread" in df.columns:
        cs = df["credit_spread"]
        conditions.append(cs.notna() & (cs > credit_spread_threshold))

    # Unemployment rise > threshold in lookback window
    unemp_rise_threshold = cs_cfg.get("unemployment_rise_pct", 3.0)
    unemp_months = cs_cfg.get("unemployment_rise_months", 6)
    if "unemployment_rate" in df.columns:
        unemp = df["unemployment_rate"]
        # Approximate N months as N*21 business days
        lookback_days = unemp_months * 21
        unemp_change = unemp - unemp.shift(lookback_days)
        conditions.append(unemp_change.notna() & (unemp_change > unemp_rise_threshold))

    # Yield curve slope < threshold
    yc_threshold = cs_cfg.get("yield_curve_slope", -0.5)
    if "yield_curve_slope" in df.columns:
        yc = df["yield_curve_slope"]
        conditions.append(yc.notna() & (yc < yc_threshold))

    # FX volatility > threshold
    fx_vol_threshold = cs_cfg.get("fx_volatility_pct", 20.0)
    if "fx_volatility" in df.columns:
        fxv = df["fx_volatility"]
        conditions.append(fxv.notna() & (fxv > fx_vol_threshold))

    if not conditions:
        logger.info(
            "No country survival macro proxies available -- defaulting to 0"
        )
        return pd.Series(0, index=df.index, name="country_survival_mode_flag")

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "country_survival_mode_flag"

    triggered = flag.sum()
    logger.info(
        "Country survival flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


# ---------------------------------------------------------------------------
# Country protection
# ---------------------------------------------------------------------------


def compute_country_protected_flag(
    df: pd.DataFrame,
    target_sector: str,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute daily country protection flag.

    A company is considered "country-protected" (and thus shielded from
    full country-survival weight shifts) if ANY of:
      - Sector is in the ``strategic_sectors`` config list
      - ``market_cap > market_cap_gdp_threshold * gdp_current_usd``
      - Emergency rate cut exceeds threshold in lookback window

    Parameters
    ----------
    df:
        Daily feature table with macro variables.
    target_sector:
        The target company's sector string.
    config:
        Override config dict (loads from YAML if None).

    Returns
    -------
    pd.Series
        Integer series: 1 = protected, 0 = not protected.
    """
    if config is None:
        config = load_config("country_protection_rules")

    conditions: list[pd.Series] = []

    # 1. Strategic sector
    strategic = config.get("strategic_sectors", [])
    sector_lower = target_sector.lower().strip()
    is_strategic = any(s.lower().strip() == sector_lower for s in strategic)
    if is_strategic:
        # Entire series is 1
        conditions.append(pd.Series(True, index=df.index))
        logger.info(
            "Sector '%s' is strategic -- country protection triggered globally",
            target_sector,
        )

    # 2. Market cap > threshold * GDP
    mc_gdp_threshold = config.get("market_cap_gdp_threshold", 0.001)
    if "market_cap" in df.columns and "gdp_current_usd" in df.columns:
        mc = df["market_cap"]
        gdp = df["gdp_current_usd"]
        # Both must be available for this check
        mc_condition = mc.notna() & gdp.notna() & (mc > mc_gdp_threshold * gdp)
        conditions.append(mc_condition)

    # 3. Emergency rate cut
    rate_cut_threshold = config.get("emergency_rate_cut_threshold", 2.0)
    rate_cut_months = config.get("emergency_rate_cut_months", 3)
    if "lending_interest_rate" in df.columns:
        rate = df["lending_interest_rate"]
        lookback_days = rate_cut_months * 21
        rate_change = rate.shift(lookback_days) - rate  # cut = positive change
        conditions.append(
            rate_change.notna() & (rate_change > rate_cut_threshold)
        )

    if not conditions:
        logger.info("No country protection conditions met -- defaulting to 0")
        return pd.Series(0, index=df.index, name="country_protected_flag")

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "country_protected_flag"

    triggered = flag.sum()
    logger.info(
        "Country protected flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


# ---------------------------------------------------------------------------
# Public API -- compute all survival flags
# ---------------------------------------------------------------------------


def compute_survival_flags(
    df: pd.DataFrame,
    target_sector: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute all three survival flags and attach to the feature table.

    Parameters
    ----------
    df:
        Daily feature table (target) with derived variables and macro
        data already merged.
    target_sector:
        Target company sector.
    config:
        Override country protection config.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``company_survival_mode_flag``
        - ``country_survival_mode_flag``
        - ``country_protected_flag``
    """
    result = df.copy()

    result["company_survival_mode_flag"] = compute_company_survival_flag(df)
    result["country_survival_mode_flag"] = compute_country_survival_flag(
        df, config,
    )
    result["country_protected_flag"] = compute_country_protected_flag(
        df, target_sector, config,
    )

    logger.info("All survival flags computed and attached to feature table")
    return result
