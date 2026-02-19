"""Financial health scoring -- Altman Z-Score, Earnings Quality, and
Liquidity Stress Testing.

Three complementary models that use data already in the cache (no API
calls) to strengthen the app's survival analysis and ethical filters:

1. **Altman Z-Score** -- Classic bankruptcy prediction model (1968).
   Computes a weighted sum of five balance-sheet ratios.  Scores below
   1.81 indicate high distress; above 2.99 is safe.  The "grey zone"
   (1.81-2.99) is uncertain.  Computed daily using as-of financials.

2. **Beneish M-Score** -- Earnings manipulation detector (1999).
   Flags companies likely cooking their books using 8 financial ratios.
   Score > -1.78 suggests manipulation.  Serves the ethical filter
   mission: if earnings are fake, downstream analysis is unreliable.

3. **Liquidity Runway** -- Months of cash runway remaining based on
   current burn rate.  Answers "how long can this company survive at
   current spending without new revenue or financing?"

All three feed into the company profile and final report as additional
survival signals alongside the existing triggers.

Integration points:
  - ``operator1/analysis/survival_mode.py`` (signals)
  - ``operator1/report/profile_builder.py`` (profile JSON)

Top-level entry points:
  - ``compute_altman_z_score`` -- daily Z-Score series
  - ``compute_beneish_m_score`` -- earnings quality assessment
  - ``compute_liquidity_runway`` -- months of cash runway
  - ``compute_financial_health`` -- runs all three, returns combined result
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Epsilon to avoid division by zero
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AltmanZResult:
    """Altman Z-Score computation result."""

    z_score_series: pd.Series | None = None
    latest_z_score: float | None = None
    zone: str = "unknown"          # "safe", "grey", "distress", "unknown"
    components: dict[str, float] = field(default_factory=dict)
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "latest_z_score": self.latest_z_score,
            "zone": self.zone,
            "components": self.components,
        }


@dataclass
class BeneishMResult:
    """Beneish M-Score computation result."""

    m_score: float | None = None
    likely_manipulator: bool = False
    verdict: str = "unknown"       # "unlikely", "possible", "likely", "unknown"
    components: dict[str, float] = field(default_factory=dict)
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "m_score": self.m_score,
            "likely_manipulator": self.likely_manipulator,
            "verdict": self.verdict,
            "components": self.components,
        }


@dataclass
class LiquidityRunwayResult:
    """Liquidity runway estimation result."""

    months_of_runway: float | None = None
    verdict: str = "unknown"       # "critical", "tight", "adequate", "strong", "unknown"
    cash_available: float | None = None
    monthly_burn_rate: float | None = None
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "months_of_runway": self.months_of_runway,
            "verdict": self.verdict,
            "cash_available": self.cash_available,
            "monthly_burn_rate": self.monthly_burn_rate,
        }


@dataclass
class FinancialHealthResult:
    """Combined result from all financial health checks."""

    altman_z: AltmanZResult = field(default_factory=AltmanZResult)
    beneish_m: BeneishMResult = field(default_factory=BeneishMResult)
    liquidity_runway: LiquidityRunwayResult = field(default_factory=LiquidityRunwayResult)
    overall_health_score: float | None = None  # 0-100 composite
    overall_verdict: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "altman_z": self.altman_z.to_dict(),
            "beneish_m": self.beneish_m.to_dict(),
            "liquidity_runway": self.liquidity_runway.to_dict(),
            "overall_health_score": self.overall_health_score,
            "overall_verdict": self.overall_verdict,
        }


# ---------------------------------------------------------------------------
# 1. Altman Z-Score
# ---------------------------------------------------------------------------

# Original Altman (1968) coefficients for public manufacturing companies.
# Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
#
# X1 = Working Capital / Total Assets         (liquidity)
# X2 = Retained Earnings / Total Assets       (leverage/age)
# X3 = EBIT / Total Assets                    (profitability)
# X4 = Market Cap / Total Liabilities         (solvency)
# X5 = Revenue / Total Assets                 (efficiency)

_Z_COEFF = {
    "x1_working_capital_ta": 1.2,
    "x2_retained_earnings_ta": 1.4,
    "x3_ebit_ta": 3.3,
    "x4_market_cap_tl": 0.6,
    "x5_revenue_ta": 1.0,
}

_Z_SAFE_THRESHOLD = 2.99
_Z_DISTRESS_THRESHOLD = 1.81


def compute_altman_z_score(df: pd.DataFrame) -> AltmanZResult:
    """Compute the daily Altman Z-Score series.

    Uses as-of financial data already aligned in the cache:
      - ``current_assets``, ``current_liabilities`` -> working capital
      - ``total_equity`` as proxy for retained earnings
      - ``ebit`` (or ``ebitda`` fallback)
      - ``market_cap``, ``total_liabilities``
      - ``revenue``, ``total_assets``

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature table with as-of financials.

    Returns
    -------
    AltmanZResult
        Z-Score series and classification.
    """
    result = AltmanZResult()

    total_assets = df.get("total_assets")
    if total_assets is None or total_assets.notna().sum() == 0:
        result.error = "total_assets not available"
        return result

    # Denominators -- guard against zero
    ta = total_assets.replace(0, np.nan)
    tl = df.get("total_liabilities", pd.Series(np.nan, index=df.index)).replace(0, np.nan)

    # X1: Working Capital / Total Assets
    current_assets = df.get("current_assets", pd.Series(np.nan, index=df.index))
    current_liabilities = df.get("current_liabilities", pd.Series(np.nan, index=df.index))
    working_capital = current_assets - current_liabilities
    x1 = working_capital / ta

    # X2: Retained Earnings / Total Assets
    # Retained earnings is not directly available; use total_equity as a proxy
    # (retained earnings is the largest component of equity for mature companies)
    total_equity = df.get("total_equity", pd.Series(np.nan, index=df.index))
    x2 = total_equity / ta

    # X3: EBIT / Total Assets
    ebit = df.get("ebit", df.get("ebitda", pd.Series(np.nan, index=df.index)))
    x3 = ebit / ta

    # X4: Market Cap / Total Liabilities
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))
    x4 = market_cap / tl

    # X5: Revenue / Total Assets
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    x5 = revenue / ta

    # Compute Z-Score
    z = (
        _Z_COEFF["x1_working_capital_ta"] * x1
        + _Z_COEFF["x2_retained_earnings_ta"] * x2
        + _Z_COEFF["x3_ebit_ta"] * x3
        + _Z_COEFF["x4_market_cap_tl"] * x4
        + _Z_COEFF["x5_revenue_ta"] * x5
    )

    result.z_score_series = z
    result.available = z.notna().any()

    if result.available:
        latest = z.dropna().iloc[-1] if z.notna().any() else None
        result.latest_z_score = float(latest) if latest is not None else None

        if result.latest_z_score is not None:
            if result.latest_z_score >= _Z_SAFE_THRESHOLD:
                result.zone = "safe"
            elif result.latest_z_score <= _Z_DISTRESS_THRESHOLD:
                result.zone = "distress"
            else:
                result.zone = "grey"

        # Store latest component values for the report
        for label, series in [
            ("x1_working_capital_ta", x1),
            ("x2_retained_earnings_ta", x2),
            ("x3_ebit_ta", x3),
            ("x4_market_cap_tl", x4),
            ("x5_revenue_ta", x5),
        ]:
            val = series.dropna().iloc[-1] if series.notna().any() else None
            result.components[label] = float(val) if val is not None else None

    logger.info(
        "Altman Z-Score: %.2f (%s)",
        result.latest_z_score or 0.0, result.zone,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Beneish M-Score (earnings quality)
# ---------------------------------------------------------------------------

# Beneish (1999) coefficients.  M > -1.78 suggests manipulation.
# M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
#     + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
#
# We compute these from year-over-year (or period-over-period) changes
# in the financial data.

_M_INTERCEPT = -4.84
_M_COEFFS = {
    "DSRI": 0.920,   # Days Sales in Receivables Index
    "GMI": 0.528,    # Gross Margin Index
    "AQI": 0.404,    # Asset Quality Index
    "SGI": 0.892,    # Sales Growth Index
    "DEPI": 0.115,   # Depreciation Index (approx from EBIT-EBITDA gap)
    "SGAI": -0.172,  # SGA Index (use operating expenses as proxy)
    "TATA": 4.679,   # Total Accruals to Total Assets
    "LVGI": -0.327,  # Leverage Index
}

_M_THRESHOLD = -1.78  # scores above this suggest manipulation


def compute_beneish_m_score(df: pd.DataFrame) -> BeneishMResult:
    """Compute the Beneish M-Score for earnings manipulation detection.

    Uses period-over-period changes in financial ratios.  Since the cache
    has daily as-of data (financial statements change quarterly), we
    compare the latest statement period against the previous one.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature table with as-of financials.

    Returns
    -------
    BeneishMResult
        M-Score and manipulation verdict.
    """
    result = BeneishMResult()

    # We need at least two distinct statement periods to compute changes.
    # Use revenue changes as a proxy for period transitions.
    revenue = df.get("revenue")
    if revenue is None or revenue.notna().sum() < 2:
        result.error = "Insufficient revenue data for M-Score"
        return result

    # Find distinct financial periods (where revenue changes)
    rev_clean = revenue.dropna()
    rev_changes = rev_clean.diff().abs()
    period_breaks = rev_changes[rev_changes > _EPS].index

    if len(period_breaks) < 1:
        result.error = "No distinct financial periods detected"
        return result

    # Use the last period break to define "current" vs "prior" periods
    break_point = period_breaks[-1]
    prior_mask = df.index < break_point
    current_mask = df.index >= break_point

    if prior_mask.sum() == 0 or current_mask.sum() == 0:
        result.error = "Cannot split data into prior/current periods"
        return result

    def _period_val(col_name: str, mask: pd.Series) -> float | None:
        """Get the mean value of a column over a period mask."""
        col = df.get(col_name)
        if col is None:
            return None
        vals = col.loc[mask].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    # Helper: safe ratio
    def _safe_ratio(num: float | None, denom: float | None) -> float | None:
        if num is None or denom is None or abs(denom) < _EPS:
            return None
        return num / denom

    # --- Compute the 8 Beneish indices ---
    components: dict[str, float | None] = {}

    # DSRI: Days Sales in Receivables Index
    # (Receivables_t/Revenue_t) / (Receivables_t-1/Revenue_t-1)
    recv_curr = _period_val("receivables", current_mask)
    recv_prior = _period_val("receivables", prior_mask)
    rev_curr = _period_val("revenue", current_mask)
    rev_prior = _period_val("revenue", prior_mask)
    dsri_curr = _safe_ratio(recv_curr, rev_curr)
    dsri_prior = _safe_ratio(recv_prior, rev_prior)
    components["DSRI"] = _safe_ratio(dsri_curr, dsri_prior) if dsri_curr and dsri_prior else None

    # GMI: Gross Margin Index
    # GM_t-1 / GM_t  (decline in margins = red flag)
    gp_curr = _period_val("gross_profit", current_mask)
    gp_prior = _period_val("gross_profit", prior_mask)
    gm_curr = _safe_ratio(gp_curr, rev_curr)
    gm_prior = _safe_ratio(gp_prior, rev_prior)
    components["GMI"] = _safe_ratio(gm_prior, gm_curr) if gm_curr and gm_prior else None

    # AQI: Asset Quality Index
    # (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_t-1 + PPE_t-1) / TA_t-1)
    # Without PPE, use: (1 - CA/TA) as proxy for non-current intangible ratio
    ca_curr = _period_val("current_assets", current_mask)
    ca_prior = _period_val("current_assets", prior_mask)
    ta_curr = _period_val("total_assets", current_mask)
    ta_prior = _period_val("total_assets", prior_mask)
    aqi_curr = 1.0 - (_safe_ratio(ca_curr, ta_curr) or 0.0) if ta_curr else None
    aqi_prior = 1.0 - (_safe_ratio(ca_prior, ta_prior) or 0.0) if ta_prior else None
    components["AQI"] = _safe_ratio(aqi_curr, aqi_prior) if aqi_curr is not None and aqi_prior is not None else None

    # SGI: Sales Growth Index
    # Revenue_t / Revenue_t-1
    components["SGI"] = _safe_ratio(rev_curr, rev_prior)

    # DEPI: Depreciation Index (proxy)
    # Use (EBITDA - EBIT) as depreciation proxy
    ebit_curr = _period_val("ebit", current_mask)
    ebit_prior = _period_val("ebit", prior_mask)
    ebitda_curr = _period_val("ebitda", current_mask)
    ebitda_prior = _period_val("ebitda", prior_mask)
    dep_curr = (ebitda_curr - ebit_curr) if ebitda_curr is not None and ebit_curr is not None else None
    dep_prior = (ebitda_prior - ebit_prior) if ebitda_prior is not None and ebit_prior is not None else None
    dep_rate_curr = _safe_ratio(dep_curr, (dep_curr + (ta_curr or 0))) if dep_curr is not None else None
    dep_rate_prior = _safe_ratio(dep_prior, (dep_prior + (ta_prior or 0))) if dep_prior is not None else None
    components["DEPI"] = _safe_ratio(dep_rate_prior, dep_rate_curr) if dep_rate_curr and dep_rate_prior else None

    # SGAI: SGA Expense Index (proxy via operating expenses)
    # Use (revenue - ebit) as SGA proxy
    sga_curr = (rev_curr - ebit_curr) if rev_curr is not None and ebit_curr is not None else None
    sga_prior = (rev_prior - ebit_prior) if rev_prior is not None and ebit_prior is not None else None
    sgai_curr = _safe_ratio(sga_curr, rev_curr)
    sgai_prior = _safe_ratio(sga_prior, rev_prior)
    components["SGAI"] = _safe_ratio(sgai_curr, sgai_prior) if sgai_curr and sgai_prior else None

    # TATA: Total Accruals to Total Assets
    # (Net Income - Operating Cash Flow) / Total Assets
    ni_curr = _period_val("net_income", current_mask)
    ocf_curr = _period_val("operating_cash_flow", current_mask)
    if ni_curr is not None and ocf_curr is not None and ta_curr is not None and abs(ta_curr) > _EPS:
        components["TATA"] = (ni_curr - ocf_curr) / ta_curr
    else:
        components["TATA"] = None

    # LVGI: Leverage Index
    # (TL_t / TA_t) / (TL_t-1 / TA_t-1)
    tl_curr = _period_val("total_liabilities", current_mask)
    tl_prior = _period_val("total_liabilities", prior_mask)
    lev_curr = _safe_ratio(tl_curr, ta_curr)
    lev_prior = _safe_ratio(tl_prior, ta_prior)
    components["LVGI"] = _safe_ratio(lev_curr, lev_prior) if lev_curr and lev_prior else None

    # --- Compute M-Score ---
    m_score = _M_INTERCEPT
    n_available = 0
    for key, coeff in _M_COEFFS.items():
        val = components.get(key)
        if val is not None and np.isfinite(val):
            m_score += coeff * val
            n_available += 1

    if n_available < 4:
        result.error = f"Only {n_available}/8 M-Score components available"
        return result

    result.m_score = float(m_score)
    result.available = True
    result.components = {k: float(v) if v is not None else None for k, v in components.items()}
    result.likely_manipulator = m_score > _M_THRESHOLD

    if m_score > -1.78:
        result.verdict = "likely"
    elif m_score > -2.22:
        result.verdict = "possible"
    else:
        result.verdict = "unlikely"

    logger.info(
        "Beneish M-Score: %.2f (%s manipulation)",
        result.m_score, result.verdict,
    )
    return result


# ---------------------------------------------------------------------------
# 3. Liquidity Runway
# ---------------------------------------------------------------------------


def compute_liquidity_runway(df: pd.DataFrame) -> LiquidityRunwayResult:
    """Estimate months of cash runway at current burn rate.

    Answers: "If revenue stopped today, how many months can this company
    survive on its current cash reserves at the current spending rate?"

    Burn rate = abs(operating_cash_flow) when OCF is negative,
    or (operating_expenses - revenue) as a proxy.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature table with as-of financials.

    Returns
    -------
    LiquidityRunwayResult
    """
    result = LiquidityRunwayResult()

    cash = df.get("cash_and_equivalents")
    if cash is None or cash.notna().sum() == 0:
        result.error = "cash_and_equivalents not available"
        return result

    latest_cash = cash.dropna().iloc[-1]
    result.cash_available = float(latest_cash)

    # Compute monthly burn rate from operating cash flow
    ocf = df.get("operating_cash_flow")
    if ocf is not None and ocf.notna().sum() > 0:
        latest_ocf = float(ocf.dropna().iloc[-1])

        if latest_ocf < 0:
            # Company is burning cash -- OCF is negative
            monthly_burn = abs(latest_ocf) / 12.0  # annual -> monthly
            result.monthly_burn_rate = float(monthly_burn)
        else:
            # Positive OCF -- company is generating cash
            # Use capex as a proxy for minimum spend
            capex = df.get("capex")
            if capex is not None and capex.notna().sum() > 0:
                latest_capex = float(capex.dropna().iloc[-1])
                net_outflow = abs(latest_capex) - latest_ocf
                if net_outflow > 0:
                    result.monthly_burn_rate = float(net_outflow / 12.0)
                else:
                    # Company generates more cash than it spends
                    result.monthly_burn_rate = 0.0
            else:
                result.monthly_burn_rate = 0.0
    else:
        result.error = "operating_cash_flow not available for burn rate"
        return result

    result.available = True

    if result.monthly_burn_rate is not None and result.monthly_burn_rate > _EPS:
        result.months_of_runway = float(latest_cash / result.monthly_burn_rate)
    elif result.monthly_burn_rate == 0.0:
        result.months_of_runway = float("inf")
    else:
        result.months_of_runway = None

    # Verdict
    runway = result.months_of_runway
    if runway is None:
        result.verdict = "unknown"
    elif runway == float("inf") or runway > 24:
        result.verdict = "strong"
    elif runway > 12:
        result.verdict = "adequate"
    elif runway > 6:
        result.verdict = "tight"
    else:
        result.verdict = "critical"

    logger.info(
        "Liquidity runway: %.1f months (%s)",
        result.months_of_runway if result.months_of_runway is not None and result.months_of_runway != float("inf") else -1,
        result.verdict,
    )
    return result


# ---------------------------------------------------------------------------
# 4. Combined financial health assessment
# ---------------------------------------------------------------------------


def compute_financial_health(df: pd.DataFrame) -> FinancialHealthResult:
    """Run all financial health checks and produce a composite score.

    The composite health score (0-100) is a weighted average:
      - Altman Z-Score: 40% (survival prediction)
      - Beneish M-Score: 25% (earnings integrity)
      - Liquidity Runway: 35% (cash sustainability)

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature table with as-of financials.

    Returns
    -------
    FinancialHealthResult
        Combined results from all three models.
    """
    result = FinancialHealthResult()

    # Run individual models
    result.altman_z = compute_altman_z_score(df)
    result.beneish_m = compute_beneish_m_score(df)
    result.liquidity_runway = compute_liquidity_runway(df)

    # Compute composite score
    scores: list[tuple[float, float]] = []  # (score, weight)

    # Z-Score contribution (0-100 scale)
    if result.altman_z.available and result.altman_z.latest_z_score is not None:
        z = result.altman_z.latest_z_score
        # Map Z-Score to 0-100: 0 or below -> 0, 3+ -> 100
        z_score_normalized = max(0.0, min(100.0, (z / 3.0) * 100.0))
        scores.append((z_score_normalized, 0.40))

    # M-Score contribution (0-100 scale, inverted: lower M = healthier)
    if result.beneish_m.available and result.beneish_m.m_score is not None:
        m = result.beneish_m.m_score
        # Map M-Score to 0-100: -3.0 -> 100 (clean), -1.0 -> 0 (manipulator)
        m_score_normalized = max(0.0, min(100.0, ((m + 1.0) / -2.0) * 100.0 + 50.0))
        scores.append((m_score_normalized, 0.25))

    # Runway contribution (0-100 scale)
    if result.liquidity_runway.available and result.liquidity_runway.months_of_runway is not None:
        runway = result.liquidity_runway.months_of_runway
        if runway == float("inf"):
            r_score = 100.0
        else:
            # Map months to 0-100: 0 months -> 0, 24+ months -> 100
            r_score = max(0.0, min(100.0, (runway / 24.0) * 100.0))
        scores.append((r_score, 0.35))

    if scores:
        total_weight = sum(w for _, w in scores)
        if total_weight > 0:
            result.overall_health_score = sum(s * w for s, w in scores) / total_weight

    # Overall verdict
    if result.overall_health_score is not None:
        if result.overall_health_score >= 75:
            result.overall_verdict = "healthy"
        elif result.overall_health_score >= 50:
            result.overall_verdict = "moderate"
        elif result.overall_health_score >= 25:
            result.overall_verdict = "weak"
        else:
            result.overall_verdict = "critical"
    else:
        result.overall_verdict = "insufficient_data"

    logger.info(
        "Financial health: score=%.1f (%s)",
        result.overall_health_score or 0.0, result.overall_verdict,
    )
    return result
