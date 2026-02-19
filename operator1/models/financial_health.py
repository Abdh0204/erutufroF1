"""Financial Health Scoring -- daily composite scores injected into cache.

Computes per-day financial health scores across the five survival tiers
so that downstream temporal models (forecasting, forward pass, burn-out)
automatically learn from them.  Each score is a 0-100 normalized value
written as a daily column in the cache DataFrame.

**Scores produced (all daily columns):**

- ``fh_liquidity_score``     -- Tier 1: cash, FCF, operating CF
- ``fh_solvency_score``      -- Tier 2: debt ratios, interest coverage
- ``fh_stability_score``     -- Tier 3: volatility, drawdown
- ``fh_profitability_score`` -- Tier 4: margins
- ``fh_growth_score``        -- Tier 5: revenue trend, valuation
- ``fh_composite_score``     -- Weighted blend of tiers 1-5
- ``fh_composite_label``     -- Categorical: Critical / Weak / Fair / Strong / Excellent

The composite uses hierarchy weights when provided, so a company in
survival mode will have its composite dominated by liquidity/solvency,
exactly matching how the temporal models should prioritize learning.

Top-level entry point:
    ``compute_financial_health(cache, hierarchy_weights=None)``

Spec refs: Sec 17 (temporal learning), Sec C.3-C.4 (hierarchy weights)
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

# Default equal weights across 5 tiers (matches normal regime)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "tier1": 0.20,
    "tier2": 0.20,
    "tier3": 0.20,
    "tier4": 0.20,
    "tier5": 0.20,
}

# Composite label thresholds
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (20.0, "Critical"),
    (40.0, "Weak"),
    (60.0, "Fair"),
    (80.0, "Strong"),
    (100.1, "Excellent"),
]

# Rolling window for trend-based scoring (business days)
_TREND_WINDOW: int = 63  # ~3 months


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FinancialHealthResult:
    """Summary statistics from the financial health computation."""

    columns_added: list[str] = field(default_factory=list)
    latest_composite: float = float("nan")
    latest_label: str = "Unknown"
    mean_composite: float = float("nan")
    tier_means: dict[str, float] = field(default_factory=dict)
    n_days_scored: int = 0


# ---------------------------------------------------------------------------
# Individual tier scoring helpers
# ---------------------------------------------------------------------------


def _normalize_series(
    s: pd.Series,
    *,
    lower: float | None = None,
    upper: float | None = None,
    invert: bool = False,
) -> pd.Series:
    """Normalize a series to 0-100 using rolling percentile rank.

    Parameters
    ----------
    s : pd.Series
        Raw metric series.
    lower, upper : float, optional
        If provided, clip the series before normalizing. Useful for
        known-range metrics like ratios.
    invert : bool
        If True, higher raw values map to *lower* scores (e.g. debt ratios,
        volatility).

    Returns
    -------
    pd.Series
        Normalized 0-100 score (NaN where input is NaN).
    """
    if s.isna().all():
        return pd.Series(np.nan, index=s.index)

    work = s.copy()
    if lower is not None or upper is not None:
        work = work.clip(lower=lower, upper=upper)

    # Use expanding percentile rank for a stable, non-leaking normalization
    ranked = work.expanding(min_periods=1).rank(pct=True) * 100.0

    if invert:
        ranked = 100.0 - ranked

    return ranked


def _score_liquidity(cache: pd.DataFrame) -> pd.Series:
    """Tier 1 -- Liquidity & Cash score.

    Looks at: cash_ratio, free_cash_flow_ttm (or operating_cash_flow),
    cash_and_equivalents.
    """
    components: list[pd.Series] = []

    if "cash_ratio" in cache.columns:
        components.append(
            _normalize_series(cache["cash_ratio"], lower=0, upper=10)
        )

    if "free_cash_flow_ttm" in cache.columns:
        components.append(
            _normalize_series(cache["free_cash_flow_ttm"])
        )
    elif "operating_cash_flow_asof" in cache.columns:
        components.append(
            _normalize_series(cache["operating_cash_flow_asof"])
        )
    elif "operating_cash_flow" in cache.columns:
        components.append(
            _normalize_series(cache["operating_cash_flow"])
        )

    if "cash_and_equivalents_asof" in cache.columns:
        components.append(
            _normalize_series(cache["cash_and_equivalents_asof"])
        )
    elif "cash_and_equivalents" in cache.columns:
        components.append(
            _normalize_series(cache["cash_and_equivalents"])
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_liquidity_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_liquidity_score"
    return score


def _score_solvency(cache: pd.DataFrame) -> pd.Series:
    """Tier 2 -- Debt & Solvency score.

    Looks at: debt_to_equity (inverted), net_debt_to_ebitda (inverted),
    interest_coverage, current_ratio.
    """
    components: list[pd.Series] = []

    if "debt_to_equity" in cache.columns:
        components.append(
            _normalize_series(cache["debt_to_equity"], invert=True)
        )

    if "net_debt_to_ebitda" in cache.columns:
        components.append(
            _normalize_series(cache["net_debt_to_ebitda"], invert=True)
        )

    if "interest_coverage" in cache.columns:
        components.append(
            _normalize_series(cache["interest_coverage"], lower=0, upper=50)
        )

    if "current_ratio" in cache.columns:
        components.append(
            _normalize_series(cache["current_ratio"], lower=0, upper=10)
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_solvency_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_solvency_score"
    return score


def _score_stability(cache: pd.DataFrame) -> pd.Series:
    """Tier 3 -- Market Stability score.

    Looks at: volatility_21d (inverted), drawdown_252d (inverted), volume.
    """
    components: list[pd.Series] = []

    if "volatility_21d" in cache.columns:
        components.append(
            _normalize_series(cache["volatility_21d"], invert=True)
        )

    if "drawdown_252d" in cache.columns:
        # Drawdown is typically negative; more negative = worse
        components.append(
            _normalize_series(cache["drawdown_252d"])
        )

    if "volume" in cache.columns:
        components.append(
            _normalize_series(cache["volume"])
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_stability_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_stability_score"
    return score


def _score_profitability(cache: pd.DataFrame) -> pd.Series:
    """Tier 4 -- Profitability score.

    Looks at: gross_margin, operating_margin, net_margin.
    """
    components: list[pd.Series] = []

    for col in ("gross_margin", "operating_margin", "net_margin"):
        if col in cache.columns:
            components.append(
                _normalize_series(cache[col])
            )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_profitability_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_profitability_score"
    return score


def _score_growth(cache: pd.DataFrame) -> pd.Series:
    """Tier 5 -- Growth & Valuation score.

    Looks at: revenue trend (rolling % change), pe_ratio (inverted --
    lower PE = cheaper = higher score), ev_to_ebitda (inverted).
    """
    components: list[pd.Series] = []

    # Revenue growth trend
    if "revenue_asof" in cache.columns:
        rev = cache["revenue_asof"]
        rev_growth = rev.pct_change(periods=_TREND_WINDOW)
        components.append(_normalize_series(rev_growth))
    elif "revenue" in cache.columns:
        rev = cache["revenue"]
        rev_growth = rev.pct_change(periods=_TREND_WINDOW)
        components.append(_normalize_series(rev_growth))

    if "pe_ratio" in cache.columns:
        # Low PE -> potentially undervalued -> higher score
        pe = cache["pe_ratio"].clip(lower=0, upper=200)
        components.append(_normalize_series(pe, invert=True))

    if "ev_to_ebitda" in cache.columns:
        ev = cache["ev_to_ebitda"].clip(lower=0, upper=100)
        components.append(_normalize_series(ev, invert=True))

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_growth_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_growth_score"
    return score


def _composite_label(score: float) -> str:
    """Map a composite score to a categorical label."""
    if np.isnan(score):
        return "Unknown"
    for threshold, label in _LABEL_THRESHOLDS:
        if score < threshold:
            return label
    return "Excellent"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_financial_health(
    cache: pd.DataFrame,
    hierarchy_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, FinancialHealthResult]:
    """Compute daily financial health scores and inject into the cache.

    This function MUST be called before temporal models (Step 6) so that
    forecasting, forward pass, and burn-out automatically learn from the
    health scores as additional daily features.

    Parameters
    ----------
    cache : pd.DataFrame
        The daily cache DataFrame (DatetimeIndex) with derived features
        from Steps 4-5.
    hierarchy_weights : dict, optional
        Tier weights for the composite score.  Keys: ``tier1`` .. ``tier5``.
        If None, uses equal weights (20% each).  When the company is in
        survival mode, these weights shift toward liquidity/solvency,
        making the composite reflect survival priorities.

    Returns
    -------
    (cache, result)
        The cache with new ``fh_*`` columns appended, and a
        ``FinancialHealthResult`` summary.
    """
    logger.info("Computing financial health scores...")

    weights = dict(_DEFAULT_WEIGHTS)
    if hierarchy_weights:
        for k, v in hierarchy_weights.items():
            if k in weights:
                weights[k] = float(v)
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    result = FinancialHealthResult()

    # Compute tier scores
    tier_scores: dict[str, pd.Series] = {
        "tier1": _score_liquidity(cache),
        "tier2": _score_solvency(cache),
        "tier3": _score_stability(cache),
        "tier4": _score_profitability(cache),
        "tier5": _score_growth(cache),
    }

    col_names = {
        "tier1": "fh_liquidity_score",
        "tier2": "fh_solvency_score",
        "tier3": "fh_stability_score",
        "tier4": "fh_profitability_score",
        "tier5": "fh_growth_score",
    }

    # Inject tier scores into cache
    for tier_key, score_series in tier_scores.items():
        col = col_names[tier_key]
        cache[col] = score_series
        result.columns_added.append(col)
        mean_val = float(score_series.mean()) if not score_series.isna().all() else float("nan")
        result.tier_means[col] = mean_val

    # Compute weighted composite
    composite_parts: list[pd.Series] = []
    for tier_key in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        w = weights[tier_key]
        s = tier_scores[tier_key]
        if not s.isna().all() and w > 0:
            composite_parts.append(s * w)

    if composite_parts:
        # Sum weighted components; re-normalize by actual weight coverage
        composite_df = pd.concat(composite_parts, axis=1)
        # For each row, compute weighted sum / sum of weights of non-NaN tiers
        weight_vals = []
        for tier_key in ("tier1", "tier2", "tier3", "tier4", "tier5"):
            s = tier_scores[tier_key]
            if not s.isna().all() and weights[tier_key] > 0:
                weight_vals.append(weights[tier_key])

        raw_sum = composite_df.sum(axis=1)
        # Track which tiers have data per row for proper normalization
        coverage_mask = pd.concat(
            [tier_scores[tk].notna().astype(float) * weights[tk]
             for tk in ("tier1", "tier2", "tier3", "tier4", "tier5")
             if not tier_scores[tk].isna().all() and weights[tk] > 0],
            axis=1,
        )
        weight_coverage = coverage_mask.sum(axis=1).replace(0, np.nan)
        composite = raw_sum / weight_coverage
    else:
        composite = pd.Series(np.nan, index=cache.index)

    composite = composite.clip(0, 100)
    cache["fh_composite_score"] = composite
    result.columns_added.append("fh_composite_score")

    # Label
    cache["fh_composite_label"] = composite.apply(_composite_label)
    result.columns_added.append("fh_composite_label")

    # Also add rate-of-change for temporal models to learn trends
    if not composite.isna().all():
        cache["fh_composite_delta_5d"] = composite.diff(5)
        cache["fh_composite_delta_21d"] = composite.diff(21)
        result.columns_added.extend(["fh_composite_delta_5d", "fh_composite_delta_21d"])

    # Summary stats
    result.n_days_scored = int(composite.notna().sum())
    result.mean_composite = float(composite.mean()) if result.n_days_scored > 0 else float("nan")
    if result.n_days_scored > 0:
        result.latest_composite = float(composite.iloc[-1]) if not np.isnan(composite.iloc[-1]) else float("nan")
        result.latest_label = _composite_label(result.latest_composite)

    logger.info(
        "Financial health: %d days scored, composite=%.1f (%s), cols=%d",
        result.n_days_scored,
        result.latest_composite,
        result.latest_label,
        len(result.columns_added),
    )

    return cache, result
