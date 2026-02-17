"""Fuzzy Logic government protection assessment.

Replaces the binary ``country_protected_flag`` (0 or 1) with a
continuous fuzzy membership score in [0, 1] that captures the
*degree* of government protection.

**Fuzzy variables and membership functions:**

1. **Sector strategicness** (0-1):
   - Core strategic (defense, energy, banking): 0.9-1.0
   - Semi-strategic (utilities, telecom, transport): 0.5-0.7
   - Non-strategic: 0.0-0.2

2. **Economic significance** (market_cap / GDP ratio):
   - Systemically important (>0.5%): 0.9-1.0
   - Significant (0.1-0.5%): 0.4-0.8
   - Minor (<0.1%): 0.0-0.3

3. **Policy responsiveness** (rate cut magnitude in lookback):
   - Emergency intervention (>2%): 0.8-1.0
   - Moderate adjustment (1-2%): 0.3-0.6
   - Normal policy (<1%): 0.0-0.2

The final protection score uses a fuzzy OR (maximum) of all three
dimensions, then applies a defuzzification step to produce a single
score.

Integration point: ``operator1/analysis/survival_mode.py``
Replaces: binary ``country_protected_flag``

Top-level entry points:
    ``compute_fuzzy_protection`` -- per-day fuzzy protection scores.
    ``FuzzyProtectionResult`` -- result container with per-dimension scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy membership functions
# ---------------------------------------------------------------------------


def _sigmoid(x: float, center: float, steepness: float = 10.0) -> float:
    """Smooth sigmoid membership function in [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))


def _trapezoidal(
    x: float,
    a: float, b: float, c: float, d: float,
) -> float:
    """Trapezoidal membership function.

    Returns 0 for x <= a or x >= d, rises linearly from a to b,
    flat 1.0 from b to c, falls linearly from c to d.
    """
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


# ---------------------------------------------------------------------------
# Sector strategicness
# ---------------------------------------------------------------------------

# Sector -> fuzzy membership score.  Core strategic sectors get 0.9+,
# semi-strategic get 0.5-0.7, everything else gets 0.1.
_SECTOR_SCORES: dict[str, float] = {
    # Core strategic
    "defense": 1.0,
    "energy": 0.95,
    "banking": 0.90,
    "oil & gas": 0.90,
    "aerospace & defense": 1.0,
    # Semi-strategic
    "utilities": 0.70,
    "telecom": 0.65,
    "telecommunications": 0.65,
    "transportation": 0.55,
    "healthcare": 0.50,
    "pharmaceuticals": 0.50,
    "insurance": 0.45,
    "financial services": 0.60,
    # Lower priority
    "technology": 0.30,
    "consumer electronics": 0.20,
    "retail": 0.15,
    "consumer discretionary": 0.15,
    "media": 0.10,
    "entertainment": 0.10,
}


def _sector_membership(sector: str | None) -> float:
    """Return fuzzy membership for sector strategicness."""
    if not sector:
        return 0.1  # unknown -> low protection
    s = sector.lower().strip()
    # Exact match
    if s in _SECTOR_SCORES:
        return _SECTOR_SCORES[s]
    # Partial match
    for key, score in _SECTOR_SCORES.items():
        if key in s or s in key:
            return score
    return 0.1  # default: non-strategic


# ---------------------------------------------------------------------------
# Economic significance
# ---------------------------------------------------------------------------


def _economic_significance(
    market_cap: float | None,
    gdp: float | None,
) -> float:
    """Fuzzy membership for market_cap / GDP ratio."""
    if market_cap is None or gdp is None or gdp <= 0:
        return 0.0  # unknown -> no protection signal

    ratio = market_cap / gdp

    # Sigmoid centered at 0.001 (0.1% of GDP) with a smooth ramp
    # to 1.0 for very large companies (>0.5% of GDP)
    if ratio >= 0.005:
        return 1.0
    if ratio >= 0.001:
        # Linear ramp from 0.4 at 0.1% to 1.0 at 0.5%
        return 0.4 + 0.6 * (ratio - 0.001) / (0.005 - 0.001)
    if ratio >= 0.0001:
        # Linear ramp from 0.0 at 0.01% to 0.4 at 0.1%
        return 0.4 * (ratio - 0.0001) / (0.001 - 0.0001)
    return 0.0


# ---------------------------------------------------------------------------
# Policy responsiveness
# ---------------------------------------------------------------------------


def _policy_responsiveness(
    rate_cut_pct: float | None,
) -> float:
    """Fuzzy membership for central bank emergency intervention."""
    if rate_cut_pct is None:
        return 0.0  # no data -> no signal

    cut = abs(rate_cut_pct)
    if cut >= 2.0:
        return min(1.0, 0.8 + 0.1 * (cut - 2.0))
    if cut >= 1.0:
        return 0.3 + 0.5 * (cut - 1.0)
    if cut >= 0.25:
        return 0.1 * (cut - 0.25) / 0.75
    return 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FuzzyProtectionResult:
    """Per-day fuzzy government protection assessment."""

    # Per-dimension scores (0-1)
    sector_score: float = 0.0
    economic_score: float = 0.0
    policy_score: float = 0.0

    # Aggregated protection degree (fuzzy OR = max)
    protection_degree: float = 0.0

    # Human-readable label
    label: str = "unprotected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_score": round(self.sector_score, 4),
            "economic_score": round(self.economic_score, 4),
            "policy_score": round(self.policy_score, 4),
            "protection_degree": round(self.protection_degree, 4),
            "label": self.label,
        }


def _label_from_degree(degree: float) -> str:
    """Map protection degree to a human-readable label."""
    if degree >= 0.8:
        return "strongly_protected"
    if degree >= 0.5:
        return "moderately_protected"
    if degree >= 0.25:
        return "weakly_protected"
    return "unprotected"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fuzzy_protection(
    cache: pd.DataFrame,
    sector: str | None = None,
    gdp: float | None = None,
) -> pd.DataFrame:
    """Compute daily fuzzy government protection scores.

    Parameters
    ----------
    cache:
        Daily feature table with at least ``market_cap`` column.
    sector:
        Target company sector (from verified profile).
    gdp:
        Country GDP in USD (from World Bank data).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``fuzzy_sector_score``
        - ``fuzzy_economic_score``
        - ``fuzzy_policy_score``
        - ``fuzzy_protection_degree`` (0-1, replaces binary flag)
        - ``fuzzy_protection_label``
    """
    result = cache.copy()
    n = len(result)

    # Sector strategicness (constant for all days)
    sector_score = _sector_membership(sector)
    result["fuzzy_sector_score"] = sector_score

    # Economic significance (varies daily with market_cap)
    if "market_cap" in result.columns and gdp is not None:
        result["fuzzy_economic_score"] = result["market_cap"].apply(
            lambda mc: _economic_significance(mc, gdp) if pd.notna(mc) else 0.0
        )
    else:
        result["fuzzy_economic_score"] = 0.0

    # Policy responsiveness (from lending/policy rate changes)
    # Check if we have rate data in the cache
    rate_col = None
    for col_name in ("lending_interest_rate", "policy_rate", "real_interest_rate"):
        if col_name in result.columns:
            rate_col = col_name
            break

    if rate_col is not None:
        # Compute 3-month (~63 business days) rate change
        rate_change = result[rate_col] - result[rate_col].shift(63)
        # Negative change = rate cut
        result["fuzzy_policy_score"] = rate_change.apply(
            lambda rc: _policy_responsiveness(-rc) if pd.notna(rc) else 0.0
        )
    else:
        result["fuzzy_policy_score"] = 0.0

    # Aggregation: fuzzy OR (element-wise maximum across dimensions)
    result["fuzzy_protection_degree"] = result[
        ["fuzzy_sector_score", "fuzzy_economic_score", "fuzzy_policy_score"]
    ].max(axis=1)

    # Label
    result["fuzzy_protection_label"] = result["fuzzy_protection_degree"].apply(
        _label_from_degree
    )

    # Also update the binary flag for backward compatibility
    result["country_protected_flag"] = (
        result["fuzzy_protection_degree"] >= 0.5
    ).astype(int)

    logger.info(
        "Fuzzy protection: sector=%.2f, mean_economic=%.2f, mean_policy=%.2f, "
        "mean_degree=%.2f",
        sector_score,
        result["fuzzy_economic_score"].mean(),
        result["fuzzy_policy_score"].mean(),
        result["fuzzy_protection_degree"].mean(),
    )

    return result
