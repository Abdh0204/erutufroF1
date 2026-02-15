"""T4.3 -- Survival hierarchy weight assignment.

For each day, selects a **regime** based on the survival flags computed
by T4.1 and looks up the corresponding tier weight vector from
``config/survival_hierarchy.yml``.

Regime selection logic:
  - Neither company nor country flag -> ``normal``
  - Company flag only                -> ``company_survival``
  - Country flag only (no protection)-> ``modified_survival``
  - Both flags                       -> ``extreme_survival``

When ``vanity_percentage`` exceeds the configured threshold, a small
weight delta shifts from tier 5 to tier 3.

Output columns:
  - ``survival_regime`` (categorical label)
  - ``hierarchy_tier1_weight`` ... ``hierarchy_tier5_weight``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

# Number of tiers in the hierarchy
_NUM_TIERS = 5


# ---------------------------------------------------------------------------
# Regime selection
# ---------------------------------------------------------------------------


def _select_regime(
    company_flag: int,
    country_flag: int,
    protected_flag: int,
) -> str:
    """Determine the survival regime for a single day.

    Parameters
    ----------
    company_flag:
        1 if company survival mode is active.
    country_flag:
        1 if country survival mode is active.
    protected_flag:
        1 if the company is country-protected.

    Returns
    -------
    str
        One of: ``"normal"``, ``"company_survival"``,
        ``"modified_survival"``, ``"extreme_survival"``.
    """
    company = bool(company_flag)
    country = bool(country_flag)
    protected = bool(protected_flag)

    if not company and not country:
        return "normal"
    if company and not country:
        return "company_survival"
    if not company and country:
        # Country crisis only; if protected, use modified_survival
        # (protection reduces the severity of weight shifts)
        return "modified_survival"
    # Both company and country in crisis
    return "extreme_survival"


def select_regime_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised regime selection for the full feature table.

    Parameters
    ----------
    df:
        Must contain ``company_survival_mode_flag``,
        ``country_survival_mode_flag``, ``country_protected_flag``.

    Returns
    -------
    pd.Series
        String series with regime labels.
    """
    company = df.get(
        "company_survival_mode_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)
    country = df.get(
        "country_survival_mode_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)
    protected = df.get(
        "country_protected_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)

    regimes = pd.Series("normal", index=df.index, dtype="object")

    # Company only
    regimes = regimes.where(
        ~((company == 1) & (country == 0)),
        other="company_survival",
    )
    # Country only
    regimes = regimes.where(
        ~((company == 0) & (country == 1)),
        other="modified_survival",
    )
    # Both
    regimes = regimes.where(
        ~((company == 1) & (country == 1)),
        other="extreme_survival",
    )

    regimes.name = "survival_regime"
    return regimes


# ---------------------------------------------------------------------------
# Weight lookup & vanity adjustment
# ---------------------------------------------------------------------------


def _get_weight_vector(
    regime: str,
    regimes_cfg: dict[str, Any],
) -> list[float]:
    """Look up tier weights for a given regime from config.

    Falls back to ``normal`` if the regime is not found.
    """
    preset = regimes_cfg.get(regime, regimes_cfg.get("normal", {}))
    weights = preset.get("weights", [0.2] * _NUM_TIERS)
    return weights[:_NUM_TIERS]


def _apply_vanity_adjustment(
    weights: list[float],
    vanity_pct: float,
    vanity_cfg: dict[str, Any],
) -> list[float]:
    """Adjust tier weights based on vanity percentage.

    When ``vanity_pct`` exceeds the threshold, shift weight from tier 5
    to tier 3 by the configured delta amounts.

    Parameters
    ----------
    weights:
        Current 5-element weight list.
    vanity_pct:
        The vanity percentage for this day.
    vanity_cfg:
        The ``vanity_adjustment`` section from config.

    Returns
    -------
    list[float]
        Adjusted and normalised weights.
    """
    threshold = vanity_cfg.get("threshold", 10.0)
    if np.isnan(vanity_pct) or vanity_pct <= threshold:
        return weights

    tier3_delta = vanity_cfg.get("tier3_delta", 0.05)
    tier5_delta = vanity_cfg.get("tier5_delta", -0.05)

    adjusted = list(weights)
    # tier3 is index 2, tier5 is index 4
    adjusted[2] = adjusted[2] + tier3_delta
    adjusted[4] = adjusted[4] + tier5_delta

    # Clamp to non-negative
    adjusted = [max(0.0, w) for w in adjusted]

    # Normalise to sum to 1.0
    total = sum(adjusted)
    if total > 0:
        adjusted = [w / total for w in adjusted]

    return adjusted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_hierarchy_weights(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute survival hierarchy tier weights for every day.

    Parameters
    ----------
    df:
        Daily feature table with survival flags and vanity_percentage
        already computed.
    config:
        Override survival hierarchy config (loads YAML if None).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``survival_regime``
        - ``hierarchy_tier1_weight`` ... ``hierarchy_tier5_weight``
    """
    if config is None:
        config = load_config("survival_hierarchy")

    regimes_cfg = config.get("regimes", {})
    vanity_cfg = config.get("vanity_adjustment", {})

    result = df.copy()

    # 1. Select regime for each day
    result["survival_regime"] = select_regime_series(result)

    # 2. Assign weights per day
    vanity_pct = result.get(
        "vanity_percentage", pd.Series(np.nan, index=result.index),
    )

    # Pre-compute weight vectors for each regime
    regime_weights_cache: dict[str, list[float]] = {}
    for regime_name in ("normal", "company_survival", "modified_survival", "extreme_survival"):
        regime_weights_cache[regime_name] = _get_weight_vector(
            regime_name, regimes_cfg,
        )

    # Build weight columns
    tier_columns: dict[str, list[float]] = {
        f"hierarchy_tier{i+1}_weight": [] for i in range(_NUM_TIERS)
    }

    for idx in range(len(result)):
        regime = result["survival_regime"].iloc[idx]
        base_weights = list(regime_weights_cache.get(regime, [0.2] * _NUM_TIERS))

        # Apply vanity adjustment
        vp = vanity_pct.iloc[idx]
        adjusted = _apply_vanity_adjustment(base_weights, vp, vanity_cfg)

        for tier_idx in range(_NUM_TIERS):
            col = f"hierarchy_tier{tier_idx+1}_weight"
            tier_columns[col].append(adjusted[tier_idx])

    for col, values in tier_columns.items():
        result[col] = values

    # Log regime distribution
    regime_counts = result["survival_regime"].value_counts()
    logger.info("Regime distribution:\n%s", regime_counts.to_string())

    # Verify weights sum to 1.0
    weight_cols = [f"hierarchy_tier{i+1}_weight" for i in range(_NUM_TIERS)]
    weight_sums = result[weight_cols].sum(axis=1)
    max_drift = (weight_sums - 1.0).abs().max()
    if max_drift > 1e-6:
        logger.warning(
            "Weight normalisation drift detected: max abs deviation = %.8f",
            max_drift,
        )

    logger.info(
        "Hierarchy weights computed: %d days, %d unique regimes",
        len(result), result["survival_regime"].nunique(),
    )

    return result
