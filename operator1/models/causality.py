"""T6.1 (cont.) -- Causality analysis and variable pruning.

Provides Granger-causality testing and relationship pruning that
feed into the forecasting models (T6.2).

**Granger Causality:**
  Tests whether the past values of variable X improve the prediction
  of variable Y beyond what Y's own past provides.  The result is a
  pairwise causality matrix indicating statistically significant
  (p < 0.05) causal links.

**Variable Pruning:**
  Removes weakly connected variables from the modelling set to reduce
  overfitting and speed up downstream VAR/LSTM fitting.  The pruning
  threshold is configurable.

**Tier-Aware Pruning (Sec 10.4):**
  Variables belonging to Tier 1-2 (liquidity, solvency) are never
  pruned regardless of their Granger score, because survival mode
  logic depends on them unconditionally.

Output:
  - ``causality_matrix``: DataFrame of shape (n_vars, n_vars) where
    entry (y, x) = 1 if x Granger-causes y at p < 0.05.
  - ``pruned_variables``: list of variable names retained after pruning.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

# Default Granger test settings.
DEFAULT_MAX_LAG: int = 5
DEFAULT_SIGNIFICANCE: float = 0.05

# Default pruning threshold: minimum incoming causal links to keep a var.
DEFAULT_PRUNE_THRESHOLD: float = 1.0

# Maximum variables for VAR to remain numerically stable.
DEFAULT_MAX_VARS_FOR_VAR: int = 20


# ---------------------------------------------------------------------------
# Granger causality matrix
# ---------------------------------------------------------------------------


def compute_granger_causality(
    cache: pd.DataFrame,
    variables: list[str],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> pd.DataFrame:
    """Compute pairwise Granger-causality matrix.

    Parameters
    ----------
    cache:
        Daily cache DataFrame containing all ``variables`` as columns.
    variables:
        List of variable names to test.
    max_lag:
        Maximum lag order for the Granger test.
    significance:
        p-value threshold for declaring significance.

    Returns
    -------
    causality_matrix:
        DataFrame of shape ``(len(variables), len(variables))`` where
        entry ``[y, x]`` is 1 if ``x`` Granger-causes ``y`` at the
        given significance level, else 0.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "statsmodels not installed -- returning empty causality matrix"
        )
        return pd.DataFrame(
            np.zeros((len(variables), len(variables))),
            index=variables,
            columns=variables,
        )

    # Filter to available columns.
    available = [v for v in variables if v in cache.columns]
    missing = set(variables) - set(available)
    if missing:
        logger.info(
            "Granger: %d variables not in cache, skipping: %s",
            len(missing),
            sorted(missing)[:5],
        )

    if len(available) < 2:
        logger.warning("Need at least 2 variables for Granger test")
        return pd.DataFrame(
            np.zeros((len(available), len(available))),
            index=available,
            columns=available,
        )

    # Prepare clean data.
    data = cache[available].dropna()

    if len(data) < max_lag + 5:
        logger.warning(
            "Insufficient observations (%d) for Granger test (need > %d)",
            len(data),
            max_lag + 5,
        )
        return pd.DataFrame(
            np.zeros((len(available), len(available))),
            index=available,
            columns=available,
        )

    logger.info(
        "Computing Granger causality: %d variables, %d observations, max_lag=%d",
        len(available),
        len(data),
        max_lag,
    )

    matrix = pd.DataFrame(
        np.zeros((len(available), len(available))),
        index=available,
        columns=available,
    )

    n_tests = 0
    n_significant = 0

    for y_var in available:
        for x_var in available:
            if y_var == x_var:
                continue

            try:
                # grangercausalitytests expects [y, x] column order.
                test_result = grangercausalitytests(
                    data[[y_var, x_var]],
                    maxlag=max_lag,
                    verbose=False,
                )

                # Get minimum p-value across all tested lags.
                p_values = []
                for lag in range(1, max_lag + 1):
                    if lag in test_result:
                        p_val = test_result[lag][0]["ssr_ftest"][1]
                        p_values.append(p_val)

                if p_values:
                    min_p = min(p_values)
                    if min_p < significance:
                        matrix.loc[y_var, x_var] = 1
                        n_significant += 1

                n_tests += 1

            except Exception:
                # Singular matrix, constant series, etc. -- skip pair.
                pass

    logger.info(
        "Granger causality: %d pairs tested, %d significant (%.1f%%)",
        n_tests,
        n_significant,
        100 * n_significant / max(n_tests, 1),
    )

    return matrix


# ---------------------------------------------------------------------------
# Variable pruning
# ---------------------------------------------------------------------------


def _get_protected_variables() -> set[str]:
    """Return variable names that must never be pruned.

    Tier 1 and Tier 2 variables from the survival hierarchy config are
    always retained because survival mode depends on them.
    """
    try:
        cfg = load_config("survival_hierarchy")
    except FileNotFoundError:
        logger.warning("survival_hierarchy config not found -- no protected vars")
        return set()

    protected: set[str] = set()
    tiers = cfg.get("tiers", {})

    for tier_key in ("tier1", "tier2"):
        tier_data = tiers.get(tier_key, {})
        tier_vars = tier_data.get("variables", [])
        protected.update(tier_vars)

    return protected


def prune_weak_relationships(
    causality_matrix: pd.DataFrame,
    *,
    threshold: float = DEFAULT_PRUNE_THRESHOLD,
    max_vars: int = DEFAULT_MAX_VARS_FOR_VAR,
    protect_tiers_1_2: bool = True,
) -> list[str]:
    """Remove weakly-connected variables from the modelling set.

    Parameters
    ----------
    causality_matrix:
        Square DataFrame from ``compute_granger_causality``.
    threshold:
        Minimum number of incoming causal links to retain a variable.
    max_vars:
        Hard cap on variables retained (for VAR numerical stability).
    protect_tiers_1_2:
        If *True*, variables in Tier 1-2 of the survival hierarchy are
        never pruned.

    Returns
    -------
    List of variable names to keep, ordered by causal strength
    (descending).
    """
    if causality_matrix.empty:
        return []

    # Count incoming causal links per variable.
    incoming = causality_matrix.sum(axis=1)

    # Protected variables (never pruned).
    protected = _get_protected_variables() if protect_tiers_1_2 else set()

    # Select variables meeting the threshold.
    strong_vars = set(incoming[incoming >= threshold].index)

    # Always include protected variables.
    all_vars = strong_vars | (protected & set(causality_matrix.index))

    # Sort by causal strength descending.
    sorted_vars = (
        incoming.loc[list(all_vars)]
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Apply hard cap.
    if len(sorted_vars) > max_vars:
        # Keep protected variables and fill remaining slots by strength.
        protected_in_list = [v for v in sorted_vars if v in protected]
        non_protected = [v for v in sorted_vars if v not in protected]
        remaining_slots = max(0, max_vars - len(protected_in_list))
        sorted_vars = protected_in_list + non_protected[:remaining_slots]

    n_pruned = len(causality_matrix.index) - len(sorted_vars)
    logger.info(
        "Variable pruning: kept %d / %d variables (%d pruned, "
        "%d protected, threshold=%.1f, cap=%d)",
        len(sorted_vars),
        len(causality_matrix.index),
        n_pruned,
        len(protected & set(sorted_vars)),
        threshold,
        max_vars,
    )

    return sorted_vars


# ---------------------------------------------------------------------------
# Convenience: combined pipeline step
# ---------------------------------------------------------------------------


def run_causality_analysis(
    cache: pd.DataFrame,
    variables: list[str],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    significance: float = DEFAULT_SIGNIFICANCE,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    max_vars: int = DEFAULT_MAX_VARS_FOR_VAR,
) -> tuple[pd.DataFrame, list[str]]:
    """Run Granger causality and variable pruning in one call.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variables:
        Candidate variable names to analyse.
    max_lag:
        Maximum lag for Granger test.
    significance:
        p-value threshold.
    prune_threshold:
        Minimum incoming causal links to keep.
    max_vars:
        Hard cap on retained variables.

    Returns
    -------
    (causality_matrix, strong_variables)
    """
    logger.info("Running causality analysis pipeline...")

    matrix = compute_granger_causality(
        cache,
        variables,
        max_lag=max_lag,
        significance=significance,
    )

    strong_vars = prune_weak_relationships(
        matrix,
        threshold=prune_threshold,
        max_vars=max_vars,
    )

    return matrix, strong_vars
