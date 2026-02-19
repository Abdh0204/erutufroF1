"""Tests for operator1.models.financial_health module.

Validates that financial health scores are correctly computed, injected
into the cache as daily columns, and that downstream temporal models
would see them as features.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from operator1.models.financial_health import (
    FinancialHealthResult,
    compute_financial_health,
    _normalize_series,
    _composite_label,
    _score_liquidity,
    _score_solvency,
    _score_stability,
    _score_profitability,
    _score_growth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cache(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Build a minimal daily cache with tier-relevant columns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame(index=dates)

    # Tier 1: Liquidity
    df["cash_ratio"] = rng.uniform(0.5, 3.0, n)
    df["free_cash_flow_ttm"] = rng.uniform(1e6, 1e9, n)
    df["cash_and_equivalents_asof"] = rng.uniform(1e7, 1e10, n)

    # Tier 2: Solvency
    df["debt_to_equity"] = rng.uniform(0.1, 5.0, n)
    df["net_debt_to_ebitda"] = rng.uniform(0.5, 8.0, n)
    df["interest_coverage"] = rng.uniform(1.0, 20.0, n)
    df["current_ratio"] = rng.uniform(0.5, 4.0, n)

    # Tier 3: Stability
    df["volatility_21d"] = rng.uniform(0.01, 0.5, n)
    df["drawdown_252d"] = rng.uniform(-0.6, 0.0, n)
    df["volume"] = rng.uniform(1e5, 1e8, n)

    # Tier 4: Profitability
    df["gross_margin"] = rng.uniform(0.1, 0.8, n)
    df["operating_margin"] = rng.uniform(0.05, 0.4, n)
    df["net_margin"] = rng.uniform(0.01, 0.3, n)

    # Tier 5: Growth
    df["revenue_asof"] = np.linspace(1e8, 1.2e8, n) + rng.normal(0, 1e6, n)
    df["pe_ratio"] = rng.uniform(5, 40, n)
    df["ev_to_ebitda"] = rng.uniform(3, 25, n)

    # Price column for returns (needed by derived vars but not by fh directly)
    df["close"] = rng.uniform(100, 200, n).cumsum() / n + 100

    return df


@pytest.fixture
def cache() -> pd.DataFrame:
    return _make_cache()


@pytest.fixture
def sparse_cache() -> pd.DataFrame:
    """Cache with only tier 1 and tier 3 columns."""
    n = 100
    rng = np.random.RandomState(99)
    dates = pd.bdate_range("2023-06-01", periods=n, freq="B")
    df = pd.DataFrame(index=dates)
    df["cash_ratio"] = rng.uniform(0.5, 3.0, n)
    df["volatility_21d"] = rng.uniform(0.01, 0.5, n)
    df["volume"] = rng.uniform(1e5, 1e8, n)
    return df


# ---------------------------------------------------------------------------
# Tests: _normalize_series
# ---------------------------------------------------------------------------


class TestNormalizeSeries:
    def test_output_range(self):
        s = pd.Series([10, 20, 30, 40, 50])
        result = _normalize_series(s)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_invert(self):
        # Use a series with variance so expanding rank produces different scores
        rng = np.random.RandomState(123)
        s = pd.Series(rng.uniform(0, 100, 200))
        normal = _normalize_series(s)
        inverted = _normalize_series(s, invert=True)
        # Inverted should be 100 - normal (within floating point tolerance)
        diff = (inverted - (100.0 - normal)).abs()
        assert diff.max() < 1e-10

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = _normalize_series(s)
        assert result.isna().all()

    def test_clipping(self):
        s = pd.Series([-5, 0, 5, 10, 100])
        result = _normalize_series(s, lower=0, upper=10)
        # After clipping, 100 becomes 10, -5 becomes 0
        assert result.notna().all()


# ---------------------------------------------------------------------------
# Tests: _composite_label
# ---------------------------------------------------------------------------


class TestCompositeLabel:
    def test_critical(self):
        assert _composite_label(10.0) == "Critical"

    def test_weak(self):
        assert _composite_label(30.0) == "Weak"

    def test_fair(self):
        assert _composite_label(50.0) == "Fair"

    def test_strong(self):
        assert _composite_label(70.0) == "Strong"

    def test_excellent(self):
        assert _composite_label(90.0) == "Excellent"

    def test_nan(self):
        assert _composite_label(float("nan")) == "Unknown"


# ---------------------------------------------------------------------------
# Tests: tier scoring functions
# ---------------------------------------------------------------------------


class TestTierScoring:
    def test_liquidity_score_range(self, cache):
        score = _score_liquidity(cache)
        valid = score.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_solvency_score_range(self, cache):
        score = _score_solvency(cache)
        valid = score.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_stability_score_range(self, cache):
        score = _score_stability(cache)
        valid = score.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_profitability_score_range(self, cache):
        score = _score_profitability(cache)
        valid = score.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_growth_score_range(self, cache):
        score = _score_growth(cache)
        # Some NaN at start due to pct_change window -- that's expected
        valid = score.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_empty_cache_returns_nan(self):
        empty = pd.DataFrame(index=pd.bdate_range("2023-01-02", periods=10))
        for scorer in (_score_liquidity, _score_solvency, _score_stability,
                       _score_profitability, _score_growth):
            result = scorer(empty)
            assert result.isna().all()


# ---------------------------------------------------------------------------
# Tests: compute_financial_health (main entry point)
# ---------------------------------------------------------------------------


class TestComputeFinancialHealth:
    def test_columns_injected(self, cache):
        """All fh_* columns should be present after computing."""
        result_cache, result = compute_financial_health(cache)

        expected_cols = [
            "fh_liquidity_score",
            "fh_solvency_score",
            "fh_stability_score",
            "fh_profitability_score",
            "fh_growth_score",
            "fh_composite_score",
            "fh_composite_label",
            "fh_composite_delta_5d",
            "fh_composite_delta_21d",
        ]
        for col in expected_cols:
            assert col in result_cache.columns, f"Missing column: {col}"

    def test_result_summary(self, cache):
        """Result object should have valid summary stats."""
        _, result = compute_financial_health(cache)

        assert isinstance(result, FinancialHealthResult)
        assert result.n_days_scored > 0
        assert not math.isnan(result.latest_composite)
        assert 0 <= result.latest_composite <= 100
        assert result.latest_label in ("Critical", "Weak", "Fair", "Strong", "Excellent")
        assert len(result.columns_added) >= 7

    def test_composite_range(self, cache):
        """Composite score should be within 0-100."""
        result_cache, _ = compute_financial_health(cache)
        valid = result_cache["fh_composite_score"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_equal_weights_default(self, cache):
        """With default weights all tiers contribute equally."""
        _, result = compute_financial_health(cache)
        # All tier means should be populated
        assert len(result.tier_means) == 5

    def test_survival_weights_shift_composite(self, cache):
        """Survival weights should change the composite vs equal weights."""
        _, result_equal = compute_financial_health(cache.copy())
        _, result_survival = compute_financial_health(
            cache.copy(),
            hierarchy_weights={"tier1": 0.50, "tier2": 0.30, "tier3": 0.15, "tier4": 0.04, "tier5": 0.01},
        )
        # The composites should differ since weights differ
        assert result_equal.mean_composite != result_survival.mean_composite

    def test_sparse_cache_handles_missing_tiers(self, sparse_cache):
        """With only tier 1 and tier 3 data, other tiers should be NaN."""
        result_cache, result = compute_financial_health(sparse_cache)

        # Liquidity and stability should have data
        assert result_cache["fh_liquidity_score"].notna().any()
        assert result_cache["fh_stability_score"].notna().any()

        # Solvency, profitability, growth should be all NaN
        assert result_cache["fh_solvency_score"].isna().all()
        assert result_cache["fh_profitability_score"].isna().all()
        assert result_cache["fh_growth_score"].isna().all()

        # Composite should still be computed from available tiers
        assert result_cache["fh_composite_score"].notna().any()
        assert result.n_days_scored > 0

    def test_cache_not_mutated_columns(self, cache):
        """Original cache columns should still be present."""
        original_cols = set(cache.columns)
        result_cache, _ = compute_financial_health(cache)
        # All original columns should still be there
        assert original_cols.issubset(set(result_cache.columns))

    def test_temporal_model_sees_fh_columns(self, cache):
        """Simulate what run_forecasting does: filter to available columns.

        After computing financial health, the fh_composite_score column
        should be visible to the forecasting variable filter.
        """
        result_cache, _ = compute_financial_health(cache)

        # The forecasting pipeline filters: [v for v in variables if v in cache.columns]
        test_vars = ["fh_composite_score", "fh_liquidity_score", "nonexistent_var"]
        available = [v for v in test_vars if v in result_cache.columns]
        assert "fh_composite_score" in available
        assert "fh_liquidity_score" in available
        assert "nonexistent_var" not in available

    def test_delta_columns_present(self, cache):
        """Rate-of-change columns should be present for trend learning."""
        result_cache, _ = compute_financial_health(cache)
        assert "fh_composite_delta_5d" in result_cache.columns
        assert "fh_composite_delta_21d" in result_cache.columns
        # First few values will be NaN due to diff, but later values should have data
        assert result_cache["fh_composite_delta_5d"].notna().sum() > 0
        assert result_cache["fh_composite_delta_21d"].notna().sum() > 0


# ---------------------------------------------------------------------------
# Tests: profile builder integration
# ---------------------------------------------------------------------------


class TestProfileBuilderIntegration:
    def test_financial_health_section_in_profile(self, cache):
        """The profile builder should include a financial_health section."""
        from operator1.report.profile_builder import _build_financial_health_section

        result_cache, result = compute_financial_health(cache)
        from dataclasses import asdict
        fh_dict = asdict(result)

        section = _build_financial_health_section(result_cache, fh_dict)

        assert section["available"] is True
        assert "latest_composite" in section
        assert "latest_label" in section
        assert "tier_means" in section
        assert "series_summary" in section
        assert "fh_composite_score" in section["series_summary"]

    def test_financial_health_section_without_data(self):
        """Section should be unavailable when no data."""
        from operator1.report.profile_builder import _build_financial_health_section

        section = _build_financial_health_section(None, None)
        assert section["available"] is False
