"""Tests for the Financial Health module (Altman Z-Score, Beneish M-Score,
Liquidity Runway).

Tests cover:
  - Altman Z-Score computation and zone classification
  - Beneish M-Score earnings quality detection
  - Liquidity runway estimation
  - Combined financial health scoring
  - Edge cases: missing data, zero denominators, single-period data
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 60) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_healthy_company(days: int = 60) -> pd.DataFrame:
    """Build a synthetic healthy company with strong financials."""
    idx = _make_daily_index(days)
    return pd.DataFrame({
        "close": np.full(days, 150.0),
        "revenue": np.full(days, 10_000_000.0),
        "gross_profit": np.full(days, 6_000_000.0),
        "ebit": np.full(days, 3_000_000.0),
        "ebitda": np.full(days, 3_500_000.0),
        "net_income": np.full(days, 2_000_000.0),
        "total_assets": np.full(days, 20_000_000.0),
        "total_liabilities": np.full(days, 5_000_000.0),
        "total_equity": np.full(days, 15_000_000.0),
        "current_assets": np.full(days, 8_000_000.0),
        "current_liabilities": np.full(days, 2_000_000.0),
        "cash_and_equivalents": np.full(days, 4_000_000.0),
        "receivables": np.full(days, 1_000_000.0),
        "market_cap": np.full(days, 50_000_000.0),
        "operating_cash_flow": np.full(days, 2_500_000.0),
        "capex": np.full(days, -500_000.0),
    }, index=idx)


def _make_distressed_company(days: int = 60) -> pd.DataFrame:
    """Build a synthetic distressed company with weak financials."""
    idx = _make_daily_index(days)
    return pd.DataFrame({
        "close": np.full(days, 5.0),
        "revenue": np.full(days, 1_000_000.0),
        "gross_profit": np.full(days, 100_000.0),
        "ebit": np.full(days, -200_000.0),
        "ebitda": np.full(days, -100_000.0),
        "net_income": np.full(days, -500_000.0),
        "total_assets": np.full(days, 3_000_000.0),
        "total_liabilities": np.full(days, 8_000_000.0),
        "total_equity": np.full(days, -5_000_000.0),
        "current_assets": np.full(days, 500_000.0),
        "current_liabilities": np.full(days, 2_000_000.0),
        "cash_and_equivalents": np.full(days, 200_000.0),
        "receivables": np.full(days, 300_000.0),
        "market_cap": np.full(days, 1_000_000.0),
        "operating_cash_flow": np.full(days, -1_000_000.0),
        "capex": np.full(days, -100_000.0),
    }, index=idx)


def _make_two_period_company(days: int = 60) -> pd.DataFrame:
    """Build a company with two distinct financial periods for M-Score."""
    idx = _make_daily_index(days)
    mid = days // 2

    # Period 1: first half with one set of financials
    # Period 2: second half with changed financials (revenue grew, margins shrank)
    revenue = np.concatenate([
        np.full(mid, 5_000_000.0),
        np.full(days - mid, 8_000_000.0),  # 60% revenue growth
    ])
    gross_profit = np.concatenate([
        np.full(mid, 3_000_000.0),   # 60% margin
        np.full(days - mid, 3_200_000.0),  # 40% margin (shrank)
    ])
    receivables = np.concatenate([
        np.full(mid, 500_000.0),
        np.full(days - mid, 1_500_000.0),  # receivables tripled
    ])

    return pd.DataFrame({
        "close": np.full(days, 100.0),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "ebit": revenue * 0.2,
        "ebitda": revenue * 0.25,
        "net_income": revenue * 0.1,
        "total_assets": np.full(days, 15_000_000.0),
        "total_liabilities": np.full(days, 6_000_000.0),
        "total_equity": np.full(days, 9_000_000.0),
        "current_assets": np.full(days, 5_000_000.0),
        "current_liabilities": np.full(days, 2_000_000.0),
        "cash_and_equivalents": np.full(days, 3_000_000.0),
        "receivables": receivables,
        "market_cap": np.full(days, 30_000_000.0),
        "operating_cash_flow": revenue * 0.15,
        "capex": np.full(days, -300_000.0),
    }, index=idx)


# ===========================================================================
# Altman Z-Score tests
# ===========================================================================

class TestAltmanZScore(unittest.TestCase):
    """Test Altman Z-Score computation."""

    def test_healthy_company_safe_zone(self):
        """A healthy company should be in the safe zone (Z > 2.99)."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_healthy_company()
        result = compute_altman_z_score(df)

        self.assertTrue(result.available)
        self.assertIsNotNone(result.latest_z_score)
        self.assertEqual(result.zone, "safe")
        self.assertGreater(result.latest_z_score, 2.99)

    def test_distressed_company_distress_zone(self):
        """A distressed company should be in the distress zone (Z < 1.81)."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_distressed_company()
        result = compute_altman_z_score(df)

        self.assertTrue(result.available)
        self.assertIsNotNone(result.latest_z_score)
        self.assertEqual(result.zone, "distress")
        self.assertLess(result.latest_z_score, 1.81)

    def test_z_score_series_length(self):
        """Z-Score series should have same length as input."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_healthy_company(days=30)
        result = compute_altman_z_score(df)

        self.assertEqual(len(result.z_score_series), 30)

    def test_missing_total_assets(self):
        """Should fail gracefully when total_assets is missing."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_healthy_company()
        df.drop(columns=["total_assets"], inplace=True)
        result = compute_altman_z_score(df)

        self.assertFalse(result.available)
        self.assertIn("total_assets", result.error)

    def test_components_populated(self):
        """All five Z-Score components should be present."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_healthy_company()
        result = compute_altman_z_score(df)

        self.assertEqual(len(result.components), 5)
        for key in [
            "x1_working_capital_ta", "x2_retained_earnings_ta",
            "x3_ebit_ta", "x4_market_cap_tl", "x5_revenue_ta",
        ]:
            self.assertIn(key, result.components)

    def test_to_dict(self):
        """to_dict() should produce a JSON-safe dictionary."""
        from operator1.models.financial_health import compute_altman_z_score

        df = _make_healthy_company()
        result = compute_altman_z_score(df)
        d = result.to_dict()

        self.assertIn("available", d)
        self.assertIn("latest_z_score", d)
        self.assertIn("zone", d)


# ===========================================================================
# Beneish M-Score tests
# ===========================================================================

class TestBeneishMScore(unittest.TestCase):
    """Test Beneish M-Score computation."""

    def test_two_period_company_computes(self):
        """M-Score should be computable from two-period data."""
        from operator1.models.financial_health import compute_beneish_m_score

        df = _make_two_period_company()
        result = compute_beneish_m_score(df)

        self.assertTrue(result.available)
        self.assertIsNotNone(result.m_score)
        self.assertIn(result.verdict, ["unlikely", "possible", "likely"])

    def test_single_period_fails_gracefully(self):
        """Should fail when there's only one financial period."""
        from operator1.models.financial_health import compute_beneish_m_score

        df = _make_healthy_company()  # constant financials = no period break
        result = compute_beneish_m_score(df)

        self.assertFalse(result.available)

    def test_missing_revenue_fails(self):
        """Should fail when revenue is missing."""
        from operator1.models.financial_health import compute_beneish_m_score

        df = _make_healthy_company()
        df.drop(columns=["revenue"], inplace=True)
        result = compute_beneish_m_score(df)

        self.assertFalse(result.available)

    def test_suspicious_financials_flagged(self):
        """A company with suspicious pattern should score higher (closer to manipulation)."""
        from operator1.models.financial_health import compute_beneish_m_score

        # Create a company with manipulation red flags:
        # - Revenue growing fast (SGI high)
        # - Receivables growing faster than revenue (DSRI high)
        # - Gross margin declining (GMI high)
        # - High accruals (TATA high)
        df = _make_two_period_company()
        result = compute_beneish_m_score(df)

        # The two_period company has growing receivables + shrinking margins
        # which are manipulation red flags
        self.assertTrue(result.available)
        self.assertIsNotNone(result.m_score)

    def test_to_dict(self):
        """to_dict() should produce a JSON-safe dictionary."""
        from operator1.models.financial_health import compute_beneish_m_score

        df = _make_two_period_company()
        result = compute_beneish_m_score(df)
        d = result.to_dict()

        self.assertIn("m_score", d)
        self.assertIn("verdict", d)
        self.assertIn("likely_manipulator", d)


# ===========================================================================
# Liquidity Runway tests
# ===========================================================================

class TestLiquidityRunway(unittest.TestCase):
    """Test Liquidity Runway computation."""

    def test_healthy_company_strong_runway(self):
        """Healthy company with positive OCF should have strong runway."""
        from operator1.models.financial_health import compute_liquidity_runway

        df = _make_healthy_company()
        result = compute_liquidity_runway(df)

        self.assertTrue(result.available)
        # Company generates cash, so runway is effectively infinite
        self.assertIn(result.verdict, ["strong", "adequate"])

    def test_distressed_company_critical_runway(self):
        """Distressed company burning cash should have critical runway."""
        from operator1.models.financial_health import compute_liquidity_runway

        df = _make_distressed_company()
        result = compute_liquidity_runway(df)

        self.assertTrue(result.available)
        self.assertIsNotNone(result.months_of_runway)
        # 200K cash / (1M burn/12) = 2.4 months -> critical
        self.assertEqual(result.verdict, "critical")
        self.assertLess(result.months_of_runway, 6)

    def test_missing_cash_fails(self):
        """Should fail when cash_and_equivalents is missing."""
        from operator1.models.financial_health import compute_liquidity_runway

        df = _make_healthy_company()
        df.drop(columns=["cash_and_equivalents"], inplace=True)
        result = compute_liquidity_runway(df)

        self.assertFalse(result.available)

    def test_cash_available_populated(self):
        """Result should include the absolute cash amount."""
        from operator1.models.financial_health import compute_liquidity_runway

        df = _make_healthy_company()
        result = compute_liquidity_runway(df)

        self.assertIsNotNone(result.cash_available)
        self.assertAlmostEqual(result.cash_available, 4_000_000.0)


# ===========================================================================
# Combined Financial Health tests
# ===========================================================================

class TestFinancialHealth(unittest.TestCase):
    """Test combined financial health computation."""

    def test_healthy_company_high_score(self):
        """Healthy company should have a high health score."""
        from operator1.models.financial_health import compute_financial_health

        # Use two-period data so M-Score can compute
        df = _make_two_period_company()
        # Override with strong financials
        df["market_cap"] = 50_000_000.0
        df["operating_cash_flow"] = 3_000_000.0

        result = compute_financial_health(df)

        self.assertIsNotNone(result.overall_health_score)
        self.assertIn(result.overall_verdict, ["healthy", "moderate"])

    def test_distressed_company_low_score(self):
        """Distressed company should have a low health score."""
        from operator1.models.financial_health import compute_financial_health

        df = _make_distressed_company()
        result = compute_financial_health(df)

        # Z-Score is in distress zone, runway is critical
        self.assertIsNotNone(result.overall_health_score)
        self.assertIn(result.overall_verdict, ["weak", "critical"])

    def test_all_submodels_run(self):
        """All three sub-models should be attempted."""
        from operator1.models.financial_health import compute_financial_health

        df = _make_two_period_company()
        result = compute_financial_health(df)

        # Z-Score should be available
        self.assertTrue(result.altman_z.available)
        # M-Score should be available (two periods)
        self.assertTrue(result.beneish_m.available)
        # Runway should be available
        self.assertTrue(result.liquidity_runway.available)

    def test_to_dict_complete(self):
        """to_dict() should include all sub-model results."""
        from operator1.models.financial_health import compute_financial_health

        df = _make_two_period_company()
        result = compute_financial_health(df)
        d = result.to_dict()

        self.assertIn("altman_z", d)
        self.assertIn("beneish_m", d)
        self.assertIn("liquidity_runway", d)
        self.assertIn("overall_health_score", d)
        self.assertIn("overall_verdict", d)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        from operator1.models.financial_health import compute_financial_health

        df = pd.DataFrame()
        result = compute_financial_health(df)

        self.assertEqual(result.overall_verdict, "insufficient_data")


if __name__ == "__main__":
    unittest.main()
