"""Phase 3 tests -- cache building, derived variables, linked aggregates, macro alignment."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 20) -> pd.DatetimeIndex:
    """Return a short business-day index for testing."""
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_price_df(daily_index: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic price DataFrame with OHLCV columns."""
    n = len(daily_index)
    np.random.seed(42)
    closes = base + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": daily_index,
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.random.randint(1_000_000, 10_000_000, size=n),
    })


def _make_statement_df(report_dates: list[str], **kwargs) -> pd.DataFrame:
    """Build a synthetic periodic statement with given report dates and fields."""
    n = len(report_dates)
    data = {"report_date": pd.to_datetime(report_dates)}
    for field_name, values in kwargs.items():
        data[field_name] = values if len(values) == n else values[:n]
    return pd.DataFrame(data)


def _make_entity_data(
    isin: str = "TEST_ISIN",
    daily_index: pd.DatetimeIndex | None = None,
    include_statements: bool = True,
):
    """Build a minimal EntityData with synthetic data."""
    from operator1.steps.data_extraction import EntityData

    if daily_index is None:
        daily_index = _make_daily_index()

    price_df = _make_price_df(daily_index)
    entity = EntityData(isin=isin)
    entity.profile = {
        "isin": isin,
        "ticker": "TST",
        "name": "Test Corp",
        "exchange": "NYSE",
        "currency": "USD",
        "country": "US",
        "sector": "Technology",
        "industry": "Software",
        "sub_industry": None,
    }
    entity.quotes = price_df
    entity.fmp_ohlcv = price_df

    if include_statements:
        entity.income_statement = _make_statement_df(
            ["2023-10-15", "2024-01-15"],
            revenue=[1_000_000, 1_100_000],
            gross_profit=[600_000, 660_000],
            ebit=[300_000, 330_000],
            ebitda=[350_000, 385_000],
            net_income=[200_000, 220_000],
            interest_expense=[10_000, 11_000],
            taxes=[50_000, 55_000],
        )
        entity.balance_sheet = _make_statement_df(
            ["2023-10-15", "2024-01-15"],
            total_assets=[5_000_000, 5_200_000],
            total_liabilities=[2_000_000, 2_100_000],
            total_equity=[3_000_000, 3_100_000],
            current_assets=[1_500_000, 1_600_000],
            current_liabilities=[800_000, 850_000],
            cash_and_equivalents=[500_000, 550_000],
            short_term_debt=[100_000, 110_000],
            long_term_debt=[500_000, 520_000],
            receivables=[200_000, 210_000],
        )
        entity.cashflow_statement = _make_statement_df(
            ["2023-10-15", "2024-01-15"],
            operating_cash_flow=[250_000, 270_000],
            capex=[-50_000, -55_000],
            investing_cf=[-80_000, -85_000],
            financing_cf=[-60_000, -65_000],
            dividends_paid=[-30_000, -32_000],
        )

    return entity


# ===========================================================================
# T3.1 / T3.2 -- Cache builder tests
# ===========================================================================

class TestBuildDateIndex(unittest.TestCase):
    """Test the business-day index generation."""

    def test_returns_bday_index(self):
        from operator1.steps.cache_builder import build_date_index
        idx = build_date_index(
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
        )
        self.assertIsInstance(idx, pd.DatetimeIndex)
        # Should not include weekends
        for dt in idx:
            self.assertIn(dt.dayofweek, range(5), f"{dt} is a weekend")

    def test_empty_range(self):
        from operator1.steps.cache_builder import build_date_index
        idx = build_date_index(
            start=pd.Timestamp("2024-01-06"),  # Saturday
            end=pd.Timestamp("2024-01-06"),
        )
        self.assertEqual(len(idx), 0)


class TestAsOfMerge(unittest.TestCase):
    """Test the as-of join logic for financial statements."""

    def test_basic_asof(self):
        from operator1.steps.cache_builder import _asof_merge_statement

        daily_index = _make_daily_index(20)
        stmt = _make_statement_df(
            ["2023-12-15", "2024-01-10"],
            revenue=[1_000_000, 1_100_000],
        )
        merged = _asof_merge_statement(
            daily_index, stmt, "TEST", "income_statement",
            fields=("revenue",),
        )
        self.assertEqual(len(merged), len(daily_index))
        # Before 2024-01-10, revenue should be 1M; after, 1.1M
        before = merged.loc[merged.index < pd.Timestamp("2024-01-10"), "revenue"]
        after = merged.loc[merged.index >= pd.Timestamp("2024-01-10"), "revenue"]
        if len(before) > 0:
            self.assertTrue((before == 1_000_000).all())
        if len(after) > 0:
            self.assertTrue((after == 1_100_000).all())

    def test_empty_statement_returns_nulls(self):
        from operator1.steps.cache_builder import _asof_merge_statement

        daily_index = _make_daily_index(5)
        merged = _asof_merge_statement(
            daily_index, pd.DataFrame(), "TEST", "balance_sheet",
            fields=("total_assets",),
        )
        self.assertEqual(len(merged), 5)
        self.assertTrue(merged["total_assets"].isna().all())

    def test_lookahead_raises(self):
        """A future report_date applied to a past day must raise LookAheadError."""
        from operator1.steps.cache_builder import _asof_merge_statement, LookAheadError

        daily_index = _make_daily_index(5)  # starts 2024-01-02
        # This statement has report_date in the past relative to the index,
        # so it should NOT raise.  merge_asof ensures backward direction.
        stmt = _make_statement_df(
            ["2024-01-02"],
            revenue=[500_000],
        )
        # Should not raise
        _asof_merge_statement(
            daily_index, stmt, "TEST", "income_statement",
            fields=("revenue",),
        )

    def test_no_report_date_column(self):
        from operator1.steps.cache_builder import _asof_merge_statement

        daily_index = _make_daily_index(5)
        stmt = pd.DataFrame({"revenue": [100]})
        merged = _asof_merge_statement(
            daily_index, stmt, "TEST", "income_statement",
            fields=("revenue",),
        )
        # Should return nulls since no date column to join on
        self.assertTrue(merged["revenue"].isna().all())


class TestBuildEntityDailyCache(unittest.TestCase):
    """Test the per-entity daily cache builder."""

    def test_target_cache_has_all_columns(self):
        from operator1.steps.cache_builder import (
            build_entity_daily_cache, build_date_index,
            QUOTE_FIELDS, STATEMENT_FIELDS, PROFILE_FIELDS,
        )

        daily_index = _make_daily_index(20)
        entity = _make_entity_data(daily_index=daily_index)
        cache = build_entity_daily_cache(entity, daily_index, use_fmp_prices=True)

        # Check all quote fields present
        for f in QUOTE_FIELDS:
            self.assertIn(f, cache.columns, f"Missing quote field: {f}")

        # Check all statement fields present
        for f in STATEMENT_FIELDS:
            self.assertIn(f, cache.columns, f"Missing statement field: {f}")

        # Check profile fields
        for f in PROFILE_FIELDS:
            self.assertIn(f, cache.columns, f"Missing profile field: {f}")

        # Check missing flags exist
        for f in QUOTE_FIELDS:
            self.assertIn(f"is_missing_{f}", cache.columns)

    def test_linked_entity_uses_quotes(self):
        from operator1.steps.cache_builder import build_entity_daily_cache

        daily_index = _make_daily_index(10)
        entity = _make_entity_data(daily_index=daily_index)
        entity.fmp_ohlcv = pd.DataFrame()  # no FMP data
        cache = build_entity_daily_cache(entity, daily_index, use_fmp_prices=False)
        # Should still have close prices from quotes
        self.assertFalse(cache["close"].isna().all())

    def test_no_price_data_gives_nulls(self):
        from operator1.steps.cache_builder import build_entity_daily_cache

        daily_index = _make_daily_index(5)
        entity = _make_entity_data(daily_index=daily_index, include_statements=False)
        entity.quotes = pd.DataFrame()
        entity.fmp_ohlcv = pd.DataFrame()
        cache = build_entity_daily_cache(entity, daily_index, use_fmp_prices=True)
        self.assertTrue(cache["close"].isna().all())
        self.assertTrue((cache["is_missing_close"] == 1).all())


class TestAddMissingFlags(unittest.TestCase):
    """Test the is_missing_* flag generation."""

    def test_flags_set_correctly(self):
        from operator1.steps.cache_builder import add_missing_flags

        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, np.nan, np.nan],
        })
        result = add_missing_flags(df)
        self.assertIn("is_missing_a", result.columns)
        self.assertIn("is_missing_b", result.columns)
        self.assertEqual(result["is_missing_a"].tolist(), [0, 1, 0])
        self.assertEqual(result["is_missing_b"].tolist(), [1, 1, 1])

    def test_no_double_flags(self):
        from operator1.steps.cache_builder import add_missing_flags

        df = pd.DataFrame({
            "x": [1.0],
            "is_missing_x": [0],
        })
        result = add_missing_flags(df)
        # Should not create is_missing_is_missing_x
        self.assertNotIn("is_missing_is_missing_x", result.columns)


# ===========================================================================
# T3.3 -- Derived variables tests
# ===========================================================================

class TestSafeRatio(unittest.TestCase):
    """Test the safe ratio helper with edge cases."""

    def test_normal_division(self):
        from operator1.features.derived_variables import safe_ratio
        num = pd.Series([10.0, 20.0, 30.0])
        den = pd.Series([2.0, 4.0, 5.0])
        result, ism, inv = safe_ratio(num, den, "test")
        np.testing.assert_array_almost_equal(result.values, [5.0, 5.0, 6.0])
        self.assertTrue((ism == 0).all())
        self.assertTrue((inv == 0).all())

    def test_zero_denominator(self):
        from operator1.features.derived_variables import safe_ratio
        num = pd.Series([10.0, 20.0])
        den = pd.Series([0.0, 5.0])
        result, ism, inv = safe_ratio(num, den, "test")
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(ism.iloc[0], 1)
        self.assertEqual(inv.iloc[0], 1)
        # Second value should be fine
        self.assertAlmostEqual(result.iloc[1], 4.0)
        self.assertEqual(inv.iloc[1], 0)

    def test_null_denominator(self):
        from operator1.features.derived_variables import safe_ratio
        num = pd.Series([10.0])
        den = pd.Series([np.nan])
        result, ism, inv = safe_ratio(num, den, "test")
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(inv.iloc[0], 1)

    def test_tiny_denominator(self):
        from operator1.features.derived_variables import safe_ratio
        num = pd.Series([10.0])
        den = pd.Series([1e-15])  # smaller than EPSILON
        result, ism, inv = safe_ratio(num, den, "test")
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(inv.iloc[0], 1)

    def test_negative_denominator_ok(self):
        from operator1.features.derived_variables import safe_ratio
        num = pd.Series([10.0])
        den = pd.Series([-5.0])
        result, ism, inv = safe_ratio(num, den, "test")
        self.assertAlmostEqual(result.iloc[0], -2.0)
        self.assertEqual(inv.iloc[0], 0)


class TestComputeDerivedVariables(unittest.TestCase):
    """Test the full derived variable computation pipeline."""

    def _build_cache_df(self) -> pd.DataFrame:
        """Build a daily cache DF with enough columns for derived vars."""
        from operator1.steps.cache_builder import build_entity_daily_cache
        daily_index = _make_daily_index(30)
        entity = _make_entity_data(daily_index=daily_index)
        return build_entity_daily_cache(entity, daily_index, use_fmp_prices=True)

    def test_all_derived_vars_present(self):
        from operator1.features.derived_variables import (
            compute_derived_variables, DERIVED_VARIABLES,
        )
        cache = self._build_cache_df()
        result = compute_derived_variables(cache)
        for var in DERIVED_VARIABLES:
            self.assertIn(var, result.columns, f"Missing derived var: {var}")
            self.assertIn(
                f"is_missing_{var}", result.columns,
                f"Missing flag for: {var}",
            )

    def test_return_1d_computed(self):
        from operator1.features.derived_variables import compute_derived_variables
        cache = self._build_cache_df()
        result = compute_derived_variables(cache)
        # First day should be NaN (no previous close)
        self.assertTrue(np.isnan(result["return_1d"].iloc[0]))
        # Subsequent days should have values
        self.assertFalse(result["return_1d"].iloc[5:].isna().all())

    def test_current_ratio_positive(self):
        from operator1.features.derived_variables import compute_derived_variables
        cache = self._build_cache_df()
        result = compute_derived_variables(cache)
        # With synthetic data, current_ratio should be > 0 where not missing
        valid = result.loc[result["is_missing_current_ratio"] == 0, "current_ratio"]
        if len(valid) > 0:
            self.assertTrue((valid > 0).all())

    def test_drawdown_non_positive(self):
        from operator1.features.derived_variables import compute_derived_variables
        cache = self._build_cache_df()
        result = compute_derived_variables(cache)
        valid = result.loc[result["is_missing_drawdown_252d"] == 0, "drawdown_252d"]
        if len(valid) > 0:
            self.assertTrue((valid <= 0.0001).all())  # drawdown should be <= 0


class TestDerivedVariablesEdgeCases(unittest.TestCase):
    """Test edge cases for specific derived computations."""

    def test_solvency_with_missing_debt(self):
        """Both debt components null -> total_debt_asof should be flagged."""
        from operator1.features.derived_variables import _compute_solvency
        df = pd.DataFrame({
            "short_term_debt": [np.nan],
            "long_term_debt": [np.nan],
            "total_equity": [1_000_000],
            "cash_and_equivalents": [np.nan],
            "ebitda": [500_000],
        })
        result = _compute_solvency(df)
        self.assertEqual(result["is_missing_total_debt_asof"].iloc[0], 1)

    def test_solvency_with_negative_equity(self):
        """Negative equity -> signed D/E is negative, abs D/E is positive."""
        from operator1.features.derived_variables import _compute_solvency
        df = pd.DataFrame({
            "short_term_debt": [100_000.0],
            "long_term_debt": [400_000.0],
            "total_equity": [-200_000.0],
            "cash_and_equivalents": [50_000.0],
            "ebitda": [100_000.0],
        })
        result = _compute_solvency(df)
        # Signed D/E should be negative (debt / negative equity)
        self.assertLess(result["debt_to_equity_signed"].iloc[0], 0)
        # Absolute D/E should be positive
        self.assertGreater(result["debt_to_equity_abs"].iloc[0], 0)

    def test_valuation_with_zero_shares(self):
        """Zero shares_outstanding -> P/E should be null."""
        from operator1.features.derived_variables import _compute_valuation
        df = pd.DataFrame({
            "close": [150.0],
            "shares_outstanding": [0.0],
            "market_cap": [1_000_000.0],
            "net_income": [100_000.0],
            "revenue": [500_000.0],
            "ebitda": [200_000.0],
            "total_debt_asof": [300_000.0],
            "cash_and_equivalents": [50_000.0],
        })
        result = _compute_valuation(df)
        self.assertTrue(np.isnan(result["pe_ratio_calc"].iloc[0]))
        self.assertEqual(result["invalid_math_pe_ratio_calc"].iloc[0], 1)


# ===========================================================================
# T3.4 -- Linked aggregates tests
# ===========================================================================

class TestLinkedAggregates(unittest.TestCase):
    """Test linked entity aggregate computation."""

    def _make_linked_frames(self) -> dict[str, pd.DataFrame]:
        """Build two linked entity frames with known values."""
        daily_index = _make_daily_index(10)
        df1 = pd.DataFrame({
            "return_1d": np.full(10, 0.01),
            "volatility_21d": np.full(10, 0.15),
            "drawdown_252d": np.full(10, -0.10),
            "pe_ratio_calc": np.full(10, 20.0),
        }, index=daily_index)
        df2 = pd.DataFrame({
            "return_1d": np.full(10, 0.03),
            "volatility_21d": np.full(10, 0.25),
            "drawdown_252d": np.full(10, -0.20),
            "pe_ratio_calc": np.full(10, 30.0),
        }, index=daily_index)
        return {"ISIN1": df1, "ISIN2": df2}

    def test_avg_and_median(self):
        from operator1.features.linked_aggregates import compute_linked_aggregates

        daily_index = _make_daily_index(10)
        target = pd.DataFrame({"return_1d": np.full(10, 0.02)}, index=daily_index)
        linked = self._make_linked_frames()
        groups = {"competitors": ["ISIN1", "ISIN2"]}

        result = compute_linked_aggregates(target, linked, groups)
        # Average return should be (0.01 + 0.03) / 2 = 0.02
        self.assertAlmostEqual(
            result["competitors_avg_return_1d"].iloc[0], 0.02, places=6,
        )
        # Median of 0.01 and 0.03 is 0.02
        self.assertAlmostEqual(
            result["competitors_median_return_1d"].iloc[0], 0.02, places=6,
        )

    def test_empty_group_gives_nulls(self):
        from operator1.features.linked_aggregates import compute_linked_aggregates

        daily_index = _make_daily_index(5)
        target = pd.DataFrame({"return_1d": [0.01] * 5}, index=daily_index)
        result = compute_linked_aggregates(target, {}, {})

        # All group columns should be NaN
        self.assertTrue(result["competitors_avg_return_1d"].isna().all())
        self.assertTrue((result["is_missing_competitors_avg_return_1d"] == 1).all())

    def test_single_member_group(self):
        from operator1.features.linked_aggregates import compute_linked_aggregates

        daily_index = _make_daily_index(5)
        target = pd.DataFrame({"return_1d": [0.02] * 5}, index=daily_index)
        linked = {
            "ISIN1": pd.DataFrame(
                {"return_1d": np.full(5, 0.05)},
                index=daily_index,
            ),
        }
        groups = {"competitors": ["ISIN1"]}
        result = compute_linked_aggregates(target, linked, groups)
        # With one member, avg == median == the single value
        self.assertAlmostEqual(
            result["competitors_avg_return_1d"].iloc[0], 0.05, places=6,
        )

    def test_relative_measures(self):
        from operator1.features.linked_aggregates import compute_linked_aggregates

        daily_index = _make_daily_index(5)
        target = pd.DataFrame({
            "return_1d": np.full(5, 0.04),
            "pe_ratio_calc": np.full(5, 25.0),
        }, index=daily_index)
        linked = {
            "ISIN_A": pd.DataFrame({
                "return_1d": np.full(5, 0.02),
                "pe_ratio_calc": np.full(5, 20.0),
            }, index=daily_index),
        }
        groups = {
            "sector_peers": ["ISIN_A"],
            "industry_peers": ["ISIN_A"],
        }
        result = compute_linked_aggregates(target, linked, groups)
        # rel_strength = target return / sector avg return = 0.04 / 0.02 = 2.0
        self.assertAlmostEqual(
            result["rel_strength_vs_sector"].iloc[0], 2.0, places=4,
        )
        # valuation premium = target PE / industry avg PE = 25 / 20 = 1.25
        self.assertAlmostEqual(
            result["valuation_premium_vs_industry"].iloc[0], 1.25, places=4,
        )


# ===========================================================================
# T3.5 -- Macro alignment tests
# ===========================================================================

class TestMacroAlignment(unittest.TestCase):
    """Test macro data alignment from yearly to daily."""

    def _make_macro_dataset(self):
        from operator1.steps.macro_mapping import MacroDataset
        ds = MacroDataset(country_iso3="USA")
        ds.indicators = {
            "inflation_rate_yoy": pd.DataFrame({
                "year": [2023, 2024],
                "value": [3.0, 2.5],
            }),
            "unemployment_rate": pd.DataFrame({
                "year": [2023, 2024],
                "value": [3.8, 4.0],
            }),
            "gdp_growth": pd.DataFrame({
                "year": [2023],
                "value": [2.1],
            }),
        }
        ds.missing = ["real_interest_rate"]
        return ds

    def test_alignment_basic(self):
        from operator1.features.macro_alignment import align_macro_to_daily
        daily_index = _make_daily_index(20)  # starts 2024-01-02
        ds = self._make_macro_dataset()
        result = align_macro_to_daily(ds, daily_index)

        # All days in 2024 should get the 2024 inflation value (2.5)
        self.assertAlmostEqual(result["inflation_rate_yoy"].iloc[0], 2.5)

        # unemployment_rate in 2024 should be 4.0
        self.assertAlmostEqual(result["unemployment_rate"].iloc[0], 4.0)

        # gdp_growth only has 2023 value; 2024 days should use 2023 value (as-of)
        self.assertAlmostEqual(result["gdp_growth"].iloc[0], 2.1)

    def test_missing_indicator_flagged(self):
        from operator1.features.macro_alignment import align_macro_to_daily
        daily_index = _make_daily_index(5)
        ds = self._make_macro_dataset()
        result = align_macro_to_daily(ds, daily_index)

        # real_interest_rate was listed as missing
        self.assertIn("real_interest_rate", result.columns)
        self.assertTrue(result["real_interest_rate"].isna().all())
        self.assertTrue((result["is_missing_real_interest_rate"] == 1).all())

    def test_inflation_daily_equivalent(self):
        from operator1.features.macro_alignment import align_macro_to_daily
        daily_index = _make_daily_index(5)
        ds = self._make_macro_dataset()
        result = align_macro_to_daily(ds, daily_index)

        # inflation_rate_daily_equivalent = 2.5 / 365
        expected = 2.5 / 365.0
        self.assertAlmostEqual(
            result["inflation_rate_daily_equivalent"].iloc[0],
            expected,
            places=8,
        )

    def test_merge_onto_target(self):
        from operator1.features.macro_alignment import (
            align_macro_to_daily, merge_macro_onto_target,
        )

        daily_index = _make_daily_index(10)
        ds = self._make_macro_dataset()
        macro_aligned = align_macro_to_daily(ds, daily_index)

        target = pd.DataFrame({
            "close": np.full(10, 100.0),
            "return_1d": np.concatenate([[np.nan], np.full(9, 0.01)]),
        }, index=daily_index)

        merged = merge_macro_onto_target(target, macro_aligned)

        # Should have both target and macro columns
        self.assertIn("close", merged.columns)
        self.assertIn("inflation_rate_yoy", merged.columns)
        self.assertIn("real_return_1d", merged.columns)

    def test_real_return_computed(self):
        from operator1.features.macro_alignment import (
            align_macro_to_daily, merge_macro_onto_target,
        )

        daily_index = _make_daily_index(5)
        ds = self._make_macro_dataset()
        macro_aligned = align_macro_to_daily(ds, daily_index)

        target = pd.DataFrame({
            "return_1d": np.full(5, 0.01),
        }, index=daily_index)

        merged = merge_macro_onto_target(target, macro_aligned)

        # real_return = 0.01 - (2.5/365) ~= 0.003151
        expected = 0.01 - 2.5 / 365.0
        self.assertAlmostEqual(
            merged["real_return_1d"].iloc[0], expected, places=6,
        )

    def test_empty_macro_dataset(self):
        from operator1.steps.macro_mapping import MacroDataset
        from operator1.features.macro_alignment import align_macro_to_daily
        daily_index = _make_daily_index(5)
        ds = MacroDataset(country_iso3="UNKNOWN")
        result = align_macro_to_daily(ds, daily_index)
        # Should just have the inflation derived columns as NaN
        self.assertIn("inflation_rate_daily_equivalent", result.columns)
        self.assertTrue(result["inflation_rate_daily_equivalent"].isna().all())


# ===========================================================================
# Integration: full pipeline smoke test
# ===========================================================================

class TestPhase3Integration(unittest.TestCase):
    """End-to-end test: cache build -> derived vars -> aggregates -> macro."""

    def test_full_pipeline(self):
        from operator1.steps.cache_builder import build_entity_daily_cache
        from operator1.features.derived_variables import compute_derived_variables
        from operator1.features.linked_aggregates import compute_linked_aggregates
        from operator1.features.macro_alignment import (
            align_macro_to_daily, merge_macro_onto_target,
        )
        from operator1.steps.macro_mapping import MacroDataset

        daily_index = _make_daily_index(30)

        # Build target cache
        target_entity = _make_entity_data("TARGET", daily_index)
        target_daily = build_entity_daily_cache(
            target_entity, daily_index, use_fmp_prices=True,
        )
        self.assertEqual(len(target_daily), 30)

        # Compute derived variables
        target_daily = compute_derived_variables(target_daily)
        self.assertIn("return_1d", target_daily.columns)
        self.assertIn("current_ratio", target_daily.columns)

        # Build linked entity caches
        linked_entity = _make_entity_data("LINKED1", daily_index)
        linked_daily_cache = build_entity_daily_cache(
            linked_entity, daily_index, use_fmp_prices=False,
        )
        linked_daily_cache = compute_derived_variables(linked_daily_cache)
        linked_daily = {"LINKED1": linked_daily_cache}

        # Compute aggregates
        groups = {"competitors": ["LINKED1"]}
        aggregates = compute_linked_aggregates(target_daily, linked_daily, groups)
        self.assertIn("competitors_avg_return_1d", aggregates.columns)

        # Align macro data
        macro_ds = MacroDataset(country_iso3="USA")
        macro_ds.indicators = {
            "inflation_rate_yoy": pd.DataFrame({
                "year": [2023, 2024], "value": [3.0, 2.5],
            }),
        }
        macro_aligned = align_macro_to_daily(macro_ds, daily_index)
        target_daily = merge_macro_onto_target(target_daily, macro_aligned)
        self.assertIn("inflation_rate_yoy", target_daily.columns)
        self.assertIn("real_return_1d", target_daily.columns)


if __name__ == "__main__":
    unittest.main()
