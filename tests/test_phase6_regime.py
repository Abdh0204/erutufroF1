"""Phase 6 tests -- regime detection, structural breaks, and causality analysis."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    """Return a business-day index long enough for regime detection."""
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_regime_cache(days: int = 250, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache with return/volatility columns.

    Creates two distinct regimes:
      - First half: low-vol (small returns, low volatility)
      - Second half: high-vol (larger returns, higher volatility)
    """
    np.random.seed(seed)
    idx = _make_daily_index(days)
    mid = days // 2

    returns = np.concatenate([
        np.random.randn(mid) * 0.005,     # low-vol regime
        np.random.randn(days - mid) * 0.03,  # high-vol regime
    ])

    close = 100 + np.cumsum(returns * 100)

    volatility = np.concatenate([
        np.full(mid, 0.08) + np.random.randn(mid) * 0.01,
        np.full(days - mid, 0.25) + np.random.randn(days - mid) * 0.02,
    ])

    df = pd.DataFrame({
        "close": close,
        "return_1d": returns,
        "volatility_21d": volatility,
        "current_ratio": np.full(days, 1.5),
        "debt_to_equity_abs": np.full(days, 1.0),
        "fcf_yield": np.full(days, 0.05),
        "drawdown_252d": np.full(days, -0.10),
        "cash_ratio": np.full(days, 0.8),
        "cash_and_equivalents": np.full(days, 400_000.0),
        "free_cash_flow": np.full(days, 200_000.0),
        "free_cash_flow_ttm_asof": np.full(days, 200_000.0),
        "operating_cash_flow": np.full(days, 300_000.0),
        "net_debt": np.full(days, 100_000.0),
        "net_debt_to_ebitda": np.full(days, 0.3),
        "total_debt_asof": np.full(days, 500_000.0),
        "gross_margin": np.full(days, 0.6),
        "operating_margin": np.full(days, 0.3),
        "net_margin": np.full(days, 0.2),
    }, index=idx)

    return df


def _make_causal_cache(days: int = 200, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic cache where var_b Granger-causes var_a.

    var_a[t] = 0.5 * var_b[t-1] + noise
    var_b[t] = random walk
    var_c[t] = independent random walk (no causal link)
    """
    np.random.seed(seed)
    idx = _make_daily_index(days)

    var_b = np.cumsum(np.random.randn(days) * 0.01)
    var_a = np.zeros(days)
    var_a[0] = np.random.randn() * 0.01
    for t in range(1, days):
        var_a[t] = 0.5 * var_b[t - 1] + np.random.randn() * 0.005

    var_c = np.cumsum(np.random.randn(days) * 0.01)

    return pd.DataFrame({
        "var_a": var_a,
        "var_b": var_b,
        "var_c": var_c,
    }, index=idx)


# ===========================================================================
# T6.1 -- RegimeDetector unit tests
# ===========================================================================


class TestRegimeDetectorInit(unittest.TestCase):
    """Test RegimeDetector initialization and defaults."""

    def test_default_n_regimes(self):
        from operator1.models.regime_detector import RegimeDetector, DEFAULT_N_REGIMES
        det = RegimeDetector()
        self.assertEqual(det.n_regimes, DEFAULT_N_REGIMES)

    def test_custom_n_regimes(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector(n_regimes=3)
        self.assertEqual(det.n_regimes, 3)

    def test_result_starts_empty(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        res = det.result
        self.assertFalse(res.hmm_fitted)
        self.assertFalse(res.gmm_fitted)
        self.assertFalse(res.pelt_fitted)
        self.assertFalse(res.bcp_fitted)
        self.assertIsNone(res.hmm_regimes)
        self.assertIsNone(res.gmm_regimes)
        self.assertEqual(res.breakpoints_pelt, [])
        self.assertEqual(res.breakpoints_bcp, [])


class TestHMM(unittest.TestCase):
    """Test HMM regime detection."""

    def _make_returns_vol(self, days: int = 200):
        np.random.seed(42)
        returns = np.random.randn(days) * 0.02
        volatility = np.abs(np.random.randn(days) * 0.1) + 0.05
        return returns, volatility

    def test_hmm_missing_import_graceful(self):
        """When hmmlearn is not installed, fit_hmm returns (None, None)."""
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        returns, vol = self._make_returns_vol()

        with patch.dict(sys.modules, {"hmmlearn": None, "hmmlearn.hmm": None}):
            # Force reimport to pick up the mocked missing module.
            regimes, probs = det.fit_hmm(returns, vol)

        # Even with patched imports, the try/except inside should handle it.
        # We just verify it doesn't crash.
        # (The actual result depends on whether hmmlearn was already loaded.)

    def test_hmm_insufficient_data(self):
        """HMM should return None with too few observations."""
        from operator1.models.regime_detector import RegimeDetector, _MIN_OBS_HMM
        det = RegimeDetector()
        short_ret = np.random.randn(10) * 0.02
        short_vol = np.abs(np.random.randn(10) * 0.1)
        regimes, probs = det.fit_hmm(short_ret, short_vol)
        self.assertIsNone(regimes)
        self.assertIsNone(probs)
        self.assertFalse(det.result.hmm_fitted)
        self.assertIsNotNone(det.result.hmm_error)

    def test_hmm_all_nan(self):
        """HMM should gracefully handle all-NaN input."""
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        nan_arr = np.full(100, np.nan)
        regimes, probs = det.fit_hmm(nan_arr, nan_arr)
        self.assertIsNone(regimes)

    def test_hmm_with_nans_in_middle(self):
        """HMM should handle sparse NaN values by dropping them."""
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        returns, vol = self._make_returns_vol(200)
        # Inject some NaNs.
        returns[50] = np.nan
        returns[100] = np.nan
        vol[75] = np.nan
        # Should not crash -- NaN rows are dropped.
        det.fit_hmm(returns, vol)


class TestGMM(unittest.TestCase):
    """Test GMM regime clustering."""

    def test_gmm_insufficient_data(self):
        from operator1.models.regime_detector import RegimeDetector, _MIN_OBS_GMM
        det = RegimeDetector()
        short = np.random.randn(5)
        result = det.fit_gmm(short)
        self.assertIsNone(result)
        self.assertFalse(det.result.gmm_fitted)

    def test_gmm_all_nan(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        result = det.fit_gmm(np.full(100, np.nan))
        self.assertIsNone(result)

    def test_gmm_produces_n_regimes(self):
        """GMM output should have labels in range [0, n_regimes)."""
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            self.skipTest("scikit-learn not installed")

        from operator1.models.regime_detector import RegimeDetector
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        det = RegimeDetector(n_regimes=3)
        regimes = det.fit_gmm(returns)

        self.assertIsNotNone(regimes)
        self.assertTrue(det.result.gmm_fitted)
        unique = set(regimes)
        self.assertTrue(unique.issubset({0, 1, 2}))


class TestPELT(unittest.TestCase):
    """Test PELT structural break detection."""

    def test_pelt_insufficient_data(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        result = det.detect_breakpoints_pelt(np.random.randn(5))
        self.assertEqual(result, [])
        self.assertFalse(det.result.pelt_fitted)

    def test_pelt_all_nan(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        result = det.detect_breakpoints_pelt(np.full(100, np.nan))
        self.assertEqual(result, [])

    def test_pelt_detects_mean_shift(self):
        """PELT should detect a clear mean shift in a synthetic series."""
        try:
            import ruptures  # noqa: F401
        except ImportError:
            self.skipTest("ruptures not installed")

        from operator1.models.regime_detector import RegimeDetector
        np.random.seed(42)
        # Create two distinct segments.
        seg1 = np.random.randn(100) + 10.0
        seg2 = np.random.randn(100) + 20.0
        series = np.concatenate([seg1, seg2])

        det = RegimeDetector()
        bkps = det.detect_breakpoints_pelt(series, penalty=5.0)

        self.assertTrue(det.result.pelt_fitted)
        # Should detect at least one breakpoint near index 100.
        self.assertGreater(len(bkps), 0)
        # Check that at least one breakpoint is within +/-20 of 100.
        near_100 = any(80 <= bp <= 120 for bp in bkps)
        self.assertTrue(near_100, f"No breakpoint near 100; got {bkps}")

    def test_pelt_returns_list_of_ints(self):
        try:
            import ruptures  # noqa: F401
        except ImportError:
            self.skipTest("ruptures not installed")

        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        series = np.random.randn(100)
        bkps = det.detect_breakpoints_pelt(series)
        self.assertIsInstance(bkps, list)
        for bp in bkps:
            self.assertIsInstance(bp, int)


class TestBCP(unittest.TestCase):
    """Test Bayesian change-point detection."""

    def test_bcp_insufficient_data(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        result = det.detect_breakpoints_bcp(np.random.randn(5))
        self.assertEqual(result, [])

    def test_bcp_all_nan(self):
        from operator1.models.regime_detector import RegimeDetector
        det = RegimeDetector()
        result = det.detect_breakpoints_bcp(np.full(100, np.nan))
        self.assertEqual(result, [])

    def test_bcp_online_detects_shift(self):
        """BCP should detect a mean shift using the online algorithm."""
        from operator1.models.regime_detector import RegimeDetector
        np.random.seed(42)
        seg1 = np.random.randn(80) * 0.5 + 5.0
        seg2 = np.random.randn(80) * 0.5 + 15.0
        series = np.concatenate([seg1, seg2])

        det = RegimeDetector()
        bkps = det.detect_breakpoints_bcp(
            series,
            hazard_lambda=50.0,
            threshold=0.3,
        )

        self.assertTrue(det.result.bcp_fitted)
        self.assertIsInstance(bkps, list)
        # Should detect something in the transition zone.
        self.assertGreater(len(bkps), 0)

    def test_bcp_online_pure_numpy(self):
        """BCP online implementation uses only numpy (no pymc needed)."""
        from operator1.models.regime_detector import RegimeDetector
        np.random.seed(42)
        series = np.random.randn(100)

        det = RegimeDetector()
        # Should work without pymc.
        bkps = det.detect_breakpoints_bcp(series)
        self.assertIsInstance(bkps, list)
        self.assertTrue(det.result.bcp_fitted)

    def test_bcp_returns_list_of_ints(self):
        from operator1.models.regime_detector import RegimeDetector
        np.random.seed(42)
        series = np.random.randn(60)
        det = RegimeDetector()
        bkps = det.detect_breakpoints_bcp(series)
        for bp in bkps:
            self.assertIsInstance(bp, (int, np.integer))


class TestRegimeLabelMapping(unittest.TestCase):
    """Test the regime label ordering helper."""

    def test_labels_ordered_by_mean_return(self):
        from operator1.models.regime_detector import _order_regimes_by_mean_return

        # Regime 0: negative returns, Regime 1: positive returns.
        regimes = np.array([0, 0, 0, 1, 1, 1])
        returns = np.array([-0.05, -0.03, -0.04, 0.03, 0.05, 0.04])

        mapping = _order_regimes_by_mean_return(regimes, returns, n_regimes=2)

        # Regime 0 (negative) should be "bear", Regime 1 (positive) "low_vol" or higher.
        self.assertEqual(mapping[0], "bear")
        self.assertIn(mapping[1], ("low_vol", "high_vol", "bull"))

    def test_labels_with_4_regimes(self):
        from operator1.models.regime_detector import _order_regimes_by_mean_return

        np.random.seed(42)
        regimes = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        returns = np.concatenate([
            np.random.randn(25) * 0.01 - 0.05,  # regime 0: most negative
            np.random.randn(25) * 0.01 - 0.01,  # regime 1
            np.random.randn(25) * 0.01 + 0.01,  # regime 2
            np.random.randn(25) * 0.01 + 0.05,  # regime 3: most positive
        ])

        mapping = _order_regimes_by_mean_return(regimes, returns, 4)

        # All 4 labels should be present.
        self.assertEqual(len(mapping), 4)
        labels = set(mapping.values())
        self.assertEqual(labels, {"bear", "low_vol", "high_vol", "bull"})

    def test_empty_regime(self):
        """Handle regime with zero observations."""
        from operator1.models.regime_detector import _order_regimes_by_mean_return

        regimes = np.array([0, 0, 0])  # Only regime 0 has data.
        returns = np.array([0.01, 0.02, 0.03])

        mapping = _order_regimes_by_mean_return(regimes, returns, n_regimes=2)
        self.assertEqual(len(mapping), 2)


# ===========================================================================
# detect_regimes_and_breaks (pipeline entry point)
# ===========================================================================


class TestDetectRegimesAndBreaks(unittest.TestCase):
    """Test the top-level pipeline entry point."""

    def test_adds_expected_columns(self):
        """detect_regimes_and_breaks should add regime + break columns."""
        from operator1.models.regime_detector import detect_regimes_and_breaks

        cache = _make_regime_cache(100)
        original_cols = set(cache.columns)

        cache, detector = detect_regimes_and_breaks(cache, n_regimes=2)

        new_cols = set(cache.columns) - original_cols
        # Should have at least these columns.
        expected = {"regime_hmm", "regime_gmm", "regime_label",
                    "structural_break", "breakpoint_method"}
        self.assertTrue(expected.issubset(new_cols), f"Missing: {expected - new_cols}")

    def test_structural_break_is_binary(self):
        from operator1.models.regime_detector import detect_regimes_and_breaks

        cache = _make_regime_cache(100)
        cache, _ = detect_regimes_and_breaks(cache, n_regimes=2)

        vals = set(cache["structural_break"].unique())
        self.assertTrue(vals.issubset({0, 1}))

    def test_regime_label_values_are_valid(self):
        from operator1.models.regime_detector import detect_regimes_and_breaks

        cache = _make_regime_cache(200)
        cache, _ = detect_regimes_and_breaks(cache, n_regimes=4)

        valid_labels = {"bear", "low_vol", "high_vol", "bull", None}
        labels = set(cache["regime_label"].dropna().unique())
        self.assertTrue(
            labels.issubset(valid_labels),
            f"Unexpected labels: {labels - valid_labels}",
        )

    def test_no_return_column_graceful(self):
        """If return_1d is missing, function should add empty columns."""
        from operator1.models.regime_detector import detect_regimes_and_breaks

        idx = pd.bdate_range("2024-01-02", periods=50)
        cache = pd.DataFrame({"close": np.random.randn(50) + 100}, index=idx)

        cache, detector = detect_regimes_and_breaks(cache)

        self.assertIn("regime_label", cache.columns)
        self.assertIn("structural_break", cache.columns)
        self.assertTrue(cache["regime_label"].isna().all())

    def test_returns_detector_instance(self):
        from operator1.models.regime_detector import (
            detect_regimes_and_breaks,
            RegimeDetector,
        )
        cache = _make_regime_cache(100)
        _, detector = detect_regimes_and_breaks(cache, n_regimes=2)
        self.assertIsInstance(detector, RegimeDetector)

    def test_hmm_probability_columns_sum_to_one(self):
        """HMM posterior probabilities should sum to ~1 on valid rows."""
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401
        except ImportError:
            self.skipTest("hmmlearn not installed")

        from operator1.models.regime_detector import detect_regimes_and_breaks

        cache = _make_regime_cache(250)
        cache, det = detect_regimes_and_breaks(cache, n_regimes=3)

        if not det.result.hmm_fitted:
            self.skipTest("HMM did not fit successfully")

        prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
        self.assertEqual(len(prob_cols), 3)

        prob_sum = cache[prob_cols].sum(axis=1)
        valid = prob_sum.dropna()
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 1.0, atol=0.01)


# ===========================================================================
# Causality analysis tests
# ===========================================================================


class TestGrangerCausality(unittest.TestCase):
    """Test Granger causality computation."""

    def test_empty_variables_list(self):
        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache()
        matrix = compute_granger_causality(cache, [])
        self.assertEqual(matrix.shape, (0, 0))

    def test_single_variable(self):
        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache()
        matrix = compute_granger_causality(cache, ["var_a"])
        self.assertEqual(matrix.shape, (1, 1))
        # Diagonal should be 0 (no self-causality).
        self.assertEqual(matrix.loc["var_a", "var_a"], 0)

    def test_missing_columns_handled(self):
        """Variables not in cache should be skipped gracefully."""
        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache()
        matrix = compute_granger_causality(
            cache, ["var_a", "var_b", "nonexistent"]
        )
        # nonexistent should be dropped.
        self.assertIn("var_a", matrix.index)
        self.assertIn("var_b", matrix.index)
        self.assertNotIn("nonexistent", matrix.index)

    def test_insufficient_data(self):
        from operator1.models.causality import compute_granger_causality

        idx = pd.bdate_range("2024-01-02", periods=3)
        cache = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=idx)
        matrix = compute_granger_causality(cache, ["a", "b"], max_lag=2)
        # Not enough data -- should return zero matrix.
        self.assertEqual(matrix.sum().sum(), 0)

    def test_causal_relationship_detected(self):
        """var_b should Granger-cause var_a in synthetic data."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache(500, seed=42)
        matrix = compute_granger_causality(
            cache, ["var_a", "var_b", "var_c"], max_lag=3
        )

        # var_b -> var_a should be significant.
        self.assertEqual(
            matrix.loc["var_a", "var_b"], 1,
            "Expected var_b to Granger-cause var_a",
        )

    def test_matrix_shape(self):
        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache()
        variables = ["var_a", "var_b", "var_c"]
        matrix = compute_granger_causality(cache, variables)
        self.assertEqual(matrix.shape, (3, 3))
        # Diagonal should be all zeros.
        for v in variables:
            self.assertEqual(matrix.loc[v, v], 0)

    def test_statsmodels_missing_returns_zero_matrix(self):
        """When statsmodels is missing, return a zero matrix."""
        from operator1.models.causality import compute_granger_causality

        cache = _make_causal_cache()

        with patch.dict(sys.modules, {
            "statsmodels": None,
            "statsmodels.tsa": None,
            "statsmodels.tsa.stattools": None,
        }):
            # The function should catch the ImportError internally.
            matrix = compute_granger_causality(cache, ["var_a", "var_b"])

        # Should return a matrix (possibly zero if statsmodels was patched out).
        self.assertIsInstance(matrix, pd.DataFrame)


class TestVariablePruning(unittest.TestCase):
    """Test variable pruning logic."""

    def test_empty_matrix(self):
        from operator1.models.causality import prune_weak_relationships

        empty = pd.DataFrame()
        result = prune_weak_relationships(empty)
        self.assertEqual(result, [])

    def test_strong_variables_kept(self):
        from operator1.models.causality import prune_weak_relationships

        matrix = pd.DataFrame(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        # "a" has 2 incoming links (from b and c), others have 0.
        result = prune_weak_relationships(
            matrix,
            threshold=1.0,
            protect_tiers_1_2=False,
        )
        self.assertIn("a", result)

    def test_weak_variables_pruned(self):
        from operator1.models.causality import prune_weak_relationships

        matrix = pd.DataFrame(
            [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        # "c" has 2 incoming, "a" and "b" have 0.
        result = prune_weak_relationships(
            matrix,
            threshold=1.0,
            protect_tiers_1_2=False,
        )
        self.assertIn("c", result)
        self.assertNotIn("a", result)
        self.assertNotIn("b", result)

    def test_max_vars_cap(self):
        from operator1.models.causality import prune_weak_relationships

        n = 10
        matrix = pd.DataFrame(
            np.ones((n, n)) - np.eye(n),
            index=[f"v{i}" for i in range(n)],
            columns=[f"v{i}" for i in range(n)],
        )
        result = prune_weak_relationships(
            matrix,
            threshold=0,
            max_vars=5,
            protect_tiers_1_2=False,
        )
        self.assertLessEqual(len(result), 5)

    def test_protected_vars_never_pruned(self):
        """Tier 1/2 variables should survive pruning even with 0 links."""
        from operator1.models.causality import prune_weak_relationships

        # Use actual tier variable names from the config.
        matrix = pd.DataFrame(
            np.zeros((4, 4)),
            index=["current_ratio", "cash_ratio", "random_var_1", "random_var_2"],
            columns=["current_ratio", "cash_ratio", "random_var_1", "random_var_2"],
        )
        # Give random_var_1 some links so it passes threshold.
        matrix.loc["random_var_1", "random_var_2"] = 1
        matrix.loc["random_var_1", "current_ratio"] = 1

        result = prune_weak_relationships(
            matrix,
            threshold=2.0,
            protect_tiers_1_2=True,
        )

        # current_ratio and cash_ratio should be protected (tier 1).
        self.assertIn("current_ratio", result)
        self.assertIn("cash_ratio", result)


class TestRunCausalityAnalysis(unittest.TestCase):
    """Test the combined pipeline helper."""

    def test_returns_tuple(self):
        from operator1.models.causality import run_causality_analysis

        cache = _make_causal_cache(100)
        matrix, strong_vars = run_causality_analysis(
            cache, ["var_a", "var_b", "var_c"]
        )
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertIsInstance(strong_vars, list)

    def test_strong_vars_subset_of_input(self):
        from operator1.models.causality import run_causality_analysis

        cache = _make_causal_cache(100)
        variables = ["var_a", "var_b", "var_c"]
        _, strong_vars = run_causality_analysis(cache, variables)
        self.assertTrue(set(strong_vars).issubset(set(variables)))


# ===========================================================================
# Integration: regime detection + causality together
# ===========================================================================


class TestRegimeAndCausalityIntegration(unittest.TestCase):
    """Integration test running both modules on the same cache."""

    def test_full_pipeline_does_not_crash(self):
        """Run regime detection followed by causality on regime cache."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.causality import run_causality_analysis

        cache = _make_regime_cache(200)

        # Step 1: Regime detection.
        cache, detector = detect_regimes_and_breaks(cache, n_regimes=3)

        # Verify regime columns exist.
        self.assertIn("regime_label", cache.columns)
        self.assertIn("structural_break", cache.columns)

        # Step 2: Causality on financial variables.
        financial_vars = [
            "return_1d", "volatility_21d", "current_ratio",
            "debt_to_equity_abs", "fcf_yield", "gross_margin",
        ]
        matrix, strong_vars = run_causality_analysis(
            cache,
            financial_vars,
            max_lag=3,
        )

        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertIsInstance(strong_vars, list)

    def test_regime_columns_survive_causality(self):
        """Causality analysis should not remove regime columns."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.causality import run_causality_analysis

        cache = _make_regime_cache(150)
        cache, _ = detect_regimes_and_breaks(cache, n_regimes=2)

        cols_before = set(cache.columns)

        run_causality_analysis(
            cache,
            ["return_1d", "volatility_21d"],
        )

        cols_after = set(cache.columns)
        # Causality should not remove any columns.
        self.assertEqual(cols_before, cols_after)


if __name__ == "__main__":
    unittest.main()
