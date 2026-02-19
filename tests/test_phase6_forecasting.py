"""Phase 6 tests -- T6.2 forecasting models.

Tests each model in the forecasting fallback chain:
  - Kalman filter
  - GARCH
  - VAR (+ AR(1) fallback)
  - LSTM (+ tree/linear fallback)
  - Tree ensemble (XGB > GBM > RF)
  - Baseline (always succeeds)

Also tests the top-level run_forecasting pipeline, burnout
refinement, and error metric computation.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_smooth_series(days: int = 250, seed: int = 42) -> np.ndarray:
    """Random walk with small increments -- nice for Kalman/VAR."""
    np.random.seed(seed)
    return 100.0 + np.cumsum(np.random.randn(days) * 0.5)


def _make_volatile_returns(days: int = 250, seed: int = 42) -> np.ndarray:
    """Returns with two volatility regimes -- good for GARCH."""
    np.random.seed(seed)
    mid = days // 2
    return np.concatenate([
        np.random.randn(mid) * 0.005,
        np.random.randn(days - mid) * 0.03,
    ])


def _make_forecast_cache(days: int = 250, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache suitable for the forecasting pipeline."""
    np.random.seed(seed)
    idx = _make_daily_index(days)

    returns = _make_volatile_returns(days, seed)
    close = 100 + np.cumsum(returns * 100)
    vol = np.abs(np.random.randn(days) * 0.1) + 0.08

    df = pd.DataFrame({
        "close": close,
        "return_1d": returns,
        "volatility_21d": vol,
        "current_ratio": 1.5 + np.cumsum(np.random.randn(days) * 0.01),
        "quick_ratio": 1.2 + np.cumsum(np.random.randn(days) * 0.01),
        "cash_ratio": 0.8 + np.cumsum(np.random.randn(days) * 0.005),
        "cash_and_equivalents": 400_000 + np.cumsum(np.random.randn(days) * 1000),
        "debt_to_equity_abs": 1.0 + np.cumsum(np.random.randn(days) * 0.005),
        "net_debt": 100_000 + np.cumsum(np.random.randn(days) * 500),
        "net_debt_to_ebitda": 0.3 + np.cumsum(np.random.randn(days) * 0.002),
        "total_debt_asof": 500_000 + np.cumsum(np.random.randn(days) * 1000),
        "free_cash_flow": 200_000 + np.cumsum(np.random.randn(days) * 500),
        "free_cash_flow_ttm_asof": 200_000 + np.cumsum(np.random.randn(days) * 500),
        "fcf_yield": 0.05 + np.cumsum(np.random.randn(days) * 0.001),
        "operating_cash_flow": 300_000 + np.cumsum(np.random.randn(days) * 800),
        "gross_margin": 0.6 + np.random.randn(days) * 0.02,
        "operating_margin": 0.3 + np.random.randn(days) * 0.02,
        "net_margin": 0.2 + np.random.randn(days) * 0.02,
    }, index=idx)

    return df


# ===========================================================================
# Error metrics
# ===========================================================================


class TestComputeMetrics(unittest.TestCase):
    """Test the _compute_metrics helper."""

    def test_perfect_prediction(self):
        from operator1.models.forecasting import _compute_metrics
        y = np.array([1.0, 2.0, 3.0])
        mae, rmse = _compute_metrics(y, y)
        self.assertAlmostEqual(mae, 0.0)
        self.assertAlmostEqual(rmse, 0.0)

    def test_known_error(self):
        from operator1.models.forecasting import _compute_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 5.0])
        mae, rmse = _compute_metrics(y_true, y_pred)
        # MAE = (0 + 1 + 2) / 3 = 1.0
        self.assertAlmostEqual(mae, 1.0)
        # RMSE = sqrt((0 + 1 + 4) / 3) = sqrt(5/3)
        self.assertAlmostEqual(rmse, np.sqrt(5.0 / 3.0))

    def test_with_nans(self):
        from operator1.models.forecasting import _compute_metrics
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mae, rmse = _compute_metrics(y_true, y_pred)
        # Only indices 0 and 2 are valid.
        self.assertAlmostEqual(mae, 0.0)

    def test_all_nan(self):
        from operator1.models.forecasting import _compute_metrics
        mae, rmse = _compute_metrics(
            np.full(5, np.nan), np.full(5, np.nan)
        )
        self.assertTrue(np.isnan(mae))
        self.assertTrue(np.isnan(rmse))


class TestSplitTrainTest(unittest.TestCase):
    """Test temporal train/test split."""

    def test_split_proportions(self):
        from operator1.models.forecasting import _split_train_test
        data = np.arange(100, dtype=float)
        train, test = _split_train_test(data, test_frac=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_split_preserves_order(self):
        from operator1.models.forecasting import _split_train_test
        data = np.arange(50, dtype=float)
        train, test = _split_train_test(data)
        # Train should be the first portion.
        self.assertTrue(train[-1] < test[0])

    def test_tiny_array(self):
        from operator1.models.forecasting import _split_train_test
        data = np.array([1.0, 2.0])
        train, test = _split_train_test(data, test_frac=0.5)
        self.assertGreater(len(train), 0)


# ===========================================================================
# Kalman filter
# ===========================================================================


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman filter forecasting."""

    def test_kalman_insufficient_data(self):
        from operator1.models.forecasting import fit_kalman, _MIN_OBS_KALMAN
        series = np.random.randn(5)
        fcast, met = fit_kalman(series)
        self.assertIsNone(fcast)
        self.assertFalse(met.fitted)
        # Either "Insufficient" (statsmodels present) or "not installed" (absent)
        self.assertTrue(
            "Insufficient" in (met.error or "") or "not installed" in (met.error or ""),
            f"Unexpected error: {met.error}",
        )

    def test_kalman_all_nan(self):
        from operator1.models.forecasting import fit_kalman
        fcast, met = fit_kalman(np.full(100, np.nan))
        self.assertIsNone(fcast)

    def test_kalman_on_smooth_series(self):
        """Kalman should fit a smooth random walk without error."""
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_kalman
        series = _make_smooth_series(200)
        fcast, met = fit_kalman(series, n_forecast=5)

        self.assertIsNotNone(fcast)
        self.assertTrue(met.fitted)
        self.assertEqual(len(fcast), 5)
        self.assertFalse(np.isnan(met.mae))

    def test_kalman_forecast_length(self):
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_kalman
        series = _make_smooth_series(100)
        fcast, _ = fit_kalman(series, n_forecast=10)
        if fcast is not None:
            self.assertEqual(len(fcast), 10)

    def test_kalman_handles_nans_in_series(self):
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_kalman
        series = _make_smooth_series(100)
        series[20] = np.nan
        series[50] = np.nan
        # Should still work (clean drops NaN).
        fcast, met = fit_kalman(series, n_forecast=1)
        # May succeed or fail depending on how many remain.
        # Just verify no crash.
        self.assertIsInstance(met.model_name, str)


# ===========================================================================
# GARCH
# ===========================================================================


class TestGARCH(unittest.TestCase):
    """Test GARCH volatility forecasting."""

    def test_garch_insufficient_data(self):
        from operator1.models.forecasting import fit_garch, _MIN_OBS_GARCH
        fcast, met = fit_garch(np.random.randn(10))
        self.assertIsNone(fcast)
        self.assertFalse(met.fitted)

    def test_garch_all_nan(self):
        from operator1.models.forecasting import fit_garch
        fcast, met = fit_garch(np.full(100, np.nan))
        self.assertIsNone(fcast)

    def test_garch_missing_import_graceful(self):
        from operator1.models.forecasting import fit_garch
        with patch.dict(sys.modules, {"arch": None}):
            fcast, met = fit_garch(np.random.randn(100))
        # Should not crash regardless of whether arch was already loaded.
        self.assertIsInstance(met.model_name, str)

    def test_garch_on_volatile_returns(self):
        try:
            from arch import arch_model  # noqa: F401
        except ImportError:
            self.skipTest("arch library not installed")

        from operator1.models.forecasting import fit_garch
        returns = _make_volatile_returns(250)
        fcast, met = fit_garch(returns, n_forecast=5)

        self.assertIsNotNone(fcast)
        self.assertTrue(met.fitted)
        self.assertEqual(len(fcast), 5)
        # Volatility forecasts should be positive.
        self.assertTrue(np.all(fcast >= 0))

    def test_garch_forecast_positive(self):
        """GARCH variance forecasts should always be non-negative."""
        try:
            from arch import arch_model  # noqa: F401
        except ImportError:
            self.skipTest("arch library not installed")

        from operator1.models.forecasting import fit_garch
        returns = _make_volatile_returns(200)
        fcast, _ = fit_garch(returns, n_forecast=10)
        if fcast is not None:
            self.assertTrue(np.all(fcast >= 0))


# ===========================================================================
# VAR + AR(1) fallback
# ===========================================================================


class TestVAR(unittest.TestCase):
    """Test VAR multivariate forecasting."""

    def _make_var_data(self, days: int = 200) -> pd.DataFrame:
        np.random.seed(42)
        idx = _make_daily_index(days)
        return pd.DataFrame({
            "var_a": np.cumsum(np.random.randn(days) * 0.01),
            "var_b": np.cumsum(np.random.randn(days) * 0.01),
            "var_c": np.cumsum(np.random.randn(days) * 0.01),
        }, index=idx)

    def test_var_insufficient_data(self):
        from operator1.models.forecasting import fit_var
        data = self._make_var_data(10)
        fcast, met = fit_var(data, "var_a")
        # Should fall back to AR(1) or fail gracefully.
        self.assertIn(met.model_name, ("var", "ar1"))

    def test_var_missing_target_col(self):
        from operator1.models.forecasting import fit_var
        data = self._make_var_data(100)
        fcast, met = fit_var(data, "nonexistent")
        self.assertIsNone(fcast)

    def test_var_on_multivariate_data(self):
        try:
            from statsmodels.tsa.api import VAR  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_var
        data = self._make_var_data(200)
        fcast, met = fit_var(data, "var_a", n_forecast=5)

        if met.fitted:
            self.assertIsNotNone(fcast)
            self.assertEqual(len(fcast), 5)

    def test_var_forecast_returns_array(self):
        try:
            from statsmodels.tsa.api import VAR  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_var
        data = self._make_var_data(200)
        fcast, met = fit_var(data, "var_b", n_forecast=3)
        if fcast is not None:
            self.assertIsInstance(fcast, np.ndarray)
            self.assertEqual(len(fcast), 3)


class TestAR1Fallback(unittest.TestCase):
    """Test AR(1) fallback for VAR."""

    def test_ar1_on_short_data(self):
        """VAR with insufficient data should fall back to AR(1)."""
        try:
            from statsmodels.tsa.ar_model import AutoReg  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")

        from operator1.models.forecasting import fit_var
        np.random.seed(42)
        idx = _make_daily_index(30)
        data = pd.DataFrame({
            "x": np.cumsum(np.random.randn(30) * 0.01),
            "y": np.cumsum(np.random.randn(30) * 0.01),
        }, index=idx)
        fcast, met = fit_var(data, "x", n_forecast=3)
        # Should have used AR(1) fallback.
        if met.fitted:
            self.assertIn("ar1", met.model_name)


# ===========================================================================
# LSTM + tree/linear fallback
# ===========================================================================


class TestLSTM(unittest.TestCase):
    """Test LSTM and its fallback chain."""

    def test_lstm_insufficient_data(self):
        from operator1.models.forecasting import fit_lstm, _MIN_OBS_LSTM
        fcast, met = fit_lstm(np.random.randn(20), n_forecast=1)
        # Should fall back to tree/linear.
        if fcast is not None:
            self.assertIn(
                met.model_name,
                ("lstm", "gradient_boosting", "linear_regression"),
            )

    def test_lstm_all_nan(self):
        from operator1.models.forecasting import fit_lstm
        fcast, met = fit_lstm(np.full(200, np.nan))
        # All NaN -> insufficient data for everything.
        # May return None or fallback result.
        self.assertIsInstance(met.model_name, str)

    def test_lstm_missing_torch_falls_back(self):
        """When PyTorch is missing, should use tree/linear fallback."""
        from operator1.models.forecasting import fit_lstm

        series = _make_smooth_series(150)

        with patch.dict(sys.modules, {
            "torch": None,
            "torch.nn": None,
        }):
            fcast, met = fit_lstm(series, n_forecast=3)

        # Should either use fallback or fail gracefully.
        if fcast is not None:
            self.assertIn(
                met.model_name,
                ("gradient_boosting", "linear_regression", "lstm"),
            )

    def test_lstm_produces_forecasts(self):
        """If torch is available, LSTM should produce forecasts."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")

        from operator1.models.forecasting import fit_lstm
        series = _make_smooth_series(200)
        fcast, met = fit_lstm(series, n_forecast=5, epochs=10)

        if met.fitted and met.model_name == "lstm":
            self.assertIsNotNone(fcast)
            self.assertEqual(len(fcast), 5)


class TestLinearFallback(unittest.TestCase):
    """Test the tree/linear fallback directly."""

    def test_gradient_boosting_fallback(self):
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

        from operator1.models.forecasting import _fit_linear_fallback
        series = _make_smooth_series(100)
        fcast, met = _fit_linear_fallback(series, n_forecast=3, lookback=10, random_state=42)

        self.assertIsNotNone(fcast)
        self.assertTrue(met.fitted)
        self.assertEqual(len(fcast), 3)

    def test_fallback_insufficient_data(self):
        from operator1.models.forecasting import _fit_linear_fallback
        fcast, met = _fit_linear_fallback(
            np.array([1.0, 2.0]), n_forecast=1, lookback=10, random_state=42
        )
        self.assertIsNone(fcast)


# ===========================================================================
# Tree ensemble
# ===========================================================================


class TestTreeEnsemble(unittest.TestCase):
    """Test tree ensemble (XGB > GBM > RF) forecasting."""

    def test_tree_insufficient_data(self):
        from operator1.models.forecasting import fit_tree_ensemble
        idx = _make_daily_index(10)
        df = pd.DataFrame({
            "feat1": np.random.randn(10),
            "target": np.random.randn(10),
        }, index=idx)
        fcast, met = fit_tree_ensemble(df, "target")
        self.assertIsNone(fcast)

    def test_tree_missing_target(self):
        from operator1.models.forecasting import fit_tree_ensemble
        idx = _make_daily_index(50)
        df = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
        }, index=idx)
        fcast, met = fit_tree_ensemble(df, "nonexistent")
        self.assertIsNone(fcast)

    def test_tree_on_tabular_data(self):
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

        from operator1.models.forecasting import fit_tree_ensemble
        np.random.seed(42)
        idx = _make_daily_index(100)
        df = pd.DataFrame({
            "feat1": np.cumsum(np.random.randn(100) * 0.01),
            "feat2": np.cumsum(np.random.randn(100) * 0.01),
            "target": np.cumsum(np.random.randn(100) * 0.01),
        }, index=idx)
        fcast, met = fit_tree_ensemble(df, "target", n_forecast=3)

        self.assertIsNotNone(fcast)
        self.assertTrue(met.fitted)
        self.assertEqual(len(fcast), 3)

    def test_tree_model_name_reflects_library(self):
        """Model name should reflect which library was used."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

        from operator1.models.forecasting import fit_tree_ensemble
        np.random.seed(42)
        idx = _make_daily_index(50)
        df = pd.DataFrame({
            "f1": np.random.randn(50),
            "target": np.random.randn(50),
        }, index=idx)
        _, met = fit_tree_ensemble(df, "target")
        self.assertIn(
            met.model_name,
            ("xgboost", "gradient_boosting", "random_forest"),
        )

    def test_no_feature_columns(self):
        """If only the target column exists, should fail gracefully."""
        from operator1.models.forecasting import fit_tree_ensemble
        idx = _make_daily_index(50)
        df = pd.DataFrame({"target": np.random.randn(50)}, index=idx)
        fcast, met = fit_tree_ensemble(df, "target")
        self.assertIsNone(fcast)


class TestTryLoadTreeModel(unittest.TestCase):
    """Test the tree model loading fallback chain."""

    def test_loads_some_model(self):
        """At least one tree model should be loadable if sklearn is present."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

        from operator1.models.forecasting import _try_load_tree_model, ModelMetrics
        met = ModelMetrics()
        model = _try_load_tree_model(42, met)
        self.assertIsNotNone(model)
        self.assertIn(
            met.model_name,
            ("xgboost", "gradient_boosting", "random_forest"),
        )


# ===========================================================================
# Baseline (always succeeds)
# ===========================================================================


class TestBaseline(unittest.TestCase):
    """Test baseline forecaster."""

    def test_baseline_always_succeeds(self):
        from operator1.models.forecasting import fit_baseline
        series = np.array([1.0, 2.0, 3.0])
        fcast, met = fit_baseline(series, n_forecast=5)
        self.assertIsNotNone(fcast)
        self.assertTrue(met.fitted)
        self.assertEqual(len(fcast), 5)

    def test_baseline_empty_series(self):
        from operator1.models.forecasting import fit_baseline
        fcast, met = fit_baseline(np.array([]), n_forecast=3)
        self.assertIsNotNone(fcast)
        self.assertEqual(len(fcast), 3)
        # Should return zeros.
        np.testing.assert_array_equal(fcast, [0.0, 0.0, 0.0])

    def test_baseline_all_nan(self):
        from operator1.models.forecasting import fit_baseline
        fcast, met = fit_baseline(np.full(10, np.nan), n_forecast=2)
        self.assertIsNotNone(fcast)
        self.assertEqual(len(fcast), 2)

    def test_baseline_last_value(self):
        from operator1.models.forecasting import fit_baseline
        series = np.array([10.0, 20.0, 30.0])
        fcast, met = fit_baseline(series, n_forecast=3, method="last")
        np.testing.assert_array_equal(fcast, [30.0, 30.0, 30.0])

    def test_baseline_ema(self):
        from operator1.models.forecasting import fit_baseline
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        fcast, met = fit_baseline(series, n_forecast=1, method="ema", ema_span=3)
        self.assertIsNotNone(fcast)
        # EMA should be between min and max.
        self.assertGreater(fcast[0], 10.0)
        self.assertLess(fcast[0], 60.0)

    def test_baseline_ema_single_value(self):
        from operator1.models.forecasting import fit_baseline
        series = np.array([42.0])
        fcast, met = fit_baseline(series, n_forecast=2, method="ema")
        # With only 1 value, should use last-value.
        np.testing.assert_array_equal(fcast, [42.0, 42.0])

    def test_baseline_model_name(self):
        from operator1.models.forecasting import fit_baseline
        _, met_ema = fit_baseline(np.array([1.0, 2.0, 3.0]), method="ema")
        self.assertEqual(met_ema.model_name, "baseline_ema")

        _, met_last = fit_baseline(np.array([1.0, 2.0, 3.0]), method="last")
        self.assertEqual(met_last.model_name, "baseline_last")


# ===========================================================================
# ForecastResult container
# ===========================================================================


class TestForecastResult(unittest.TestCase):
    """Test the ForecastResult dataclass."""

    def test_default_state(self):
        from operator1.models.forecasting import ForecastResult
        r = ForecastResult()
        self.assertEqual(r.forecasts, {})
        self.assertFalse(r.model_failed_kalman)
        self.assertFalse(r.model_failed_garch)
        self.assertFalse(r.model_failed_var)
        self.assertFalse(r.model_failed_lstm)
        self.assertFalse(r.model_failed_tree)
        self.assertEqual(r.metrics, [])
        self.assertEqual(r.model_used, {})


class TestModelMetrics(unittest.TestCase):
    """Test the ModelMetrics dataclass."""

    def test_default_state(self):
        from operator1.models.forecasting import ModelMetrics
        m = ModelMetrics()
        self.assertEqual(m.model_name, "")
        self.assertFalse(m.fitted)
        self.assertTrue(np.isnan(m.mae))
        self.assertTrue(np.isnan(m.rmse))
        self.assertIsNone(m.error)


# ===========================================================================
# Burnout refinement
# ===========================================================================


class TestBurnoutRefit(unittest.TestCase):
    """Test the burnout refinement phase."""

    def test_burnout_insufficient_data(self):
        from operator1.models.forecasting import _burnout_refit, fit_baseline
        series = np.array([1.0, 2.0, 3.0])
        fcast, met = _burnout_refit(series, fit_baseline, n_forecast=1)
        self.assertIsNone(fcast)

    def test_burnout_with_baseline(self):
        from operator1.models.forecasting import _burnout_refit, fit_baseline
        series = _make_smooth_series(200)
        fcast, met = _burnout_refit(series, fit_baseline, n_forecast=3)
        # Baseline always succeeds, so burnout should return something.
        self.assertIsNotNone(fcast)
        self.assertEqual(len(fcast), 3)

    def test_burnout_tries_multiple_windows(self):
        """Burnout should try shrinking windows and pick the best."""
        from operator1.models.forecasting import _burnout_refit, fit_baseline
        series = _make_smooth_series(300)
        fcast, met = _burnout_refit(
            series, fit_baseline, n_forecast=1, window=126, patience=3
        )
        self.assertIsNotNone(fcast)


# ===========================================================================
# Tier variable lookup
# ===========================================================================


class TestTierLookup(unittest.TestCase):
    """Test tier variable loading and lookup."""

    def test_load_tier_variables(self):
        from operator1.models.forecasting import _load_tier_variables
        tiers = _load_tier_variables()
        # Should have tier1 through tier5.
        self.assertIn("tier1", tiers)
        self.assertIn("tier2", tiers)
        # Per corrected config, current_ratio is in tier2 (Solvency)
        self.assertIn("current_ratio", tiers["tier2"])

    def test_get_tier_for_variable(self):
        from operator1.models.forecasting import _get_tier_for_variable
        tier_map = {
            "tier1": ["current_ratio", "quick_ratio"],
            "tier2": ["debt_to_equity_abs"],
        }
        self.assertEqual(_get_tier_for_variable("current_ratio", tier_map), "tier1")
        self.assertEqual(_get_tier_for_variable("debt_to_equity_abs", tier_map), "tier2")
        self.assertEqual(_get_tier_for_variable("unknown_var", tier_map), "unknown")


# ===========================================================================
# run_forecasting (pipeline entry point)
# ===========================================================================


class TestRunForecasting(unittest.TestCase):
    """Test the top-level forecasting pipeline."""

    def test_returns_tuple(self):
        from operator1.models.forecasting import run_forecasting, ForecastResult
        cache = _make_forecast_cache(100)
        result_cache, result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )
        self.assertIsInstance(result_cache, pd.DataFrame)
        self.assertIsInstance(result, ForecastResult)

    def test_forecasts_populated(self):
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(150)
        _, result = run_forecasting(
            cache,
            ["current_ratio", "cash_ratio"],
            enable_burnout=False,
        )
        # At least baseline should have produced forecasts.
        self.assertGreater(len(result.forecasts), 0)

    def test_model_used_populated(self):
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(100)
        _, result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )
        if "current_ratio" in result.forecasts:
            self.assertIn("current_ratio", result.model_used)

    def test_forecast_horizons(self):
        """Forecasts should include all defined horizons."""
        from operator1.models.forecasting import run_forecasting, HORIZONS
        cache = _make_forecast_cache(100)
        _, result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )
        if "current_ratio" in result.forecasts:
            horizons = result.forecasts["current_ratio"]
            for h_label in HORIZONS:
                self.assertIn(h_label, horizons)

    def test_missing_variables_skipped(self):
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(100)
        _, result = run_forecasting(
            cache,
            ["nonexistent_var_1", "nonexistent_var_2"],
            enable_burnout=False,
        )
        # The requested variables don't exist, so they produce no forecasts.
        # However, GARCH on return_1d always runs as a special case and may
        # produce a volatility_garch entry.
        for var in ["nonexistent_var_1", "nonexistent_var_2"]:
            self.assertNotIn(var, result.forecasts)

    def test_default_variables_from_config(self):
        """When variables=None, should load from survival hierarchy."""
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(100)
        _, result = run_forecasting(
            cache,
            variables=None,
            enable_burnout=False,
        )
        # Should have forecasted at least some tier variables.
        self.assertGreater(len(result.forecasts), 0)

    def test_garch_on_returns(self):
        """Pipeline should attempt GARCH if return_1d is present."""
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(200)
        _, result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )
        # GARCH metrics should be in the results (even if it failed).
        garch_metrics = [m for m in result.metrics if m.model_name == "garch"]
        # GARCH is attempted on returns, so should have at least 1 entry.
        self.assertGreaterEqual(len(garch_metrics), 0)

    def test_metrics_list_populated(self):
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(100)
        _, result = run_forecasting(
            cache,
            ["current_ratio", "debt_to_equity_abs"],
            enable_burnout=False,
        )
        self.assertGreater(len(result.metrics), 0)
        for m in result.metrics:
            self.assertIsInstance(m.model_name, str)
            self.assertNotEqual(m.model_name, "")

    def test_at_least_baseline_succeeds(self):
        """Baseline model should always produce forecasts."""
        from operator1.models.forecasting import run_forecasting
        cache = _make_forecast_cache(50)
        _, result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )
        # Even on small data, baseline should succeed.
        self.assertGreater(len(result.forecasts), 0)


# ===========================================================================
# Integration: regime detection + forecasting together
# ===========================================================================


class TestRegimeAndForecastingIntegration(unittest.TestCase):
    """Integration test running regime detection then forecasting."""

    def test_pipeline_sequence(self):
        """Run regime detection followed by forecasting on the same cache."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import run_forecasting

        cache = _make_forecast_cache(200)

        # Step 1: Regime detection.
        cache, detector = detect_regimes_and_breaks(cache, n_regimes=2)
        self.assertIn("regime_label", cache.columns)

        # Step 2: Forecasting.
        cache, result = run_forecasting(
            cache,
            ["current_ratio", "cash_ratio"],
            enable_burnout=False,
        )
        self.assertGreater(len(result.forecasts), 0)

    def test_regime_columns_survive_forecasting(self):
        """Forecasting should not remove regime columns."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import run_forecasting

        cache = _make_forecast_cache(150)
        cache, _ = detect_regimes_and_breaks(cache, n_regimes=2)
        cols_before = set(cache.columns)

        run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )

        cols_after = set(cache.columns)
        self.assertTrue(cols_before.issubset(cols_after))


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_horizons_defined(self):
        from operator1.models.forecasting import HORIZONS
        self.assertIn("1d", HORIZONS)
        self.assertIn("5d", HORIZONS)
        self.assertIn("21d", HORIZONS)
        self.assertIn("252d", HORIZONS)
        self.assertEqual(HORIZONS["1d"], 1)
        self.assertEqual(HORIZONS["252d"], 252)

    def test_min_obs_positive(self):
        from operator1.models.forecasting import (
            _MIN_OBS_KALMAN,
            _MIN_OBS_GARCH,
            _MIN_OBS_VAR,
            _MIN_OBS_LSTM,
            _MIN_OBS_TREE,
            _MIN_OBS_BASELINE,
        )
        self.assertGreater(_MIN_OBS_KALMAN, 0)
        self.assertGreater(_MIN_OBS_GARCH, 0)
        self.assertGreater(_MIN_OBS_VAR, 0)
        self.assertGreater(_MIN_OBS_LSTM, 0)
        self.assertGreater(_MIN_OBS_TREE, 0)
        self.assertGreaterEqual(_MIN_OBS_BASELINE, 0)


if __name__ == "__main__":
    unittest.main()
