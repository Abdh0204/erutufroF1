"""Phase 6 tests -- T6.4 prediction aggregation.

Tests the prediction aggregation pipeline:
  - Ensemble weight computation (inverse-RMSE)
  - Forecast aggregation
  - Uncertainty band computation
  - Confidence scoring
  - Technical Alpha masking
  - Persistence (parquet + JSON)
  - Full pipeline (run_prediction_aggregation)
  - Integration with T6.1, T6.2, T6.3
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_cache(days: int = 250, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache for prediction aggregation tests."""
    np.random.seed(seed)
    idx = _make_daily_index(days)
    mid = days // 2

    returns = np.concatenate([
        np.random.randn(mid) * 0.005 + 0.001,
        np.random.randn(days - mid) * 0.03 - 0.002,
    ])

    close = 100 + np.cumsum(returns * 100)
    vol = np.concatenate([
        np.full(mid, 0.08) + np.random.randn(mid) * 0.01,
        np.full(days - mid, 0.25) + np.random.randn(days - mid) * 0.02,
    ])

    regime_labels = np.concatenate([
        np.full(mid, "bull"),
        np.full(days - mid, "bear"),
    ])

    return pd.DataFrame({
        "close": close,
        "open": close + np.random.randn(days) * 0.5,
        "high": close + np.abs(np.random.randn(days)) * 1.0,
        "low": close - np.abs(np.random.randn(days)) * 1.0,
        "return_1d": returns,
        "volatility_21d": vol,
        "regime_label": regime_labels,
        "current_ratio": 1.5 + np.cumsum(np.random.randn(days) * 0.005),
        "debt_to_equity_abs": 1.0 + np.cumsum(np.random.randn(days) * 0.003),
        "fcf_yield": 0.05 + np.cumsum(np.random.randn(days) * 0.001),
        "drawdown_252d": np.minimum(
            np.cumsum(np.random.randn(days) * 0.005), 0.0,
        ) - 0.10,
    }, index=idx)


def _make_forecast_result(
    variables: list[str] | None = None,
    *,
    include_failures: bool = False,
) -> "ForecastResult":
    """Build a synthetic ForecastResult for testing."""
    from operator1.models.forecasting import ForecastResult, ModelMetrics

    if variables is None:
        variables = ["current_ratio", "debt_to_equity_abs"]

    result = ForecastResult()

    for var in variables:
        result.forecasts[var] = {
            "1d": 1.5 + np.random.randn() * 0.01,
            "5d": 1.5 + np.random.randn() * 0.02,
            "21d": 1.5 + np.random.randn() * 0.05,
            "252d": 1.5 + np.random.randn() * 0.10,
        }
        result.model_used[var] = "kalman"

        result.metrics.append(ModelMetrics(
            model_name="kalman",
            variable=var,
            mae=0.05,
            rmse=0.08,
            n_train=200,
            n_test=30,
            fitted=True,
        ))
        result.metrics.append(ModelMetrics(
            model_name="baseline_ema",
            variable=var,
            mae=0.10,
            rmse=0.15,
            n_train=200,
            n_test=30,
            fitted=True,
        ))

    if include_failures:
        result.model_failed_garch = True
        result.garch_error = "arch not installed"
        result.model_failed_lstm = True
        result.lstm_error = "PyTorch not installed"

    return result


def _make_mc_result() -> "MonteCarloResult":
    """Build a synthetic MonteCarloResult for testing."""
    from operator1.models.monte_carlo import MonteCarloResult

    mc = MonteCarloResult(
        n_paths=1000,
        n_paths_importance=300,
        survival_probability={
            "1d": 0.98,
            "5d": 0.95,
            "21d": 0.88,
            "252d": 0.72,
        },
        survival_probability_mean=0.88,
        survival_probability_p5=0.70,
        survival_probability_p95=0.96,
        survival_stats={
            "1d": {"mean": 0.98, "std": 0.01, "p5": 0.96, "p95": 0.99,
                   "p25": 0.97, "median": 0.98, "p75": 0.99},
            "5d": {"mean": 0.95, "std": 0.02, "p5": 0.91, "p95": 0.98,
                   "p25": 0.94, "median": 0.95, "p75": 0.96},
            "21d": {"mean": 0.88, "std": 0.04, "p5": 0.80, "p95": 0.94,
                    "p25": 0.85, "median": 0.88, "p75": 0.91},
            "252d": {"mean": 0.72, "std": 0.08, "p5": 0.58, "p95": 0.85,
                     "p25": 0.66, "median": 0.72, "p75": 0.78},
        },
        current_regime="bear",
        importance_sampling_used=True,
        effective_sample_size=850.0,
        fitted=True,
    )
    return mc


# ===========================================================================
# Ensemble weight computation
# ===========================================================================


class TestComputeEnsembleWeights(unittest.TestCase):
    """Test inverse-RMSE ensemble weight computation."""

    def test_single_model(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [ModelMetrics(
            model_name="kalman", rmse=0.05, fitted=True,
        )]
        weights = compute_ensemble_weights(metrics)

        self.assertIn("kalman", weights)
        self.assertAlmostEqual(weights["kalman"], 1.0)

    def test_two_models_inverse_rmse(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="kalman", rmse=0.05, fitted=True),
            ModelMetrics(model_name="baseline_ema", rmse=0.10, fitted=True),
        ]
        weights = compute_ensemble_weights(metrics)

        # Kalman has half the RMSE -> double the inverse -> higher weight.
        self.assertGreater(weights["kalman"], weights["baseline_ema"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_nan_rmse_excluded(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="kalman", rmse=0.05, fitted=True),
            ModelMetrics(model_name="bad", rmse=float("nan"), fitted=True),
        ]
        weights = compute_ensemble_weights(metrics)

        self.assertIn("kalman", weights)
        self.assertNotIn("bad", weights)
        self.assertAlmostEqual(weights["kalman"], 1.0)

    def test_unfitted_excluded(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="kalman", rmse=0.05, fitted=True),
            ModelMetrics(model_name="lstm", rmse=0.03, fitted=False),
        ]
        weights = compute_ensemble_weights(metrics)

        self.assertNotIn("lstm", weights)
        self.assertAlmostEqual(weights["kalman"], 1.0)

    def test_all_nan_returns_empty(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="a", rmse=float("nan"), fitted=True),
            ModelMetrics(model_name="b", rmse=float("nan"), fitted=True),
        ]
        weights = compute_ensemble_weights(metrics)
        self.assertEqual(weights, {})

    def test_empty_metrics(self):
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        weights = compute_ensemble_weights([])
        self.assertEqual(weights, {})

    def test_weights_sum_to_one(self):
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="kalman", rmse=0.04, fitted=True),
            ModelMetrics(model_name="var(lag=3)", rmse=0.06, fitted=True),
            ModelMetrics(model_name="xgboost", rmse=0.08, fitted=True),
        ]
        weights = compute_ensemble_weights(metrics)

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)

    def test_best_rmse_per_model_name(self):
        """When a model has multiple metrics, use the best RMSE."""
        from operator1.models.forecasting import ModelMetrics
        from operator1.models.prediction_aggregator import compute_ensemble_weights

        metrics = [
            ModelMetrics(model_name="kalman", rmse=0.10, fitted=True),
            ModelMetrics(model_name="kalman", rmse=0.05, fitted=True),
        ]
        weights = compute_ensemble_weights(metrics)

        # Only one model name -> weight 1.0.
        self.assertAlmostEqual(weights["kalman"], 1.0)


# ===========================================================================
# Forecast aggregation
# ===========================================================================


class TestAggregateForecast(unittest.TestCase):
    """Test forecast extraction and aggregation."""

    def test_basic_aggregation(self):
        from operator1.models.prediction_aggregator import aggregate_forecasts

        fr = _make_forecast_result(["current_ratio"])
        agg = aggregate_forecasts(fr)

        self.assertIn("current_ratio", agg)
        self.assertIn("1d", agg["current_ratio"])
        self.assertIn("252d", agg["current_ratio"])

    def test_nan_values_excluded(self):
        from operator1.models.forecasting import ForecastResult
        from operator1.models.prediction_aggregator import aggregate_forecasts

        fr = ForecastResult()
        fr.forecasts["x"] = {"1d": 1.5, "5d": float("nan")}

        agg = aggregate_forecasts(fr)

        self.assertIn("1d", agg["x"])
        self.assertNotIn("5d", agg["x"])

    def test_empty_forecast_result(self):
        from operator1.models.forecasting import ForecastResult
        from operator1.models.prediction_aggregator import aggregate_forecasts

        fr = ForecastResult()
        agg = aggregate_forecasts(fr)

        self.assertEqual(agg, {})

    def test_all_nan_variable_excluded(self):
        from operator1.models.forecasting import ForecastResult
        from operator1.models.prediction_aggregator import aggregate_forecasts

        fr = ForecastResult()
        fr.forecasts["x"] = {"1d": float("nan"), "5d": float("nan")}

        agg = aggregate_forecasts(fr)

        self.assertNotIn("x", agg)


# ===========================================================================
# Uncertainty bands
# ===========================================================================


class TestComputeUncertaintyBands(unittest.TestCase):
    """Test confidence interval computation."""

    def test_wider_at_longer_horizons(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        _, upper_1d = compute_uncertainty_bands(100.0, 0.5, 1)
        _, upper_21d = compute_uncertainty_bands(100.0, 0.5, 21)

        spread_1d = upper_1d - 100.0
        spread_21d = upper_21d - 100.0

        self.assertGreater(spread_21d, spread_1d)

    def test_symmetric_around_forecast(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        point = 50.0
        lower, upper = compute_uncertainty_bands(point, 1.0, 1)

        # Should be symmetric when survival_probability = 1.0.
        spread_low = point - lower
        spread_high = upper - point

        self.assertAlmostEqual(spread_low, spread_high, places=8)

    def test_survival_widens_bands(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        _, upper_safe = compute_uncertainty_bands(
            100.0, 0.5, 21, survival_probability=1.0,
        )
        _, upper_risky = compute_uncertainty_bands(
            100.0, 0.5, 21, survival_probability=0.5,
        )

        spread_safe = upper_safe - 100.0
        spread_risky = upper_risky - 100.0

        self.assertGreater(spread_risky, spread_safe)

    def test_nan_point_returns_nan(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        lower, upper = compute_uncertainty_bands(float("nan"), 0.5, 1)

        self.assertTrue(math.isnan(lower))
        self.assertTrue(math.isnan(upper))

    def test_nan_rmse_uses_fallback(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        lower, upper = compute_uncertainty_bands(
            100.0, float("nan"), 1,
        )

        # Should still produce valid bounds using the 10% fallback.
        self.assertFalse(math.isnan(lower))
        self.assertFalse(math.isnan(upper))
        self.assertLess(lower, 100.0)
        self.assertGreater(upper, 100.0)

    def test_zero_rmse_uses_fallback(self):
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        lower, upper = compute_uncertainty_bands(100.0, 0.0, 1)

        self.assertFalse(math.isnan(lower))
        self.assertFalse(math.isnan(upper))

    def test_horizon_scaling_sqrt(self):
        """Spread should scale with sqrt of horizon_days."""
        from operator1.models.prediction_aggregator import compute_uncertainty_bands

        _, upper_1 = compute_uncertainty_bands(
            0.0, 1.0, 1, survival_probability=1.0,
        )
        _, upper_4 = compute_uncertainty_bands(
            0.0, 1.0, 4, survival_probability=1.0,
        )

        # sqrt(4) / sqrt(1) = 2.0, so spread at h=4 should be 2x h=1.
        self.assertAlmostEqual(upper_4 / upper_1, 2.0, places=8)


# ===========================================================================
# Confidence score
# ===========================================================================


class TestConfidenceScore(unittest.TestCase):
    """Test confidence score computation."""

    def test_high_quality_high_survival(self):
        from operator1.models.prediction_aggregator import compute_confidence_score

        score = compute_confidence_score(
            rmse=0.01, rmse_reference=0.10, survival_probability=0.99,
        )
        self.assertGreater(score, 0.8)

    def test_low_survival_lowers_confidence(self):
        from operator1.models.prediction_aggregator import compute_confidence_score

        score_safe = compute_confidence_score(
            rmse=0.05, rmse_reference=0.10, survival_probability=0.99,
        )
        score_risky = compute_confidence_score(
            rmse=0.05, rmse_reference=0.10, survival_probability=0.30,
        )
        self.assertGreater(score_safe, score_risky)

    def test_nan_rmse_default(self):
        from operator1.models.prediction_aggregator import compute_confidence_score

        score = compute_confidence_score(
            rmse=float("nan"), rmse_reference=0.10, survival_probability=1.0,
        )
        # Should be a moderate default value.
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_score_in_range(self):
        from operator1.models.prediction_aggregator import compute_confidence_score

        for rmse in [0.01, 0.1, 1.0, 10.0]:
            for surv in [0.0, 0.5, 1.0]:
                score = compute_confidence_score(rmse, 0.1, surv)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


# ===========================================================================
# Technical Alpha masking
# ===========================================================================


class TestTechnicalAlphaMask(unittest.TestCase):
    """Test Technical Alpha OHLC masking."""

    def test_open_high_close_masked(self):
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        cache = _make_cache(50)
        mask = apply_technical_alpha_mask(cache, {})

        self.assertIsNone(mask.next_day_open)
        self.assertIsNone(mask.next_day_high)
        self.assertIsNone(mask.next_day_close)
        self.assertTrue(mask.mask_applied)

    def test_low_is_visible(self):
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        cache = _make_cache(50)
        mask = apply_technical_alpha_mask(cache, {})

        self.assertFalse(math.isnan(mask.next_day_low))
        self.assertIsInstance(mask.next_day_low, float)

    def test_low_below_last_close(self):
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        cache = _make_cache(100)
        mask = apply_technical_alpha_mask(cache, {})

        last_close = float(cache["close"].dropna().iloc[-1])
        self.assertLess(mask.next_day_low, last_close)

    def test_no_close_column(self):
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({"other": np.random.randn(10)}, index=idx)

        mask = apply_technical_alpha_mask(cache, {})

        self.assertTrue(mask.mask_applied)
        self.assertTrue(math.isnan(mask.next_day_low))

    def test_all_nan_close(self):
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({
            "close": [np.nan] * 10,
        }, index=idx)

        mask = apply_technical_alpha_mask(cache, {})

        self.assertTrue(math.isnan(mask.next_day_low))

    def test_uses_volatility_from_cache(self):
        """When volatility is high, the estimated low should be further
        below last close."""
        from operator1.models.prediction_aggregator import apply_technical_alpha_mask

        idx = pd.bdate_range("2024-01-02", periods=50)
        np.random.seed(42)

        cache_low_vol = pd.DataFrame({
            "close": np.full(50, 100.0),
            "volatility_21d": np.full(50, 0.01),
        }, index=idx)

        cache_high_vol = pd.DataFrame({
            "close": np.full(50, 100.0),
            "volatility_21d": np.full(50, 0.10),
        }, index=idx)

        mask_low = apply_technical_alpha_mask(cache_low_vol, {})
        mask_high = apply_technical_alpha_mask(cache_high_vol, {})

        # High vol -> lower estimated low.
        self.assertLess(mask_high.next_day_low, mask_low.next_day_low)


# ===========================================================================
# Data class defaults
# ===========================================================================


class TestHorizonPrediction(unittest.TestCase):
    """Test HorizonPrediction dataclass."""

    def test_default_state(self):
        from operator1.models.prediction_aggregator import HorizonPrediction

        p = HorizonPrediction()
        self.assertEqual(p.variable, "")
        self.assertEqual(p.horizon, "")
        self.assertTrue(math.isnan(p.point_forecast))
        self.assertFalse(p.survival_adjusted)

    def test_populated(self):
        from operator1.models.prediction_aggregator import HorizonPrediction

        p = HorizonPrediction(
            variable="current_ratio",
            horizon="1d",
            point_forecast=1.5,
            lower_ci=1.4,
            upper_ci=1.6,
            confidence=0.9,
            model_used="kalman",
            ensemble_weight=0.7,
            survival_adjusted=True,
        )
        self.assertEqual(p.variable, "current_ratio")
        self.assertAlmostEqual(p.point_forecast, 1.5)
        self.assertTrue(p.survival_adjusted)


class TestPredictionAggregatorResult(unittest.TestCase):
    """Test PredictionAggregatorResult dataclass."""

    def test_default_state(self):
        from operator1.models.prediction_aggregator import PredictionAggregatorResult

        r = PredictionAggregatorResult()
        self.assertEqual(r.predictions, {})
        self.assertFalse(r.fitted)
        self.assertTrue(math.isnan(r.survival_probability_mean))
        self.assertIsNone(r.error)
        self.assertEqual(r.n_models_available, 0)

    def test_technical_alpha_default(self):
        from operator1.models.prediction_aggregator import PredictionAggregatorResult

        r = PredictionAggregatorResult()
        self.assertTrue(r.technical_alpha.mask_applied)


# ===========================================================================
# Persistence
# ===========================================================================


class TestSavePredictions(unittest.TestCase):
    """Test prediction persistence to parquet and JSON."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_creates_parquet_and_json(self):
        from operator1.models.prediction_aggregator import (
            PredictionAggregatorResult,
            HorizonPrediction,
            TechnicalAlphaMask,
            save_predictions,
        )

        result = PredictionAggregatorResult(
            predictions={
                "x": {
                    "1d": HorizonPrediction(
                        variable="x", horizon="1d",
                        point_forecast=1.5, lower_ci=1.4, upper_ci=1.6,
                        confidence=0.9, model_used="kalman",
                    ),
                },
            },
            technical_alpha=TechnicalAlphaMask(next_day_low=99.0),
            prediction_date="2024-01-15",
            fitted=True,
        )

        pq_path, json_path = save_predictions(result, self._tmpdir)

        self.assertTrue(pq_path.exists())
        self.assertTrue(json_path.exists())

    def test_parquet_has_correct_columns(self):
        from operator1.models.prediction_aggregator import (
            PredictionAggregatorResult,
            HorizonPrediction,
            save_predictions,
        )

        result = PredictionAggregatorResult(
            predictions={
                "x": {
                    "1d": HorizonPrediction(
                        variable="x", horizon="1d",
                        point_forecast=1.5,
                    ),
                },
            },
            fitted=True,
        )

        pq_path, _ = save_predictions(result, self._tmpdir)
        df = pd.read_parquet(pq_path)

        expected_cols = {
            "variable", "horizon", "point_forecast", "lower_ci",
            "upper_ci", "confidence", "model_used", "ensemble_weight",
            "survival_adjusted",
        }
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(len(df), 1)

    def test_json_contents(self):
        from operator1.models.prediction_aggregator import (
            PredictionAggregatorResult,
            TechnicalAlphaMask,
            save_predictions,
        )

        result = PredictionAggregatorResult(
            prediction_date="2024-01-15",
            current_regime="bear",
            ensemble_weights={"kalman": 0.7, "baseline": 0.3},
            survival_probability_mean=0.85,
            technical_alpha=TechnicalAlphaMask(next_day_low=98.5),
            fitted=True,
        )

        _, json_path = save_predictions(result, self._tmpdir)

        with open(json_path) as f:
            data = json.load(f)

        self.assertEqual(data["prediction_date"], "2024-01-15")
        self.assertEqual(data["current_regime"], "bear")
        self.assertAlmostEqual(data["survival_probability_mean"], 0.85)
        self.assertAlmostEqual(data["technical_alpha"]["next_day_low"], 98.5)
        self.assertTrue(data["fitted"])

    def test_empty_predictions_writes_empty_parquet(self):
        from operator1.models.prediction_aggregator import (
            PredictionAggregatorResult,
            save_predictions,
        )

        result = PredictionAggregatorResult(fitted=True)
        pq_path, _ = save_predictions(result, self._tmpdir)

        df = pd.read_parquet(pq_path)
        self.assertEqual(len(df), 0)

    def test_roundtrip_parquet(self):
        """Data written to parquet should survive a round-trip read."""
        from operator1.models.prediction_aggregator import (
            PredictionAggregatorResult,
            HorizonPrediction,
            save_predictions,
        )

        result = PredictionAggregatorResult(
            predictions={
                "var1": {
                    "1d": HorizonPrediction(
                        variable="var1", horizon="1d",
                        point_forecast=42.0, lower_ci=40.0,
                        upper_ci=44.0, confidence=0.85,
                        model_used="kalman", ensemble_weight=0.6,
                        survival_adjusted=True,
                    ),
                    "5d": HorizonPrediction(
                        variable="var1", horizon="5d",
                        point_forecast=43.0, lower_ci=38.0,
                        upper_ci=48.0, confidence=0.80,
                        model_used="kalman", ensemble_weight=0.6,
                        survival_adjusted=True,
                    ),
                },
            },
            fitted=True,
        )

        pq_path, _ = save_predictions(result, self._tmpdir)
        df = pd.read_parquet(pq_path)

        self.assertEqual(len(df), 2)
        row_1d = df[df["horizon"] == "1d"].iloc[0]
        self.assertAlmostEqual(row_1d["point_forecast"], 42.0)
        self.assertTrue(row_1d["survival_adjusted"])


# ===========================================================================
# Full pipeline: run_prediction_aggregation
# ===========================================================================


class TestRunPredictionAggregation(unittest.TestCase):
    """Test the top-level prediction aggregation pipeline."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_result(self):
        from operator1.models.prediction_aggregator import (
            run_prediction_aggregation,
            PredictionAggregatorResult,
        )

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertIsInstance(result, PredictionAggregatorResult)
        self.assertTrue(result.fitted)

    def test_predictions_populated(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio", "debt_to_equity_abs"])

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertGreater(len(result.predictions), 0)
        self.assertIn("current_ratio", result.predictions)

    def test_all_horizons_present(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation
        from operator1.models.forecasting import HORIZONS

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio"])

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        for h_label in HORIZONS:
            self.assertIn(
                h_label,
                result.predictions["current_ratio"],
                f"Missing horizon {h_label}",
            )

    def test_technical_alpha_applied(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertTrue(result.technical_alpha.mask_applied)
        self.assertIsNone(result.technical_alpha.next_day_open)
        self.assertIsNone(result.technical_alpha.next_day_high)
        self.assertIsNone(result.technical_alpha.next_day_close)
        self.assertFalse(math.isnan(result.technical_alpha.next_day_low))

    def test_ensemble_weights_populated(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertGreater(len(result.ensemble_weights), 0)
        self.assertAlmostEqual(sum(result.ensemble_weights.values()), 1.0)

    def test_model_counts(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(include_failures=True)

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(result.n_models_failed, 2)  # garch + lstm
        self.assertEqual(result.n_models_available, 3)

    def test_saves_parquet(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=True, cache_dir=self._tmpdir,
        )

        pq = Path(self._tmpdir) / "predictions.parquet"
        js = Path(self._tmpdir) / "prediction_summary.json"
        self.assertTrue(pq.exists())
        self.assertTrue(js.exists())

    def test_current_regime_detected(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(result.current_regime, "bear")

    def test_no_regime_column(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        idx = pd.bdate_range("2024-01-02", periods=50)
        cache = pd.DataFrame({
            "close": np.full(50, 100.0),
            "volatility_21d": np.full(50, 0.02),
        }, index=idx)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(result.current_regime, "unknown")

    def test_prediction_date_set(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation
        from datetime import date

        cache = _make_cache(50)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(result.prediction_date, date.today().isoformat())

    def test_variables_predicted_list(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(50)
        fr = _make_forecast_result(["current_ratio", "debt_to_equity_abs"])

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(
            sorted(result.variables_predicted),
            ["current_ratio", "debt_to_equity_abs"],
        )


# ===========================================================================
# With Monte Carlo
# ===========================================================================


class TestWithMonteCarlo(unittest.TestCase):
    """Test prediction aggregation with Monte Carlo results."""

    def test_survival_probs_passed_through(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()
        mc = _make_mc_result()

        result = run_prediction_aggregation(
            cache, fr, mc, save_to_cache=False,
        )

        self.assertAlmostEqual(result.survival_probability_mean, 0.88)
        self.assertAlmostEqual(result.survival_probability_p5, 0.70)
        self.assertAlmostEqual(result.survival_probability_p95, 0.96)

    def test_survival_adjusted_flag_set(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio"])
        mc = _make_mc_result()

        result = run_prediction_aggregation(
            cache, fr, mc, save_to_cache=False,
        )

        # MC survival probs are < 1.0 for all horizons.
        for h_label, pred in result.predictions["current_ratio"].items():
            self.assertTrue(
                pred.survival_adjusted,
                f"Expected survival_adjusted=True for {h_label}",
            )

    def test_bands_widened_with_low_survival(self):
        """With lower survival, uncertainty bands should be wider."""
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio"])

        # No MC -> no widening.
        result_no_mc = run_prediction_aggregation(
            cache, fr, mc_result=None, save_to_cache=False,
        )

        # With MC (survival < 1.0) -> wider bands.
        mc = _make_mc_result()
        result_mc = run_prediction_aggregation(
            cache, fr, mc, save_to_cache=False,
        )

        # Compare 252d horizon (where survival is lowest at 0.72).
        pred_no_mc = result_no_mc.predictions["current_ratio"]["252d"]
        pred_mc = result_mc.predictions["current_ratio"]["252d"]

        spread_no_mc = pred_no_mc.upper_ci - pred_no_mc.lower_ci
        spread_mc = pred_mc.upper_ci - pred_mc.lower_ci

        self.assertGreater(spread_mc, spread_no_mc)

    def test_confidence_lower_with_low_survival(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio"])

        result_no_mc = run_prediction_aggregation(
            cache, fr, mc_result=None, save_to_cache=False,
        )

        mc = _make_mc_result()
        result_mc = run_prediction_aggregation(
            cache, fr, mc, save_to_cache=False,
        )

        # 252d horizon has lowest survival (0.72).
        conf_no_mc = result_no_mc.predictions["current_ratio"]["252d"].confidence
        conf_mc = result_mc.predictions["current_ratio"]["252d"].confidence

        self.assertLessEqual(conf_mc, conf_no_mc)


# ===========================================================================
# Without Monte Carlo
# ===========================================================================


class TestWithoutMonteCarlo(unittest.TestCase):
    """Test graceful degradation without MC results."""

    def test_works_without_mc(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, mc_result=None, save_to_cache=False,
        )

        self.assertTrue(result.fitted)
        self.assertGreater(len(result.predictions), 0)

    def test_survival_probs_nan_without_mc(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result()

        result = run_prediction_aggregation(
            cache, fr, mc_result=None, save_to_cache=False,
        )

        self.assertTrue(math.isnan(result.survival_probability_mean))
        self.assertTrue(math.isnan(result.survival_probability_p5))

    def test_confidence_still_valid_without_mc(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(100)
        fr = _make_forecast_result(["current_ratio"])

        result = run_prediction_aggregation(
            cache, fr, mc_result=None, save_to_cache=False,
        )

        for pred in result.predictions["current_ratio"].values():
            self.assertFalse(math.isnan(pred.confidence))
            self.assertGreater(pred.confidence, 0.0)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_forecast_result(self):
        from operator1.models.forecasting import ForecastResult
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(50)
        fr = ForecastResult()

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(len(result.predictions), 0)
        self.assertIsNotNone(result.error)

    def test_single_variable(self):
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(50)
        fr = _make_forecast_result(["current_ratio"])

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertEqual(len(result.predictions), 1)
        self.assertIn("current_ratio", result.predictions)

    def test_minimal_cache(self):
        """Works with a minimal cache that has no regime/vol columns."""
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({"close": np.full(10, 100.0)}, index=idx)
        fr = _make_forecast_result(["current_ratio"])

        result = run_prediction_aggregation(
            cache, fr, save_to_cache=False,
        )

        self.assertTrue(result.fitted)
        self.assertEqual(result.current_regime, "unknown")

    def test_unfitted_mc_result_ignored(self):
        from operator1.models.monte_carlo import MonteCarloResult
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(50)
        fr = _make_forecast_result()
        mc = MonteCarloResult(fitted=False, error="no returns column")

        result = run_prediction_aggregation(
            cache, fr, mc, save_to_cache=False,
        )

        # Unfitted MC should be treated like None.
        self.assertTrue(math.isnan(result.survival_probability_mean))


# ===========================================================================
# Full Phase 6 integration
# ===========================================================================


class TestFullPhase6Integration(unittest.TestCase):
    """Integration test running T6.1 -> T6.2 -> T6.3 -> T6.4."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_full_pipeline(self):
        """Run all Phase 6 modules in sequence."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import run_forecasting
        from operator1.models.monte_carlo import run_monte_carlo
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(200)

        # T6.1: Regime detection.
        cache, detector = detect_regimes_and_breaks(cache, n_regimes=2)
        self.assertIn("regime_label", cache.columns)

        # T6.2: Forecasting.
        cache, forecast_result = run_forecasting(
            cache,
            ["current_ratio", "debt_to_equity_abs"],
            enable_burnout=False,
        )
        self.assertGreater(len(forecast_result.forecasts), 0)

        # T6.3: Monte Carlo.
        mc_result = run_monte_carlo(
            cache, n_paths=100, n_bootstrap=50,
        )
        self.assertTrue(mc_result.fitted)

        # T6.4: Prediction aggregation.
        result = run_prediction_aggregation(
            cache,
            forecast_result,
            mc_result,
            save_to_cache=True,
            cache_dir=self._tmpdir,
        )

        self.assertTrue(result.fitted)
        self.assertGreater(len(result.predictions), 0)
        self.assertTrue(result.technical_alpha.mask_applied)
        self.assertIsNone(result.technical_alpha.next_day_open)
        self.assertFalse(math.isnan(result.technical_alpha.next_day_low))

        # Verify persistence.
        pq = Path(self._tmpdir) / "predictions.parquet"
        js = Path(self._tmpdir) / "prediction_summary.json"
        self.assertTrue(pq.exists())
        self.assertTrue(js.exists())

        # Verify parquet content.
        df = pd.read_parquet(pq)
        self.assertGreater(len(df), 0)

    def test_pipeline_with_no_mc(self):
        """T6.4 should work even if T6.3 is skipped."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import run_forecasting
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        cache = _make_cache(150)

        cache, _ = detect_regimes_and_breaks(cache, n_regimes=2)
        cache, forecast_result = run_forecasting(
            cache,
            ["current_ratio"],
            enable_burnout=False,
        )

        result = run_prediction_aggregation(
            cache, forecast_result, mc_result=None, save_to_cache=False,
        )

        self.assertTrue(result.fitted)
        self.assertGreater(len(result.predictions), 0)


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_z_score(self):
        from operator1.models.prediction_aggregator import Z_SCORE_90
        self.assertAlmostEqual(Z_SCORE_90, 1.645)

    def test_intraday_low_factor(self):
        from operator1.models.prediction_aggregator import INTRADAY_LOW_FACTOR
        self.assertGreater(INTRADAY_LOW_FACTOR, 0)

    def test_default_volatility(self):
        from operator1.models.prediction_aggregator import DEFAULT_VOLATILITY
        self.assertGreater(DEFAULT_VOLATILITY, 0)


if __name__ == "__main__":
    unittest.main()
