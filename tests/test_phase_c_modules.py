"""Tests for Phase C -- missing math modules."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd


def _make_ohlc_cache(days: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range(start="2024-01-02", periods=days, name="date")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    return pd.DataFrame({
        "open": close + np.random.randn(days) * 0.5,
        "high": close + np.abs(np.random.randn(days) * 1.5),
        "low": close - np.abs(np.random.randn(days) * 1.5),
        "close": close,
        "return_1d": np.concatenate([[np.nan], np.diff(close) / close[:-1]]),
        "volatility_21d": np.full(days, 0.18),
        "cash_ratio": np.full(days, 0.8),
        "debt_to_equity": np.full(days, 1.2),
        "gross_margin": np.full(days, 0.4),
        "pe_ratio_calc": np.full(days, 22.0),
        "free_cash_flow_ttm_asof": np.full(days, 500_000),
        "drawdown_252d": np.full(days, -0.1),
    }, index=idx)


# ===================================================================
# C1: Candlestick Pattern Detector
# ===================================================================

class TestPatternDetector(unittest.TestCase):

    def test_detect_returns_result(self):
        from operator1.models.pattern_detector import detect_patterns
        cache = _make_ohlc_cache(100)
        result = detect_patterns(cache)
        self.assertTrue(result.available)
        self.assertIsInstance(result.recent_patterns, list)

    def test_missing_ohlc_returns_unavailable(self):
        from operator1.models.pattern_detector import detect_patterns
        cache = pd.DataFrame({"close": [100, 101, 102]})
        result = detect_patterns(cache)
        self.assertFalse(result.available)

    def test_to_dict(self):
        from operator1.models.pattern_detector import detect_patterns
        cache = _make_ohlc_cache(50)
        result = detect_patterns(cache)
        d = result.to_dict()
        self.assertIn("recent_patterns", d)
        self.assertIn("predicted_patterns_week", d)


# ===================================================================
# C2: Transformer Architecture
# ===================================================================

class TestTransformer(unittest.TestCase):

    def test_fit_transformer_runs(self):
        from operator1.models.forecasting import fit_transformer
        series = np.cumsum(np.random.randn(100)) + 100
        result = fit_transformer(series, n_forecast=1, lookback=10, epochs=5)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.metrics) > 0)


# ===================================================================
# C3: Particle Filter
# ===================================================================

class TestParticleFilter(unittest.TestCase):

    def test_fit_particle_filter_runs(self):
        from operator1.models.forecasting import fit_particle_filter
        series = np.cumsum(np.random.randn(80)) + 100
        result = fit_particle_filter(series, n_forecast=1, n_particles=100)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.metrics) > 0)

    def test_short_series(self):
        from operator1.models.forecasting import fit_particle_filter
        result = fit_particle_filter(np.array([100.0, 101.0, 102.0]))
        self.assertIsNotNone(result)


# ===================================================================
# C4: Genetic Algorithm ensemble optimisation
# ===================================================================

class TestGAOptimisation(unittest.TestCase):

    def test_ga_runs(self):
        from operator1.models.prediction_aggregator import (
            optimise_ensemble_weights_ga, ModelMetrics,
        )
        metrics = [
            ModelMetrics(model_name="kalman", fitted=True, rmse=0.05),
            ModelMetrics(model_name="lstm", fitted=True, rmse=0.08),
            ModelMetrics(model_name="var", fitted=True, rmse=0.06),
        ]
        weights = optimise_ensemble_weights_ga(metrics, generations=5)
        self.assertIsInstance(weights, dict)
        self.assertTrue(len(weights) >= 2)
        # Weights should sum to ~1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)


# ===================================================================
# C5: Transfer Entropy
# ===================================================================

class TestTransferEntropy(unittest.TestCase):

    def test_compute_transfer_entropy(self):
        from operator1.models.causality import compute_transfer_entropy
        cache = _make_ohlc_cache(100)
        variables = ["return_1d", "volatility_21d", "cash_ratio"]
        te = compute_transfer_entropy(cache, variables)
        self.assertEqual(te.shape[0], 3)
        self.assertEqual(te.shape[1], 3)
        # Diagonal should be 0
        for v in variables:
            self.assertEqual(te.loc[v, v], 0.0)

    def test_insufficient_data(self):
        from operator1.models.causality import compute_transfer_entropy
        cache = pd.DataFrame({"a": [1.0], "b": [2.0]})
        te = compute_transfer_entropy(cache, ["a", "b"])
        # Should return zero matrix (insufficient data)
        self.assertEqual(te.values.sum(), 0.0)


# ===================================================================
# C6: Sobol Sensitivity
# ===================================================================

class TestSobolSensitivity(unittest.TestCase):

    def test_sensitivity_runs(self):
        from operator1.models.sensitivity import run_sensitivity_analysis
        cache = _make_ohlc_cache(100)
        result = run_sensitivity_analysis(
            cache, target_variable="return_1d",
            feature_variables=["volatility_21d", "cash_ratio", "debt_to_equity"],
        )
        self.assertIsNotNone(result)
        # Should use permutation fallback if SALib not installed
        if result.available:
            self.assertTrue(len(result.first_order) > 0)


# ===================================================================
# C7: Copula Models
# ===================================================================

class TestCopulaModels(unittest.TestCase):

    def test_copula_runs(self):
        from operator1.models.copula import run_copula_analysis
        cache = _make_ohlc_cache(100)
        result = run_copula_analysis(
            cache, variables=["return_1d", "volatility_21d", "cash_ratio"],
        )
        self.assertTrue(result.available)
        self.assertGreater(len(result.copula_correlation), 0)
        self.assertGreater(len(result.tail_dependence), 0)

    def test_insufficient_vars(self):
        from operator1.models.copula import run_copula_analysis
        cache = pd.DataFrame({"x": np.random.randn(50)})
        result = run_copula_analysis(cache, variables=["x"])
        self.assertFalse(result.available)


# ===================================================================
# C8: Cycle Decomposition
# ===================================================================

class TestCycleDecomposition(unittest.TestCase):

    def test_fourier_runs(self):
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        cache = _make_ohlc_cache(200)
        result = run_cycle_decomposition(cache, variable="close")
        self.assertTrue(result.available)
        self.assertIsInstance(result.dominant_cycles, list)
        self.assertGreaterEqual(result.trend_strength, 0.0)
        self.assertGreaterEqual(result.noise_ratio, 0.0)

    def test_short_series(self):
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        cache = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = run_cycle_decomposition(cache, variable="close")
        self.assertFalse(result.available)

    def test_to_dict(self):
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        cache = _make_ohlc_cache(100)
        result = run_cycle_decomposition(cache, variable="close")
        d = result.to_dict()
        self.assertIn("dominant_cycles", d)
        self.assertIn("trend_strength", d)


if __name__ == "__main__":
    unittest.main()
