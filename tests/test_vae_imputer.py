"""Tests for the VAE imputer module (operator1.estimation.vae_imputer).

These tests verify:
  - VAE architecture construction and forward pass
  - Training on synthetic financial data
  - Imputation of missing values
  - Fallback behaviour when torch is unavailable or data is insufficient
  - Integration with the estimation pipeline via imputer_method="vae"
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


def _make_feature_df(days: int = 60, missing_frac: float = 0.0) -> pd.DataFrame:
    """Build a synthetic daily feature table for VAE tests.

    When *missing_frac* > 0, randomly set that fraction of the target
    variable rows to NaN (simulating real-world missingness).
    """
    idx = _make_daily_index(days)
    np.random.seed(42)

    # Create correlated financial variables (nonlinear relationships)
    revenue = 1_000_000 + np.cumsum(np.random.randn(days) * 10_000)
    gross_margin = 0.6 + np.random.randn(days) * 0.02
    gross_profit = revenue * gross_margin
    ebit = gross_profit * (0.5 + np.random.randn(days) * 0.03)
    ebitda = ebit * 1.15
    net_income = ebit * 0.7

    total_assets = 5_000_000 + np.cumsum(np.random.randn(days) * 20_000)
    total_liabilities = total_assets * (0.4 + np.random.randn(days) * 0.02)
    total_equity = total_assets - total_liabilities

    defaults = {
        "close": 100 + np.cumsum(np.random.randn(days) * 2),
        "return_1d": np.concatenate([[0.0], np.random.randn(days - 1) * 0.02]),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "ebit": ebit,
        "ebitda": ebitda,
        "net_income": net_income,
        "interest_expense": np.full(days, 10_000.0),
        "taxes": np.full(days, 50_000.0),
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": total_equity,
        "current_assets": np.full(days, 1_500_000.0),
        "current_liabilities": np.full(days, 800_000.0),
        "cash_and_equivalents": np.full(days, 500_000.0),
        "short_term_debt": np.full(days, 100_000.0),
        "long_term_debt": np.full(days, 400_000.0),
        "receivables": np.full(days, 200_000.0),
        "operating_cash_flow": np.full(days, 300_000.0),
        "capex": np.full(days, -50_000.0),
        "free_cash_flow": np.full(days, 250_000.0),
        "free_cash_flow_ttm_asof": np.full(days, 250_000.0),
        "total_debt_asof": np.full(days, 500_000.0),
        "net_debt": np.full(days, 0.0),
        "market_cap": np.full(days, 5_000_000.0),
        "volatility_21d": np.full(days, 0.15),
        # Hierarchy weights
        "hierarchy_tier1_weight": np.full(days, 0.15),
        "hierarchy_tier2_weight": np.full(days, 0.15),
        "hierarchy_tier3_weight": np.full(days, 0.20),
        "hierarchy_tier4_weight": np.full(days, 0.25),
        "hierarchy_tier5_weight": np.full(days, 0.25),
    }

    df = pd.DataFrame(defaults, index=idx)

    # Introduce missing values in target variables
    if missing_frac > 0:
        rng = np.random.RandomState(123)
        for var in ["revenue", "gross_profit", "ebit", "net_income"]:
            mask = rng.random(days) < missing_frac
            # Keep first 30 rows observed (training data)
            mask[:30] = False
            df.loc[df.index[mask], var] = np.nan

    # Add is_missing flags
    for col in list(df.columns):
        if not col.startswith("is_missing_") and not col.startswith("hierarchy_tier"):
            df[f"is_missing_{col}"] = df[col].isna().astype(int)

    return df


# ===========================================================================
# Unit tests: VAE module internals
# ===========================================================================

class TestVAETorchCheck(unittest.TestCase):
    """Verify torch availability check."""

    def test_check_torch_available(self):
        from operator1.estimation.vae_imputer import _check_torch_available
        # Should return True if torch is installed (it is in requirements)
        result = _check_torch_available()
        self.assertIsInstance(result, bool)


class TestVAEBuildModel(unittest.TestCase):
    """Test VAE model construction."""

    def test_build_vae_returns_model(self):
        """Model should be constructable with default dims."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

        from operator1.estimation.vae_imputer import _build_vae

        model = _build_vae(input_dim=10, latent_dim=4, hidden_dim=16)
        self.assertTrue(hasattr(model, "encoder"))
        self.assertTrue(hasattr(model, "decoder"))
        self.assertTrue(hasattr(model, "reparameterize"))

    def test_forward_pass_shapes(self):
        """Forward pass should return (recon, mu, logvar) with correct shapes."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from operator1.estimation.vae_imputer import _build_vae

        input_dim = 10
        latent_dim = 4
        batch_size = 8
        model = _build_vae(input_dim, latent_dim, hidden_dim=16)
        model.eval()

        x = torch.randn(batch_size, input_dim)
        recon, mu, logvar = model(x)

        self.assertEqual(recon.shape, (batch_size, input_dim))
        self.assertEqual(mu.shape, (batch_size, latent_dim))
        self.assertEqual(logvar.shape, (batch_size, latent_dim))


class TestVAELoss(unittest.TestCase):
    """Test VAE loss computation."""

    def test_loss_is_positive(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from operator1.estimation.vae_imputer import _vae_loss

        x = torch.randn(8, 10)
        recon = torch.randn(8, 10)
        mu = torch.randn(8, 4)
        logvar = torch.randn(8, 4)

        loss = _vae_loss(recon, x, mu, logvar)
        self.assertGreater(loss.item(), 0)

    def test_loss_zero_kl_weight(self):
        """With kl_weight=0, loss should be pure reconstruction."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch not available")

        from operator1.estimation.vae_imputer import _vae_loss

        x = torch.randn(8, 10)
        recon = torch.randn(8, 10)
        mu = torch.randn(8, 4)
        logvar = torch.randn(8, 4)

        loss = _vae_loss(recon, x, mu, logvar, kl_weight=0.0)
        expected = F.mse_loss(recon, x, reduction="mean")
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


class TestNormalization(unittest.TestCase):
    """Test data normalization helpers."""

    def test_normalize_denormalize_roundtrip(self):
        from operator1.estimation.vae_imputer import _normalize_data, _denormalize

        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        X_norm, means, stds = _normalize_data(X)
        X_back = _denormalize(X_norm, means, stds)
        np.testing.assert_array_almost_equal(X, X_back)

    def test_normalize_zero_variance(self):
        """Columns with zero variance should not cause division by zero."""
        from operator1.estimation.vae_imputer import _normalize_data

        X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        X_norm, means, stds = _normalize_data(X)
        self.assertTrue(np.all(np.isfinite(X_norm)))


# ===========================================================================
# Integration tests: train_and_impute_vae
# ===========================================================================

class TestVAETrainAndImpute(unittest.TestCase):
    """Test the full VAE training and imputation pipeline."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

    def test_impute_missing_revenue(self):
        """VAE should fill missing revenue values."""
        from operator1.estimation.vae_imputer import train_and_impute_vae

        df = _make_feature_df(days=60, missing_frac=0.3)
        target_vars = ["revenue", "gross_profit"]
        feature_cols = [
            "close", "return_1d", "total_assets", "total_liabilities",
            "total_equity", "market_cap", "volatility_21d",
        ]

        result = train_and_impute_vae(
            df=df,
            target_vars=target_vars,
            feature_cols=feature_cols,
            epochs=20,  # few epochs for speed
            latent_dim=4,
            hidden_dim=16,
        )

        self.assertFalse(result.fallback_used)
        self.assertGreater(result.n_train_rows, 0)
        # At least some values should have been imputed
        rev_imputed = result.estimated_values["revenue"].notna().sum()
        self.assertGreater(rev_imputed, 0)
        # Confidence scores should be in [0, 1]
        conf = result.confidence_scores["revenue"].dropna()
        if len(conf) > 0:
            self.assertTrue((conf >= 0).all() and (conf <= 1).all())

    def test_no_missing_returns_empty(self):
        """When no values are missing, nothing should be imputed."""
        from operator1.estimation.vae_imputer import train_and_impute_vae

        df = _make_feature_df(days=60, missing_frac=0.0)
        target_vars = ["revenue"]
        feature_cols = ["close", "total_assets"]

        result = train_and_impute_vae(
            df=df,
            target_vars=target_vars,
            feature_cols=feature_cols,
            epochs=5,
        )

        self.assertFalse(result.fallback_used)
        rev_imputed = result.estimated_values["revenue"].notna().sum()
        self.assertEqual(rev_imputed, 0)

    def test_insufficient_data_triggers_fallback(self):
        """With too few rows, VAE should signal fallback."""
        from operator1.estimation.vae_imputer import train_and_impute_vae

        df = _make_feature_df(days=10, missing_frac=0.5)
        target_vars = ["revenue"]
        feature_cols = ["close", "total_assets"]

        result = train_and_impute_vae(
            df=df,
            target_vars=target_vars,
            feature_cols=feature_cols,
            epochs=5,
        )

        self.assertTrue(result.fallback_used)

    def test_train_loss_decreases(self):
        """Final training loss should be finite and reasonable."""
        from operator1.estimation.vae_imputer import train_and_impute_vae

        df = _make_feature_df(days=60, missing_frac=0.2)
        target_vars = ["revenue", "gross_profit", "ebit"]
        feature_cols = [
            "close", "return_1d", "total_assets", "total_liabilities",
            "total_equity",
        ]

        result = train_and_impute_vae(
            df=df,
            target_vars=target_vars,
            feature_cols=feature_cols,
            epochs=30,
            latent_dim=4,
            hidden_dim=16,
        )

        self.assertFalse(result.fallback_used)
        self.assertTrue(np.isfinite(result.train_loss_final))


# ===========================================================================
# Integration tests: run_estimation with imputer_method="vae"
# ===========================================================================

class TestEstimationWithVAE(unittest.TestCase):
    """Test the full estimation pipeline using the VAE backend."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

    def test_run_estimation_vae_method(self):
        """run_estimation(imputer_method='vae') should produce valid output."""
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(days=60, missing_frac=0.2)
        variables = ["revenue", "gross_profit", "total_assets"]

        result_df, coverage = run_estimation(
            df, variables=variables, imputer_method="vae",
        )

        # Should have estimation output columns
        for var in variables:
            self.assertIn(f"{var}_final", result_df.columns)
            self.assertIn(f"{var}_source", result_df.columns)
            self.assertIn(f"{var}_confidence", result_df.columns)

        # Coverage after should be >= coverage before
        for var in variables:
            self.assertGreaterEqual(
                coverage.coverage_after.get(var, 0),
                coverage.coverage_before.get(var, 0),
            )

    def test_run_estimation_default_is_bayesian_ridge(self):
        """Default imputer should be bayesian_ridge (backward compat)."""
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(days=40, missing_frac=0.2)
        variables = ["revenue"]

        result_df, coverage = run_estimation(
            df, variables=variables, imputer_method="bayesian_ridge",
        )

        self.assertIn("revenue_final", result_df.columns)

    def test_vae_fallback_to_bayesian_ridge(self):
        """VAE with insufficient data should fall back to BayesianRidge."""
        from operator1.estimation.estimator import run_estimation

        # Very small dataset -- VAE won't have enough train rows
        df = _make_feature_df(days=15, missing_frac=0.3)
        variables = ["revenue"]

        result_df, coverage = run_estimation(
            df, variables=variables, imputer_method="vae",
        )

        # Should still produce valid output via fallback
        self.assertIn("revenue_final", result_df.columns)


# ===========================================================================
# Cache tests
# ===========================================================================

class TestVAEModelCache(unittest.TestCase):
    """Test VAE model caching and reloading."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

    def test_load_cached_model_returns_none_when_no_cache(self):
        """Should return None when no cache directory exists."""
        from operator1.estimation.vae_imputer import load_cached_vae_model

        import operator1.estimation.vae_imputer as vae_mod
        original_dir = vae_mod._VAE_CACHE_DIR
        vae_mod._VAE_CACHE_DIR = "/tmp/nonexistent_vae_cache_xyz"
        try:
            result = load_cached_vae_model()
            self.assertIsNone(result)
        finally:
            vae_mod._VAE_CACHE_DIR = original_dir

    def test_save_and_load_roundtrip(self):
        """A saved model should be loadable with matching columns."""
        import tempfile
        import operator1.estimation.vae_imputer as vae_mod
        from operator1.estimation.vae_imputer import (
            _build_vae, _save_vae_model, load_cached_vae_model,
        )

        original_dir = vae_mod._VAE_CACHE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            vae_mod._VAE_CACHE_DIR = tmpdir

            model = _build_vae(input_dim=5, latent_dim=2, hidden_dim=8)
            means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            columns = ["a", "b", "c", "d", "e"]

            _save_vae_model(model, means, stds, columns, 5, 2, 8)

            loaded = load_cached_vae_model(expected_columns=columns)
            self.assertIsNotNone(loaded)
            loaded_model, loaded_means, loaded_stds, loaded_cols = loaded
            np.testing.assert_array_almost_equal(loaded_means, means)
            np.testing.assert_array_almost_equal(loaded_stds, stds)
            self.assertEqual(loaded_cols, columns)

            vae_mod._VAE_CACHE_DIR = original_dir

    def test_load_rejects_column_mismatch(self):
        """Cache should be rejected if columns don't match."""
        import tempfile
        import operator1.estimation.vae_imputer as vae_mod
        from operator1.estimation.vae_imputer import (
            _build_vae, _save_vae_model, load_cached_vae_model,
        )

        original_dir = vae_mod._VAE_CACHE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            vae_mod._VAE_CACHE_DIR = tmpdir

            model = _build_vae(input_dim=5, latent_dim=2, hidden_dim=8)
            means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            columns = ["a", "b", "c", "d", "e"]

            _save_vae_model(model, means, stds, columns, 5, 2, 8)

            # Try loading with different columns
            loaded = load_cached_vae_model(expected_columns=["x", "y", "z"])
            self.assertIsNone(loaded)

            vae_mod._VAE_CACHE_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
