"""Tests for Phase B -- ethical filters and new features."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_daily_index(days: int = 20) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_cache(days: int = 20, **overrides) -> pd.DataFrame:
    idx = _make_daily_index(days)
    np.random.seed(42)
    defaults = {
        "close": 100 + np.cumsum(np.random.randn(days) * 2),
        "return_1d": np.concatenate([[np.nan], np.random.randn(days - 1) * 0.02]),
        "real_return_1d": np.concatenate([[np.nan], np.random.randn(days - 1) * 0.015]),
        "volatility_21d": np.full(days, 0.18),
        "debt_to_equity_abs": np.full(days, 1.2),
        "fcf_yield": np.full(days, 0.04),
        "fcf_margin": np.full(days, 0.12),
    }
    for k, v in overrides.items():
        defaults[k] = v
    return pd.DataFrame(defaults, index=idx)


# ===================================================================
# Purchasing Power filter
# ===================================================================

class TestPurchasingPowerFilter(unittest.TestCase):

    def test_pass_when_real_return_positive(self):
        from operator1.analysis.ethical_filters import compute_purchasing_power_filter
        cache = _make_cache(50, close=np.linspace(100, 120, 50))
        cache["real_return_1d"] = 0.001  # positive daily real return
        result = compute_purchasing_power_filter(cache)
        self.assertTrue(result["available"])
        self.assertIn("PASS", result["verdict"])

    def test_fail_when_nominal_up_real_down(self):
        from operator1.analysis.ethical_filters import compute_purchasing_power_filter
        cache = _make_cache(50, close=np.linspace(100, 105, 50))
        cache["real_return_1d"] = -0.002  # negative daily real return
        result = compute_purchasing_power_filter(cache)
        self.assertTrue(result["available"])
        self.assertIn("FAIL", result["verdict"])

    def test_unavailable_when_no_close(self):
        from operator1.analysis.ethical_filters import compute_purchasing_power_filter
        cache = _make_cache(5)
        cache.drop(columns=["close"], inplace=True)
        result = compute_purchasing_power_filter(cache)
        self.assertFalse(result["available"])


# ===================================================================
# Solvency filter
# ===================================================================

class TestSolvencyFilter(unittest.TestCase):

    def test_conservative(self):
        from operator1.analysis.ethical_filters import compute_solvency_filter
        cache = _make_cache(5, debt_to_equity_abs=np.full(5, 0.5))
        result = compute_solvency_filter(cache)
        self.assertIn("Conservative", result["verdict"])

    def test_stable(self):
        from operator1.analysis.ethical_filters import compute_solvency_filter
        cache = _make_cache(5, debt_to_equity_abs=np.full(5, 1.5))
        result = compute_solvency_filter(cache)
        self.assertIn("Stable", result["verdict"])

    def test_elevated(self):
        from operator1.analysis.ethical_filters import compute_solvency_filter
        cache = _make_cache(5, debt_to_equity_abs=np.full(5, 2.5))
        result = compute_solvency_filter(cache)
        self.assertIn("Elevated", result["verdict"])

    def test_fragile(self):
        from operator1.analysis.ethical_filters import compute_solvency_filter
        cache = _make_cache(5, debt_to_equity_abs=np.full(5, 4.0))
        result = compute_solvency_filter(cache)
        self.assertIn("Fragile", result["verdict"])


# ===================================================================
# Gharar filter
# ===================================================================

class TestGhararFilter(unittest.TestCase):

    def test_low_volatility(self):
        from operator1.analysis.ethical_filters import compute_gharar_filter
        cache = _make_cache(5, volatility_21d=np.full(5, 0.10))
        result = compute_gharar_filter(cache)
        self.assertEqual(result["stability_score"], 9)
        self.assertIn("LOW", result["verdict"])

    def test_moderate_volatility(self):
        from operator1.analysis.ethical_filters import compute_gharar_filter
        cache = _make_cache(5, volatility_21d=np.full(5, 0.20))
        result = compute_gharar_filter(cache)
        self.assertEqual(result["stability_score"], 7)

    def test_extreme_volatility(self):
        from operator1.analysis.ethical_filters import compute_gharar_filter
        cache = _make_cache(5, volatility_21d=np.full(5, 0.50))
        result = compute_gharar_filter(cache)
        self.assertEqual(result["stability_score"], 2)
        self.assertIn("EXTREME", result["verdict"])


# ===================================================================
# Cash is King filter
# ===================================================================

class TestCashIsKingFilter(unittest.TestCase):

    def test_strong(self):
        from operator1.analysis.ethical_filters import compute_cash_is_king_filter
        cache = _make_cache(5, fcf_yield=np.full(5, 0.08))
        result = compute_cash_is_king_filter(cache)
        self.assertIn("Strong", result["verdict"])

    def test_healthy(self):
        from operator1.analysis.ethical_filters import compute_cash_is_king_filter
        cache = _make_cache(5, fcf_yield=np.full(5, 0.03))
        result = compute_cash_is_king_filter(cache)
        self.assertIn("Healthy", result["verdict"])

    def test_burning_cash(self):
        from operator1.analysis.ethical_filters import compute_cash_is_king_filter
        cache = _make_cache(5, fcf_yield=np.full(5, -0.02))
        result = compute_cash_is_king_filter(cache)
        self.assertIn("Burning Cash", result["verdict"])


# ===================================================================
# Combined filter runner
# ===================================================================

class TestCombinedFilters(unittest.TestCase):

    def test_all_four_present(self):
        from operator1.analysis.ethical_filters import compute_all_ethical_filters
        cache = _make_cache(20)
        result = compute_all_ethical_filters(cache)
        self.assertIn("purchasing_power", result)
        self.assertIn("solvency", result)
        self.assertIn("gharar", result)
        self.assertIn("cash_is_king", result)

    def test_all_available(self):
        from operator1.analysis.ethical_filters import compute_all_ethical_filters
        cache = _make_cache(20)
        result = compute_all_ethical_filters(cache)
        for key in ("purchasing_power", "solvency", "gharar", "cash_is_king"):
            self.assertTrue(result[key]["available"], f"{key} should be available")


# ===================================================================
# Request log flush
# ===================================================================

class TestRequestLogFlush(unittest.TestCase):

    def test_flush_creates_file(self):
        import tempfile
        import os
        from operator1.http_utils import _request_log, flush_request_log

        # Add a fake entry
        _request_log.append({"url": "https://example.com", "status": 200})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "request_log.jsonl")
            flush_request_log(path)

            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("example.com", lines[0])

        # Log should be cleared after flush
        self.assertEqual(len(_request_log), 0)


if __name__ == "__main__":
    unittest.main()
