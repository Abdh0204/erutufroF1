"""Tests for the 4 advanced modules: PID Controller, Fuzzy Logic,
Graph Theory, and Game Theory."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 20) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_cache(days: int = 20) -> pd.DataFrame:
    idx = _make_daily_index(days)
    np.random.seed(42)
    return pd.DataFrame({
        "close": 100 + np.cumsum(np.random.randn(days) * 2),
        "market_cap": np.full(days, 5_000_000.0),
        "revenue": np.full(days, 1_000_000.0),
        "operating_margin": np.full(days, 0.25),
        "lending_interest_rate": np.linspace(5.0, 4.0, days),
    }, index=idx)


# ===========================================================================
# PID Controller tests
# ===========================================================================


class TestPIDController(unittest.TestCase):

    def test_initial_output_is_one(self):
        from operator1.models.pid_controller import PIDController
        pid = PIDController(variable="test")
        # No error fed yet, first update with 0 error should return ~1.0
        out = pid.update(0.0)
        self.assertAlmostEqual(out, 1.0, places=1)

    def test_large_error_increases_multiplier(self):
        from operator1.models.pid_controller import PIDController
        pid = PIDController(variable="test")
        out = pid.update(10.0)  # large error
        self.assertGreater(out, 1.5)

    def test_zero_error_stays_near_one(self):
        from operator1.models.pid_controller import PIDController
        pid = PIDController(variable="test")
        for _ in range(10):
            out = pid.update(0.0)
        self.assertAlmostEqual(out, 1.0, places=1)

    def test_output_is_clamped(self):
        from operator1.models.pid_controller import PIDController, MIN_MULTIPLIER, MAX_MULTIPLIER
        pid = PIDController(variable="test")
        # Feed massive errors
        for _ in range(100):
            out = pid.update(1000.0)
        self.assertLessEqual(out, MAX_MULTIPLIER)
        self.assertGreaterEqual(out, MIN_MULTIPLIER)

    def test_nan_error_returns_one(self):
        from operator1.models.pid_controller import PIDController
        pid = PIDController(variable="test")
        out = pid.update(float("nan"))
        self.assertEqual(out, 1.0)

    def test_reset(self):
        from operator1.models.pid_controller import PIDController
        pid = PIDController(variable="test")
        pid.update(5.0)
        pid.update(5.0)
        pid.reset()
        self.assertEqual(pid._n_updates, 0)
        self.assertEqual(pid._integral, 0.0)


class TestPIDBank(unittest.TestCase):

    def test_create_bank(self):
        from operator1.models.pid_controller import create_pid_bank
        bank = create_pid_bank(["close", "revenue", "debt"])
        self.assertEqual(len(bank), 3)
        self.assertIn("close", bank)

    def test_compute_adjustment(self):
        from operator1.models.pid_controller import create_pid_bank, compute_pid_adjustment
        bank = create_pid_bank(["close", "revenue"])
        result = compute_pid_adjustment(bank, {"close": 2.0, "revenue": 0.5})
        self.assertEqual(result.n_variables, 2)
        self.assertGreater(result.mean_multiplier, 1.0)

    def test_result_to_dict(self):
        from operator1.models.pid_controller import create_pid_bank, compute_pid_adjustment
        bank = create_pid_bank(["close"])
        result = compute_pid_adjustment(bank, {"close": 1.0})
        d = result.to_dict()
        self.assertIn("mean_multiplier", d)
        self.assertIn("per_variable", d)


# ===========================================================================
# Fuzzy Logic tests
# ===========================================================================


class TestFuzzyProtection(unittest.TestCase):

    def test_defense_sector_high_score(self):
        from operator1.analysis.fuzzy_protection import _sector_membership
        self.assertGreaterEqual(_sector_membership("defense"), 0.9)

    def test_retail_sector_low_score(self):
        from operator1.analysis.fuzzy_protection import _sector_membership
        self.assertLessEqual(_sector_membership("retail"), 0.2)

    def test_unknown_sector(self):
        from operator1.analysis.fuzzy_protection import _sector_membership
        score = _sector_membership("underwater basket weaving")
        self.assertEqual(score, 0.1)

    def test_economic_significance_large_company(self):
        from operator1.analysis.fuzzy_protection import _economic_significance
        score = _economic_significance(50_000_000, 1_000_000_000)  # 5% of GDP
        self.assertGreaterEqual(score, 0.9)

    def test_economic_significance_tiny_company(self):
        from operator1.analysis.fuzzy_protection import _economic_significance
        score = _economic_significance(100, 1_000_000_000)
        self.assertLessEqual(score, 0.1)

    def test_economic_significance_no_gdp(self):
        from operator1.analysis.fuzzy_protection import _economic_significance
        self.assertEqual(_economic_significance(1000, None), 0.0)

    def test_compute_fuzzy_protection_produces_columns(self):
        from operator1.analysis.fuzzy_protection import compute_fuzzy_protection
        cache = _make_cache(20)
        result = compute_fuzzy_protection(cache, sector="banking", gdp=1e10)
        self.assertIn("fuzzy_sector_score", result.columns)
        self.assertIn("fuzzy_economic_score", result.columns)
        self.assertIn("fuzzy_protection_degree", result.columns)
        self.assertIn("fuzzy_protection_label", result.columns)
        self.assertIn("country_protected_flag", result.columns)

    def test_banking_sector_gets_protected(self):
        from operator1.analysis.fuzzy_protection import compute_fuzzy_protection
        cache = _make_cache(20)
        result = compute_fuzzy_protection(cache, sector="banking", gdp=1e10)
        # Banking sector score should be >= 0.5, so flag should be 1
        self.assertEqual(result["country_protected_flag"].iloc[0], 1)

    def test_retail_sector_not_protected(self):
        from operator1.analysis.fuzzy_protection import compute_fuzzy_protection
        cache = _make_cache(20)
        cache["market_cap"] = 100.0  # tiny company
        result = compute_fuzzy_protection(cache, sector="retail", gdp=1e15)
        self.assertEqual(result["country_protected_flag"].iloc[0], 0)


# ===========================================================================
# Graph Theory tests
# ===========================================================================


class TestGraphRisk(unittest.TestCase):

    def _sample_relationships(self) -> dict:
        return {
            "competitors": [
                {"isin": "COMP1", "name": "Competitor A", "market_cap": 1e9},
                {"isin": "COMP2", "name": "Competitor B", "market_cap": 2e9},
                {"isin": "COMP3", "name": "Competitor C", "market_cap": 3e9},
            ],
            "suppliers": [
                {"isin": "SUP1", "name": "Supplier X", "market_cap": 5e8},
                {"isin": "SUP2", "name": "Supplier Y", "market_cap": 8e8},
            ],
            "customers": [
                {"isin": "CUST1", "name": "Customer Z", "market_cap": 4e9},
            ],
        }

    def test_build_graph(self):
        from operator1.models.graph_risk import build_entity_graph
        nodes, adj = build_entity_graph("TARGET", self._sample_relationships())
        self.assertEqual(len(nodes), 7)  # 1 target + 6 linked
        self.assertTrue(nodes[0].is_target)

    def test_compute_metrics(self):
        from operator1.models.graph_risk import compute_graph_risk_metrics
        result = compute_graph_risk_metrics("TARGET", self._sample_relationships())
        self.assertTrue(result.available)
        self.assertEqual(result.n_nodes, 7)
        self.assertGreater(result.target_degree_centrality, 0)
        self.assertGreater(result.target_pagerank, 0)

    def test_contagion_nonzero(self):
        from operator1.models.graph_risk import compute_graph_risk_metrics
        result = compute_graph_risk_metrics(
            "TARGET", self._sample_relationships(), contagion_sims=100,
        )
        self.assertGreater(result.contagion_expected_infected, 0)

    def test_empty_relationships(self):
        from operator1.models.graph_risk import compute_graph_risk_metrics
        result = compute_graph_risk_metrics("TARGET", {})
        self.assertTrue(result.available)
        self.assertEqual(result.n_nodes, 1)

    def test_hhi_computation(self):
        from operator1.models.graph_risk import _hhi
        # Equal shares: HHI = 1/n
        self.assertAlmostEqual(_hhi([1, 1, 1, 1]), 0.25, places=2)
        # Monopoly: HHI = 1.0
        self.assertAlmostEqual(_hhi([100]), 1.0, places=2)
        # Empty
        self.assertEqual(_hhi([]), 0.0)

    def test_result_to_dict(self):
        from operator1.models.graph_risk import compute_graph_risk_metrics
        result = compute_graph_risk_metrics("TARGET", self._sample_relationships())
        d = result.to_dict()
        self.assertIn("target_pagerank", d)
        self.assertIn("concentration_label", d)


# ===========================================================================
# Game Theory tests
# ===========================================================================


class TestGameTheory(unittest.TestCase):

    def _make_target_cache(self) -> pd.DataFrame:
        idx = _make_daily_index(20)
        return pd.DataFrame({
            "close": 150.0,
            "market_cap": 2e9,
            "revenue": 1e8,
            "operating_margin": 0.20,
            "name": "TargetCo",
        }, index=idx)

    def _make_competitor_caches(self) -> dict[str, pd.DataFrame]:
        idx = _make_daily_index(20)
        return {
            "COMP1": pd.DataFrame({
                "close": 80.0, "market_cap": 1e9,
                "revenue": 5e7, "operating_margin": 0.15,
                "name": "CompA",
            }, index=idx),
            "COMP2": pd.DataFrame({
                "close": 120.0, "market_cap": 3e9,
                "revenue": 1.5e8, "operating_margin": 0.22,
                "name": "CompB",
            }, index=idx),
        }

    def test_analyze_returns_result(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
            target_name="TargetCo",
        )
        self.assertTrue(result.available)
        self.assertEqual(result.n_competitors, 2)

    def test_cournot_produces_shares(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
        )
        self.assertGreater(len(result.cournot.equilibrium_shares), 0)
        total = sum(result.cournot.equilibrium_shares.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_stackelberg_role(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
        )
        self.assertIn(result.stackelberg.target_role, ("leader", "follower"))
        self.assertGreaterEqual(result.stackelberg.leadership_score, 0)
        self.assertLessEqual(result.stackelberg.leadership_score, 1)

    def test_competitive_pressure_bounded(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
        )
        self.assertGreaterEqual(result.competitive_pressure, 0)
        self.assertLessEqual(result.competitive_pressure, 1)

    def test_no_competitors(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(self._make_target_cache())
        self.assertEqual(result.market_structure, "monopoly")
        self.assertEqual(result.competitive_pressure, 0.0)

    def test_market_structure_classification(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
        )
        self.assertIn(
            result.market_structure,
            ("monopoly", "tight_oligopoly", "loose_oligopoly", "competitive"),
        )

    def test_result_to_dict(self):
        from operator1.models.game_theory import analyze_competitive_dynamics
        result = analyze_competitive_dynamics(
            self._make_target_cache(),
            self._make_competitor_caches(),
        )
        d = result.to_dict()
        self.assertIn("competitive_pressure", d)
        self.assertIn("cournot", d)
        self.assertIn("stackelberg", d)
        self.assertIn("market_structure", d)


if __name__ == "__main__":
    unittest.main()
