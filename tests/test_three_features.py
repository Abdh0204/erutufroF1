"""Tests for macro quadrant, peer ranking, and news sentiment features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_daily_index(n: int = 252) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n, freq="B")


def _make_cache(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = _make_daily_index(n)
    df = pd.DataFrame(index=idx)
    df["close"] = rng.uniform(100, 200, n).cumsum() / n + 100
    df["return_1d"] = df["close"].pct_change()
    df["volatility_21d"] = rng.uniform(0.01, 0.5, n)
    df["drawdown_252d"] = rng.uniform(-0.6, 0.0, n)
    df["volume"] = rng.uniform(1e5, 1e8, n)
    df["current_ratio"] = rng.uniform(0.5, 4.0, n)
    df["debt_to_equity_abs"] = rng.uniform(0.1, 5.0, n)
    df["free_cash_flow"] = rng.uniform(1e6, 1e9, n)
    df["fcf_yield"] = rng.uniform(0.01, 0.1, n)
    df["gross_margin"] = rng.uniform(0.1, 0.8, n)
    df["operating_margin"] = rng.uniform(0.05, 0.4, n)
    df["net_margin"] = rng.uniform(0.01, 0.3, n)
    df["market_cap"] = rng.uniform(1e9, 1e12, n)
    return df


# ===========================================================================
# Macro Quadrant Tests
# ===========================================================================


@dataclass
class MockMacroDataset:
    """Mimics MacroDataset from macro_mapping."""
    indicators: dict[str, pd.DataFrame] = field(default_factory=dict)


def _make_macro_data() -> MockMacroDataset:
    """Build mock macro data with GDP growth and inflation."""
    ds = MockMacroDataset()
    ds.indicators["gdp_growth"] = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024],
        "value": [5.7, 2.1, 2.5, 1.8],
    })
    ds.indicators["inflation_rate_yoy"] = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024],
        "value": [4.7, 8.0, 4.1, 2.9],
    })
    return ds


class TestMacroQuadrant:
    def test_columns_injected(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        macro = _make_macro_data()
        result_cache, result = compute_macro_quadrant(cache, macro)

        expected = [
            "macro_growth_vs_trend", "macro_inflation_vs_target",
            "macro_quadrant", "macro_quadrant_numeric",
            "macro_quadrant_stability_21d", "is_missing_macro_quadrant",
        ]
        for col in expected:
            assert col in result_cache.columns, f"Missing: {col}"

    def test_quadrant_labels_valid(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        macro = _make_macro_data()
        result_cache, _ = compute_macro_quadrant(cache, macro)

        valid_labels = {"goldilocks", "reflation", "stagflation", "deflation", "unknown"}
        unique = set(result_cache["macro_quadrant"].unique())
        assert unique.issubset(valid_labels)

    def test_result_summary(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        macro = _make_macro_data()
        _, result = compute_macro_quadrant(cache, macro)

        assert result.n_days_classified > 0
        assert result.current_quadrant != "unknown"
        assert len(result.quadrant_distribution) > 0
        assert result.n_transitions >= 0

    def test_no_macro_data(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        result_cache, result = compute_macro_quadrant(cache, None)

        assert "macro_quadrant" in result_cache.columns
        assert result.n_days_classified == 0

    def test_stability_range(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        macro = _make_macro_data()
        result_cache, _ = compute_macro_quadrant(cache, macro)

        valid = result_cache["macro_quadrant_stability_21d"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 100

    def test_custom_inflation_target(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant
        cache = _make_cache()
        macro = _make_macro_data()
        _, result_low = compute_macro_quadrant(cache.copy(), macro, inflation_target=1.0)
        _, result_high = compute_macro_quadrant(cache.copy(), macro, inflation_target=10.0)
        # Different targets should produce different distributions
        assert result_low.quadrant_distribution != result_high.quadrant_distribution


# ===========================================================================
# Peer Ranking Tests
# ===========================================================================


def _make_peer_caches(n_peers: int = 3, n_days: int = 100) -> dict[str, pd.DataFrame]:
    """Build mock linked entity caches."""
    idx = _make_daily_index(n_days)
    caches = {}
    for i in range(n_peers):
        rng = np.random.RandomState(100 + i)
        df = pd.DataFrame(index=idx)
        df["current_ratio"] = rng.uniform(0.5, 4.0, n_days)
        df["debt_to_equity_abs"] = rng.uniform(0.1, 5.0, n_days)
        df["gross_margin"] = rng.uniform(0.1, 0.8, n_days)
        df["operating_margin"] = rng.uniform(0.05, 0.4, n_days)
        df["net_margin"] = rng.uniform(0.01, 0.3, n_days)
        df["free_cash_flow"] = rng.uniform(1e6, 1e9, n_days)
        df["fcf_yield"] = rng.uniform(0.01, 0.1, n_days)
        df["market_cap"] = rng.uniform(1e9, 1e12, n_days)
        df["return_1d"] = rng.normal(0, 0.02, n_days)
        df["volatility_21d"] = rng.uniform(0.01, 0.5, n_days)
        df["drawdown_252d"] = rng.uniform(-0.6, 0.0, n_days)
        caches[f"PEER{i:04d}"] = df
    return caches


class TestPeerRanking:
    def test_columns_injected(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        peers = _make_peer_caches(3, 100)
        result_cache, result = compute_peer_ranking(cache, peers)

        assert "peer_rank_composite" in result_cache.columns
        assert "peer_rank_label" in result_cache.columns
        assert "is_missing_peer_rank" in result_cache.columns

    def test_percentile_range(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        peers = _make_peer_caches(5, 100)
        result_cache, _ = compute_peer_ranking(cache, peers)

        valid = result_cache["peer_rank_composite"].dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_result_summary(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        peers = _make_peer_caches(3, 100)
        _, result = compute_peer_ranking(cache, peers)

        assert result.n_peers == 3
        assert result.n_variables_ranked > 0
        assert not math.isnan(result.latest_composite_rank)
        assert result.latest_label in ("Laggard", "Below Average", "Average", "Above Average", "Leader")

    def test_no_peers(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        result_cache, result = compute_peer_ranking(cache, None)

        assert "peer_rank_composite" in result_cache.columns
        assert result_cache["peer_rank_composite"].isna().all()
        assert result.n_peers == 0

    def test_single_peer(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        peers = _make_peer_caches(1, 100)
        result_cache, result = compute_peer_ranking(cache, peers)

        assert result.n_peers == 1
        # With only 1 peer, target is either 50 or 100
        valid = result_cache["peer_rank_composite"].dropna()
        assert len(valid) > 0

    def test_zscore_columns_present(self):
        from operator1.features.peer_ranking import compute_peer_ranking
        cache = _make_cache(100)
        peers = _make_peer_caches(3, 100)
        result_cache, _ = compute_peer_ranking(cache, peers)

        # At least some zscore columns should be present
        zscore_cols = [c for c in result_cache.columns if c.startswith("peer_zscore_")]
        assert len(zscore_cols) > 0

    def test_inverted_ranking(self):
        """For debt_to_equity_abs (lower is better), a target with low value
        should rank higher than peers with high values."""
        from operator1.features.peer_ranking import compute_peer_ranking

        idx = _make_daily_index(50)
        # Target has very low debt
        cache = pd.DataFrame({"debt_to_equity_abs": [0.1] * 50}, index=idx)
        # Peers have very high debt
        peers = {
            "P1": pd.DataFrame({"debt_to_equity_abs": [5.0] * 50}, index=idx),
            "P2": pd.DataFrame({"debt_to_equity_abs": [4.0] * 50}, index=idx),
        }
        result_cache, _ = compute_peer_ranking(cache, peers)
        # Target should rank high (>= 50) since low debt is better
        valid = result_cache["peer_pctile_debt_to_equity_abs"].dropna()
        if len(valid) > 0:
            assert valid.iloc[-1] >= 50


# ===========================================================================
# News Sentiment Tests
# ===========================================================================


def _make_news_df(n_articles: int = 50, n_days: int = 100) -> pd.DataFrame:
    """Build mock news DataFrame."""
    rng = np.random.RandomState(77)
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")

    articles = []
    positive = ["record revenue", "strong growth", "beat estimates", "upgrade"]
    negative = ["decline in sales", "lawsuit filed", "downgrade", "loss reported"]
    neutral = ["quarterly results", "annual meeting", "board changes", "market update"]

    for i in range(n_articles):
        headline_pool = rng.choice(["pos", "neg", "neu"])
        if headline_pool == "pos":
            title = rng.choice(positive)
        elif headline_pool == "neg":
            title = rng.choice(negative)
        else:
            title = rng.choice(neutral)

        articles.append({
            "title": title,
            "text": f"Full text for article {i}",
            "publishedDate": rng.choice(dates),
            "site": "mocksite.com",
        })

    return pd.DataFrame(articles)


class TestNewsSentiment:
    def test_columns_injected(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        news = _make_news_df(30, 100)
        result_cache, result = compute_news_sentiment(cache, news_df=news)

        expected = [
            "sentiment_score", "sentiment_count",
            "sentiment_momentum_5d", "sentiment_momentum_21d",
            "sentiment_volatility_21d", "is_missing_sentiment",
        ]
        for col in expected:
            assert col in result_cache.columns, f"Missing: {col}"

    def test_keyword_fallback(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        news = _make_news_df(20, 100)
        # No Gemini client -> keyword fallback
        result_cache, result = compute_news_sentiment(cache, news_df=news)

        assert result.scoring_method == "keyword"
        assert result.n_articles_scored == 20

    def test_sentiment_range(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        news = _make_news_df(50, 100)
        result_cache, _ = compute_news_sentiment(cache, news_df=news)

        valid = result_cache["sentiment_score"].dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_result_summary(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        news = _make_news_df(30, 100)
        _, result = compute_news_sentiment(cache, news_df=news)

        assert result.n_articles_fetched == 30
        assert result.n_articles_scored == 30
        assert not math.isnan(result.mean_sentiment)
        assert result.latest_label in ("Bearish", "Slightly Bearish", "Neutral", "Slightly Bullish", "Bullish")

    def test_no_news(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        result_cache, result = compute_news_sentiment(cache)

        assert "sentiment_score" in result_cache.columns
        assert result_cache["sentiment_score"].isna().all()
        assert result.n_articles_fetched == 0

    def test_momentum_columns(self):
        from operator1.features.news_sentiment import compute_news_sentiment
        cache = _make_cache(100)
        news = _make_news_df(50, 100)
        result_cache, _ = compute_news_sentiment(cache, news_df=news)

        assert result_cache["sentiment_momentum_5d"].notna().any()
        assert result_cache["sentiment_momentum_21d"].notna().any()

    def test_keyword_scorer_positive(self):
        from operator1.features.news_sentiment import _keyword_score
        score = _keyword_score("Record revenue and strong growth beat estimates")
        assert score > 0

    def test_keyword_scorer_negative(self):
        from operator1.features.news_sentiment import _keyword_score
        score = _keyword_score("Decline in sales and lawsuit leads to loss")
        assert score < 0

    def test_keyword_scorer_neutral(self):
        from operator1.features.news_sentiment import _keyword_score
        score = _keyword_score("The quick brown fox jumps over the lazy dog")
        assert score == 0.0


# ===========================================================================
# Profile Builder Integration Tests
# ===========================================================================


class TestProfileBuilderNewSections:
    def test_sentiment_section(self):
        from operator1.report.profile_builder import _build_sentiment_section
        section = _build_sentiment_section(None, {
            "n_articles_scored": 50,
            "scoring_method": "keyword",
            "mean_sentiment": 0.15,
            "latest_sentiment": 0.2,
            "latest_label": "Slightly Bullish",
        })
        assert section["available"] is True
        assert section["n_articles_scored"] == 50

    def test_peer_ranking_section(self):
        from operator1.report.profile_builder import _build_peer_ranking_section
        section = _build_peer_ranking_section(None, {
            "n_peers": 5,
            "n_variables_ranked": 10,
            "latest_composite_rank": 65.0,
            "latest_label": "Above Average",
            "variable_ranks": {"current_ratio": 70.0},
        })
        assert section["available"] is True
        assert section["n_peers"] == 5

    def test_macro_quadrant_section(self):
        from operator1.report.profile_builder import _build_macro_quadrant_section
        section = _build_macro_quadrant_section(None, {
            "current_quadrant": "goldilocks",
            "quadrant_distribution": {"goldilocks": 0.6, "reflation": 0.4},
            "n_transitions": 3,
            "growth_trend": 2.5,
            "inflation_target": 2.0,
            "n_days_classified": 252,
        })
        assert section["available"] is True
        assert section["current_quadrant"] == "goldilocks"

    def test_all_sections_unavailable(self):
        from operator1.report.profile_builder import (
            _build_sentiment_section,
            _build_peer_ranking_section,
            _build_macro_quadrant_section,
        )
        assert _build_sentiment_section(None, None)["available"] is False
        assert _build_peer_ranking_section(None, None)["available"] is False
        assert _build_macro_quadrant_section(None, None)["available"] is False
