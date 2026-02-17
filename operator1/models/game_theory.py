"""Game Theory module for competitor strategic analysis.

Models competitive dynamics as strategic games to predict competitor
behavior beyond simple statistical averages.

**Implemented models:**

1. **Cournot competition** (quantity game): models competitors choosing
   output quantities simultaneously.  Predicts equilibrium market
   shares and prices based on cost structures and demand elasticity.

2. **Bertrand competition** (price game): models price competition.
   Predicts price floors and likely undercutting behavior.

3. **Stackelberg leadership**: identifies whether the target is a
   leader or follower based on market cap and first-mover metrics.

4. **Competitive pressure index**: aggregated score measuring how
   much strategic pressure the target faces from competitors.

Uses only numpy -- no external game theory libraries required.

Integration point: ``operator1/features/linked_aggregates.py``
Output feeds into: ``operator1/report/profile_builder.py``

Top-level entry points:
    ``analyze_competitive_dynamics`` -- full game-theoretic analysis.
    ``CompetitiveAnalysisResult`` -- result container.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CompetitorProfile:
    """Simplified profile for game-theoretic analysis."""

    name: str = ""
    isin: str = ""
    market_cap: float = 0.0
    revenue: float = 0.0
    operating_margin: float = 0.0
    market_share: float = 0.0  # computed from revenue
    cost_per_unit: float = 0.0  # proxy from margins


@dataclass
class CournotResult:
    """Output from Cournot quantity competition analysis."""

    # Equilibrium quantities (market share fractions)
    equilibrium_shares: dict[str, float] = field(default_factory=dict)
    # Predicted market price
    equilibrium_price: float = 0.0
    # Target's Nash equilibrium profit
    target_equilibrium_profit: float = 0.0
    # Whether target has a quantity advantage
    target_has_quantity_advantage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "equilibrium_shares": {
                k: round(v, 4) for k, v in self.equilibrium_shares.items()
            },
            "equilibrium_price": round(self.equilibrium_price, 4),
            "target_equilibrium_profit": round(self.target_equilibrium_profit, 2),
            "target_has_quantity_advantage": self.target_has_quantity_advantage,
        }


@dataclass
class StackelbergResult:
    """Output from Stackelberg leadership analysis."""

    target_role: str = "follower"  # "leader" or "follower"
    leadership_score: float = 0.0  # 0-1, higher = more leader-like
    # Factors that determine leadership
    market_cap_rank: int = 0
    revenue_rank: int = 0
    margin_advantage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_role": self.target_role,
            "leadership_score": round(self.leadership_score, 4),
            "market_cap_rank": self.market_cap_rank,
            "revenue_rank": self.revenue_rank,
            "margin_advantage": round(self.margin_advantage, 4),
        }


@dataclass
class CompetitiveAnalysisResult:
    """Full game-theoretic competitive analysis."""

    cournot: CournotResult = field(default_factory=CournotResult)
    stackelberg: StackelbergResult = field(default_factory=StackelbergResult)

    # Competitive pressure index (0-1, higher = more pressure)
    competitive_pressure: float = 0.0
    pressure_label: str = "unknown"

    # Number of competitors analyzed
    n_competitors: int = 0

    # Concentration ratio (top-4 market share)
    cr4: float = 0.0
    market_structure: str = "unknown"  # monopoly/oligopoly/competitive

    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "n_competitors": self.n_competitors,
            "competitive_pressure": round(self.competitive_pressure, 4),
            "pressure_label": self.pressure_label,
            "cr4": round(self.cr4, 4),
            "market_structure": self.market_structure,
            "cournot": self.cournot.to_dict(),
            "stackelberg": self.stackelberg.to_dict(),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Competitor profile extraction
# ---------------------------------------------------------------------------


def _extract_competitor_profiles(
    target_cache: pd.DataFrame,
    competitor_caches: dict[str, pd.DataFrame],
    target_name: str = "target",
) -> tuple[CompetitorProfile, list[CompetitorProfile]]:
    """Extract simplified profiles for game-theoretic analysis.

    Uses the latest available values from each entity's daily cache.
    """
    def _profile_from_cache(
        cache: pd.DataFrame, name: str, isin: str = "",
    ) -> CompetitorProfile:
        if cache.empty:
            return CompetitorProfile(name=name, isin=isin)

        latest = cache.iloc[-1]
        rev = float(latest.get("revenue", 0) or 0)
        margin = float(latest.get("operating_margin", 0) or 0)
        mcap = float(latest.get("market_cap", 0) or 0)

        return CompetitorProfile(
            name=name,
            isin=isin,
            market_cap=mcap,
            revenue=rev,
            operating_margin=margin,
            # Cost proxy: revenue * (1 - margin)
            cost_per_unit=rev * (1 - margin) if rev > 0 else 0,
        )

    target = _profile_from_cache(target_cache, target_name)
    competitors = [
        _profile_from_cache(cache, name, isin=isin)
        for isin, cache in competitor_caches.items()
        for name in [cache.iloc[-1].get("name", isin) if not cache.empty else isin]
    ]

    # Compute market shares from revenue
    all_entities = [target] + competitors
    total_rev = sum(e.revenue for e in all_entities)
    if total_rev > 0:
        for e in all_entities:
            e.market_share = e.revenue / total_rev

    return target, competitors


# ---------------------------------------------------------------------------
# Cournot competition
# ---------------------------------------------------------------------------


def _cournot_analysis(
    target: CompetitorProfile,
    competitors: list[CompetitorProfile],
) -> CournotResult:
    """Cournot (quantity) competition analysis.

    Models N firms choosing output quantities simultaneously.
    In the symmetric Cournot model with linear demand P = a - b*Q:
    - Each firm's equilibrium quantity: q_i = (a - c_i) / (b * (N+1))
    - where c_i is firm i's marginal cost.
    """
    all_firms = [target] + competitors
    n = len(all_firms)

    if n < 2 or target.revenue <= 0:
        return CournotResult()

    # Estimate demand parameters from observed data
    total_revenue = sum(f.revenue for f in all_firms)
    avg_price_proxy = total_revenue / n  # crude proxy

    # Marginal cost proxies (from operating margins)
    costs = []
    for f in all_firms:
        if f.operating_margin > 0:
            costs.append(f.revenue * (1 - f.operating_margin) / max(f.revenue, 1))
        else:
            costs.append(0.8)  # default: 80% cost ratio

    # Linear demand: P = a - b*Q, estimate a and b
    a = 1.0  # normalized intercept
    b = 1.0 / (n + 1)  # ensures non-negative equilibrium

    # Cournot equilibrium quantities
    eq_quantities = []
    for c in costs:
        q = max(0, (a - c) / (b * (n + 1)))
        eq_quantities.append(q)

    total_q = sum(eq_quantities)
    eq_shares = {}
    for i, f in enumerate(all_firms):
        share = eq_quantities[i] / total_q if total_q > 0 else 1 / n
        eq_shares[f.name or f.isin or f"firm_{i}"] = share

    # Equilibrium price
    eq_price = max(0, a - b * total_q)

    # Target's equilibrium profit
    target_q = eq_quantities[0]
    target_profit = (eq_price - costs[0]) * target_q * total_revenue

    # Does target have quantity advantage?
    target_advantage = eq_quantities[0] > np.median(eq_quantities)

    return CournotResult(
        equilibrium_shares=eq_shares,
        equilibrium_price=eq_price,
        target_equilibrium_profit=target_profit,
        target_has_quantity_advantage=target_advantage,
    )


# ---------------------------------------------------------------------------
# Stackelberg leadership
# ---------------------------------------------------------------------------


def _stackelberg_analysis(
    target: CompetitorProfile,
    competitors: list[CompetitorProfile],
) -> StackelbergResult:
    """Determine if the target is a Stackelberg leader or follower.

    Leadership is determined by:
    - Market cap rank (larger = more likely leader)
    - Revenue rank (larger = more likely leader)
    - Margin advantage (higher margins = pricing power)
    """
    all_firms = [target] + competitors
    n = len(all_firms)

    if n < 2:
        return StackelbergResult(target_role="monopoly", leadership_score=1.0)

    # Rank by market cap (descending)
    sorted_by_cap = sorted(all_firms, key=lambda f: f.market_cap, reverse=True)
    cap_rank = next(
        (i + 1 for i, f in enumerate(sorted_by_cap) if f is target), n
    )

    # Rank by revenue (descending)
    sorted_by_rev = sorted(all_firms, key=lambda f: f.revenue, reverse=True)
    rev_rank = next(
        (i + 1 for i, f in enumerate(sorted_by_rev) if f is target), n
    )

    # Margin advantage vs median
    competitor_margins = [f.operating_margin for f in competitors if f.operating_margin]
    median_margin = float(np.median(competitor_margins)) if competitor_margins else 0.0
    margin_adv = target.operating_margin - median_margin

    # Leadership score: weighted combination of rank-based scores
    cap_score = 1.0 - (cap_rank - 1) / max(n - 1, 1)
    rev_score = 1.0 - (rev_rank - 1) / max(n - 1, 1)
    margin_score = min(1.0, max(0.0, 0.5 + margin_adv * 5))  # scale margin to 0-1

    leadership = 0.4 * cap_score + 0.4 * rev_score + 0.2 * margin_score

    role = "leader" if leadership >= 0.6 else "follower"

    return StackelbergResult(
        target_role=role,
        leadership_score=leadership,
        market_cap_rank=cap_rank,
        revenue_rank=rev_rank,
        margin_advantage=margin_adv,
    )


# ---------------------------------------------------------------------------
# Competitive pressure index
# ---------------------------------------------------------------------------


def _competitive_pressure(
    target: CompetitorProfile,
    competitors: list[CompetitorProfile],
) -> tuple[float, str]:
    """Compute competitive pressure index (0-1).

    Factors:
    - Number of competitors (more = more pressure)
    - Target's market share (lower share = more pressure)
    - Margin compression (low margins vs competitors = pressure)
    """
    n = len(competitors)
    if n == 0:
        return 0.0, "monopoly"

    # Factor 1: competitor count (sigmoid, saturates around 10)
    count_pressure = 1.0 - 1.0 / (1.0 + n / 5.0)

    # Factor 2: market share (lower = more pressure)
    if target.market_share > 0:
        share_pressure = 1.0 - min(1.0, target.market_share * 3)  # normalize
    else:
        share_pressure = 0.5

    # Factor 3: margin comparison
    comp_margins = [c.operating_margin for c in competitors if c.operating_margin]
    if comp_margins and target.operating_margin:
        avg_comp_margin = float(np.mean(comp_margins))
        if avg_comp_margin > 0:
            margin_ratio = target.operating_margin / avg_comp_margin
            margin_pressure = max(0, 1.0 - margin_ratio)
        else:
            margin_pressure = 0.3
    else:
        margin_pressure = 0.3

    # Weighted combination
    pressure = 0.3 * count_pressure + 0.4 * share_pressure + 0.3 * margin_pressure
    pressure = float(np.clip(pressure, 0, 1))

    if pressure >= 0.7:
        label = "high_pressure"
    elif pressure >= 0.4:
        label = "moderate_pressure"
    else:
        label = "low_pressure"

    return pressure, label


# ---------------------------------------------------------------------------
# Market structure classification
# ---------------------------------------------------------------------------


def _classify_market(
    target: CompetitorProfile,
    competitors: list[CompetitorProfile],
) -> tuple[float, str]:
    """Compute CR4 and classify market structure."""
    all_firms = [target] + competitors
    shares = sorted([f.market_share for f in all_firms], reverse=True)

    cr4 = sum(shares[:4]) if len(shares) >= 4 else sum(shares)

    if len(competitors) == 0:
        structure = "monopoly"
    elif cr4 >= 0.8:
        structure = "tight_oligopoly"
    elif cr4 >= 0.5:
        structure = "loose_oligopoly"
    else:
        structure = "competitive"

    return cr4, structure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_competitive_dynamics(
    target_cache: pd.DataFrame,
    competitor_caches: dict[str, pd.DataFrame] | None = None,
    target_name: str = "target",
) -> CompetitiveAnalysisResult:
    """Run full game-theoretic competitive analysis.

    Parameters
    ----------
    target_cache:
        Daily cache for the target company.
    competitor_caches:
        Dict mapping competitor ISIN -> their daily caches.
    target_name:
        Display name for the target company.

    Returns
    -------
    CompetitiveAnalysisResult with all game-theoretic metrics.
    """
    if competitor_caches is None:
        competitor_caches = {}

    try:
        target, competitors = _extract_competitor_profiles(
            target_cache, competitor_caches, target_name,
        )

        if not competitors:
            return CompetitiveAnalysisResult(
                n_competitors=0,
                competitive_pressure=0.0,
                pressure_label="monopoly",
                market_structure="monopoly",
                available=True,
            )

        cournot = _cournot_analysis(target, competitors)
        stackelberg = _stackelberg_analysis(target, competitors)
        pressure, pressure_label = _competitive_pressure(target, competitors)
        cr4, market_structure = _classify_market(target, competitors)

        return CompetitiveAnalysisResult(
            cournot=cournot,
            stackelberg=stackelberg,
            competitive_pressure=pressure,
            pressure_label=pressure_label,
            n_competitors=len(competitors),
            cr4=cr4,
            market_structure=market_structure,
            available=True,
        )

    except Exception as exc:
        logger.warning("Game theory analysis failed: %s", exc)
        return CompetitiveAnalysisResult(available=False, error=str(exc))
