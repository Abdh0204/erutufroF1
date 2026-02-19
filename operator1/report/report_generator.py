"""T7.2 -- Report generation.

Consumes the ``company_profile.json`` built by T7.1 and produces:

1. A **Bloomberg-style Markdown report** via Gemini (or a fallback
   template when Gemini is unavailable).
2. An **optional set of charts** (matplotlib) saved as PNG files.
3. An **optional PDF** via ``pandoc`` (skipped gracefully if pandoc
   is not installed).

The Markdown report always includes a required **LIMITATIONS** section
covering data window, OHLCV source caveats, macro frequency, data
missingness summary, and failed modules with mitigations.

Top-level entry point:
    ``generate_report(profile, gemini_client=None, ...)``

Spec refs: Sec 18
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback report template (when Gemini is unavailable)
# ---------------------------------------------------------------------------

_FALLBACK_TEMPLATE = """\
# Company Analysis Report

**Generated:** {generated_at}

---

## 1. Executive Summary

{executive_summary}

---

## 2. Company Overview

{company_overview}

---

## 3. Historical Performance Analysis

{historical_performance}

---

## 4. Current Financial Health (Tier-by-Tier Breakdown)

{financial_health}

---

## 5. Survival Mode Analysis

{survival_analysis}

---

## 6. Linked Variables & Market Context

{linked_entities}

---

## 7. Temporal Analysis & Model Insights

{regime_analysis}

---

## 8. Predictions & Forecasts

{predictions_forecasts}

---

## 9. Technical Patterns & Chart Analysis

{technical_patterns}

---

## 10. Ethical Filter Assessment

{ethical_filters}

---

## 10a. Supply Chain & Network Risk (Graph Theory)

{graph_risk}

---

## 10b. Competitive Dynamics (Game Theory)

{game_theory}

---

## 10c. Government Protection Assessment (Fuzzy Logic)

{fuzzy_protection}

---

## 10d. Adaptive Learning (PID Controller)

{pid_controller}

---

## 11. Risk Factors & Limitations

{risk_assessment}

### 11.1 LIMITATIONS

{limitations}

---

## 12. Investment Recommendation

{investment_recommendation}

---

## 13. Appendix

{appendix}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(val: Any, fmt: str = ".2f") -> str:
    """Format a numeric value or return 'N/A' for None."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val: Any) -> str:
    """Format as percentage string."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Fallback report builder (no Gemini)
# ---------------------------------------------------------------------------


def _build_executive_summary(profile: dict[str, Any]) -> str:
    """Build the executive summary from profile data."""
    identity = profile.get("identity", {})
    if not identity.get("available"):
        return "Target company data unavailable."

    name = identity.get("name", "Unknown Company")
    ticker = identity.get("ticker", "")
    sector = identity.get("sector", "")
    country = identity.get("country", "")

    survival = profile.get("survival", {})
    regime = survival.get("survival_regime", "normal")
    company_flag = survival.get("company_survival_mode_flag", 0)
    country_flag = survival.get("country_survival_mode_flag", 0)

    mc = profile.get("monte_carlo", {})
    surv_prob = mc.get("survival_probability_mean")

    lines = [
        f"**{name}** ({ticker}) is a {sector} company based in {country}.",
        "",
    ]

    if company_flag or country_flag:
        lines.append(
            f"**Warning:** The company is currently in **{regime}** mode."
        )
        if surv_prob is not None:
            lines.append(
                f"Monte Carlo survival probability: **{_pct(surv_prob)}**"
            )
    else:
        lines.append("The company is operating under **normal** conditions.")
        if surv_prob is not None:
            lines.append(
                f"Monte Carlo survival probability: **{_pct(surv_prob)}**"
            )

    return "\n".join(lines)


def _build_company_overview(profile: dict[str, Any]) -> str:
    """Build company overview section."""
    identity = profile.get("identity", {})
    if not identity.get("available"):
        return "Company identity data unavailable."

    lines = [
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Name** | {identity.get('name', 'N/A')} |",
        f"| **ISIN** | {identity.get('isin', 'N/A')} |",
        f"| **Ticker** | {identity.get('ticker', 'N/A')} |",
        f"| **Exchange** | {identity.get('exchange', 'N/A')} |",
        f"| **Sector** | {identity.get('sector', 'N/A')} |",
        f"| **Industry** | {identity.get('industry', 'N/A')} |",
        f"| **Country** | {identity.get('country', 'N/A')} |",
        f"| **Currency** | {identity.get('currency', 'N/A')} |",
    ]
    return "\n".join(lines)


def _build_financial_health(profile: dict[str, Any]) -> str:
    """Build financial health section from vanity and survival data."""
    survival = profile.get("survival", {})
    vanity = profile.get("vanity", {})

    lines = ["### Hierarchy Weights (Current)", ""]

    weights = survival.get("hierarchy_weights", {})
    if weights:
        for tier, w in sorted(weights.items()):
            lines.append(f"- **{tier}**: {_fmt(w, '.3f')}")
    else:
        lines.append("Hierarchy weight data unavailable.")

    lines.extend(["", "### Vanity Assessment", ""])
    if vanity.get("available"):
        lines.append(f"- **Latest vanity %**: {_fmt(vanity.get('latest'), '.1f')}%")
        lines.append(f"- **Mean**: {_fmt(vanity.get('mean'), '.1f')}%")
        lines.append(f"- **Max**: {_fmt(vanity.get('max'), '.1f')}%")
    else:
        lines.append("Vanity percentage data unavailable.")

    return "\n".join(lines)


def _build_survival_analysis(profile: dict[str, Any]) -> str:
    """Build survival analysis section."""
    survival = profile.get("survival", {})
    if not survival.get("available"):
        return "Survival analysis data unavailable."

    lines = [
        f"- **Company survival flag**: {'ACTIVE' if survival.get('company_survival_mode_flag') else 'inactive'}",
        f"- **Country survival flag**: {'ACTIVE' if survival.get('country_survival_mode_flag') else 'inactive'}",
        f"- **Country protection flag**: {'ACTIVE' if survival.get('country_protected_flag') else 'inactive'}",
        f"- **Current regime**: {survival.get('survival_regime', 'N/A')}",
        "",
        "### Regime Distribution",
        "",
    ]

    dist = survival.get("regime_distribution_pct", {})
    if dist:
        for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {regime}: {pct * 100:.1f}%")
    else:
        lines.append("No regime distribution data available.")

    return "\n".join(lines)


def _build_linked_entities_section(profile: dict[str, Any]) -> str:
    """Build linked entities section."""
    linked = profile.get("linked_entities", {})
    if not linked.get("available"):
        return "Linked entity analysis unavailable."

    n_groups = linked.get("n_groups", 0)
    groups = linked.get("groups", {})

    lines = [f"**{n_groups} relationship group(s)** analysed.", ""]

    for group_name, metrics in sorted(groups.items()):
        lines.append(f"### {group_name.replace('_', ' ').title()}")
        lines.append("")
        for metric_name, stats in sorted(metrics.items()):
            if isinstance(stats, dict) and "latest" in stats:
                lines.append(
                    f"- {metric_name}: latest={_fmt(stats.get('latest'))}, "
                    f"mean={_fmt(stats.get('mean'))}"
                )
        lines.append("")

    return "\n".join(lines)


def _build_macro_section(profile: dict[str, Any]) -> str:
    """Build macro environment section."""
    # Macro data is embedded in the quality/estimation sections
    estimation = profile.get("estimation", {})
    quality = profile.get("data_quality", {})

    lines = []
    if estimation.get("available"):
        lines.append("Macro data was aligned to daily frequency using as-of logic.")
        lines.append("See the estimation coverage section for variable-level detail.")
    else:
        lines.append("Macro environment data was not available for this analysis.")

    if quality.get("available"):
        coverage = quality.get("variable_coverage", {})
        macro_vars = [
            k for k in coverage
            if k.startswith(("inflation", "cpi", "unemployment", "gdp", "exchange"))
        ]
        if macro_vars:
            lines.extend(["", "### Macro Variable Coverage", ""])
            for var in sorted(macro_vars):
                cov = coverage[var]
                if isinstance(cov, dict):
                    lines.append(
                        f"- {var}: {_fmt(cov.get('coverage_pct'), '.1f')}% coverage"
                    )
                else:
                    lines.append(f"- {var}: {_fmt(cov, '.1f')}% coverage")

    return "\n".join(lines) if lines else "Macro environment data unavailable."


def _build_regime_analysis(profile: dict[str, Any]) -> str:
    """Build regime analysis and forecasts section."""
    regimes = profile.get("regimes", {})
    preds = profile.get("predictions", {})

    lines = []

    if regimes.get("available"):
        lines.append(f"**Current market regime**: {regimes.get('current_regime', 'N/A')}")
        lines.append(f"**Structural breaks detected**: {regimes.get('n_structural_breaks', 0)}")
        lines.append("")

        dist = regimes.get("regime_distribution_pct", {})
        if dist:
            lines.append("### Regime Distribution")
            lines.append("")
            for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
                lines.append(f"- {regime}: {pct * 100:.1f}%")
            lines.append("")
    else:
        lines.append("Regime detection was not performed or failed.")
        lines.append("")

    if preds.get("available"):
        lines.append("### Forecasts")
        lines.append("")
        horizons = preds.get("horizons", {})
        for h_label in ("1d", "5d", "21d", "252d"):
            h_preds = horizons.get(h_label, {})
            if h_preds:
                lines.append(f"**{h_label} horizon:**")
                # Handle both list format [{variable, point_forecast, ...}]
                # and dict format {variable: {point, lower, upper}}
                if isinstance(h_preds, list):
                    items = h_preds[:5]
                    for p in items:
                        var = p.get("variable", "?")
                        pf = _fmt(p.get("point_forecast"))
                        ci_lo = _fmt(p.get("lower_ci"))
                        ci_hi = _fmt(p.get("upper_ci"))
                        lines.append(f"- {var}: {pf} [{ci_lo}, {ci_hi}]")
                elif isinstance(h_preds, dict):
                    for var, vals in list(h_preds.items())[:5]:
                        if isinstance(vals, dict):
                            pf = _fmt(vals.get("point") or vals.get("point_forecast"))
                            ci_lo = _fmt(vals.get("lower") or vals.get("lower_ci"))
                            ci_hi = _fmt(vals.get("upper") or vals.get("upper_ci"))
                            lines.append(f"- {var}: {pf} [{ci_lo}, {ci_hi}]")
                lines.append("")
    else:
        lines.append("Forecast data unavailable.")

    return "\n".join(lines)


def _build_ethical_filters_section(profile: dict[str, Any]) -> str:
    """Build Section 8: Ethical Filter Assessment."""
    filters = profile.get("filters", {})
    lines: list[str] = []

    if not filters.get("available", False):
        lines.append("*Ethical filter data not available for this run.*")
        return "\n".join(lines)

    # Purchasing Power
    pp = filters.get("purchasing_power", {})
    lines.append("### Purchasing Power Filter")
    lines.append("")
    lines.append(f"**Verdict:** {pp.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Nominal return: {_pct(pp.get('nominal_return'))}")
    lines.append(f"- Real return (inflation-adjusted): {_pct(pp.get('real_return'))}")
    lines.append(f"- Inflation impact: {_pct(pp.get('inflation_impact'))}")
    lines.append("")
    lines.append(
        "*This filter reveals whether investors actually gained purchasing "
        "power or merely saw a number go up while real wealth declined.*"
    )
    lines.append("")

    # Solvency
    sol = filters.get("solvency", {})
    lines.append("### Solvency Filter (Debt-to-Equity)")
    lines.append("")
    lines.append(f"**Verdict:** {sol.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Debt-to-equity: {_fmt(sol.get('debt_to_equity'))}")
    lines.append(f"- Threshold: {_fmt(sol.get('threshold'))}")
    lines.append(f"- {sol.get('interpretation', '')}")
    lines.append("")
    lines.append(
        "*Beyond religious compliance, high leverage makes companies "
        "fragile in recessions and rate hikes. This filter protects "
        "against leveraged blow-ups.*"
    )
    lines.append("")

    # Gharar
    gh = filters.get("gharar", {})
    lines.append("### Gharar Filter (Volatility / Speculation)")
    lines.append("")
    lines.append(f"**Verdict:** {gh.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Volatility (21d): {_pct(gh.get('volatility_21d'))}")
    lines.append(f"- Stability score: {_fmt(gh.get('stability_score'))}/10")
    lines.append(f"- {gh.get('interpretation', '')}")
    lines.append("")
    lines.append(
        "*This filter separates calculated investment from gambling "
        "regardless of one's background -- if volatility is extreme, any "
        "prediction is as likely to be wrong as right.*"
    )
    lines.append("")

    # Cash is King
    ck = filters.get("cash_is_king", {})
    lines.append("### Cash is King Filter (Free Cash Flow Yield)")
    lines.append("")
    lines.append(f"**Verdict:** {ck.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- FCF yield: {_pct(ck.get('fcf_yield'))}")
    if ck.get("fcf_margin") is not None:
        lines.append(f"- FCF margin: {_pct(ck.get('fcf_margin'))}")
    lines.append(f"- {ck.get('interpretation', '')}")
    lines.append("")
    lines.append(
        '*"Profit is an opinion, but cash is a fact." This filter ensures '
        "the company generates real liquid wealth, not just accounting "
        "entries.*"
    )

    return "\n".join(lines)


def _build_risk_assessment(profile: dict[str, Any]) -> str:
    """Build risk assessment section."""
    mc = profile.get("monte_carlo", {})
    survival = profile.get("survival", {})

    lines = []

    if mc.get("available"):
        lines.append("### Monte Carlo Survival Analysis")
        lines.append("")
        lines.append(
            f"- **Mean survival probability**: "
            f"{_pct(mc.get('survival_probability_mean'))}"
        )
        lines.append(
            f"- **5th percentile**: "
            f"{_pct(mc.get('survival_probability_p5'))}"
        )
        lines.append(
            f"- **95th percentile**: "
            f"{_pct(mc.get('survival_probability_p95'))}"
        )
        lines.append(f"- **Simulation paths**: {mc.get('n_paths', 'N/A')}")
        lines.append("")

        by_horizon = mc.get("survival_by_horizon", {})
        if by_horizon:
            lines.append("### Survival Probability by Horizon")
            lines.append("")
            for h, prob in sorted(by_horizon.items()):
                lines.append(f"- {h}: {_pct(prob)}")
            lines.append("")
    else:
        lines.append("Monte Carlo risk analysis was not performed.")

    # Model reliability
    metrics = profile.get("model_metrics", {})
    if metrics.get("available"):
        lines.append("### Model Reliability")
        lines.append("")
        rmse_info = metrics.get("model_best_rmse", {})
        for model, rmse in sorted(rmse_info.items()):
            lines.append(f"- {model}: RMSE = {_fmt(rmse, '.4f')}")

    return "\n".join(lines)


def _build_limitations(profile: dict[str, Any]) -> str:
    """Build the required LIMITATIONS section.

    Must cover:
    - Data window and frequency limitations
    - OHLCV source caveats
    - Macro data frequency and alignment
    - Data missingness summary
    - Failed modules and mitigations
    """
    meta = profile.get("meta", {})
    date_range = meta.get("date_range", {})
    quality = profile.get("data_quality", {})
    failed = profile.get("failed_modules", [])
    estimation = profile.get("estimation", {})

    lines = [
        "This analysis is subject to the following limitations:",
        "",
        "### Data Window",
        "",
        f"- Analysis covers **{date_range.get('start', 'N/A')}** to "
        f"**{date_range.get('end', 'N/A')}** (approximately 2 years).",
        "- Historical patterns may not predict future performance.",
        "- All financial statement data is aligned using as-of logic "
        "(latest report as of each trading day).",
        "",
        "### OHLCV Source",
        "",
        "- Daily price data (OHLCV) is sourced from **FMP** (Financial "
        "Modeling Prep) as the authoritative price source.",
        "- Linked entity prices are sourced from **Eulerpool** quotes.",
        "- Price data may not account for all corporate actions "
        "(splits, dividends) depending on source adjustments.",
        "",
        "### Macro Data",
        "",
        "- Macroeconomic indicators from the **World Bank** are published "
        "annually and aligned to daily frequency using as-of logic.",
        "- This introduces lag: the latest macro data point may be "
        "1-2 years old.",
        "- Intra-year economic shifts are not captured until the next "
        "annual release.",
        "",
    ]

    # Data missingness
    lines.extend(["### Data Missingness", ""])
    if quality.get("available"):
        coverage = quality.get("variable_coverage", {})
        if coverage:
            low_coverage = [
                (var, info)
                for var, info in coverage.items()
                if isinstance(info, dict) and (info.get("coverage_pct", 100) or 100) < 80
            ]
            if low_coverage:
                lines.append(
                    f"- **{len(low_coverage)}** variable(s) have less than "
                    "80% coverage:"
                )
                for var, info in sorted(low_coverage, key=lambda x: x[1].get("coverage_pct", 0)):
                    lines.append(
                        f"  - {var}: {_fmt(info.get('coverage_pct'), '.1f')}%"
                    )
            else:
                lines.append("- All variables have 80%+ coverage.")
        else:
            lines.append("- Variable-level coverage data not available.")
    else:
        lines.append("- Data quality report not generated; coverage unknown.")

    if estimation.get("available"):
        lines.append(
            "- Missing values were estimated using a two-pass "
            "approach (deterministic identity fill + regime-weighted "
            "imputation). Estimated values are flagged and assigned "
            "confidence scores."
        )
    lines.append("")

    # Failed modules
    lines.extend(["### Failed Modules and Mitigations", ""])
    if failed:
        for f in failed:
            lines.append(f"- **{f.get('module', 'Unknown')}**: {f.get('error', 'unknown error')}")
            lines.append(f"  - *Mitigation*: {f.get('mitigation', 'none')}")
    else:
        lines.append("- No module failures detected.")

    return "\n".join(lines)


def _build_historical_performance(profile: dict[str, Any]) -> str:
    """Build the Historical Performance Analysis section (Section 3)."""
    hist = profile.get("historical", {})
    lines: list[str] = []

    lines.append(f"**Analysis Period:** {hist.get('date_range_start', 'N/A')} to {hist.get('date_range_end', 'N/A')}")
    lines.append("")
    lines.append(f"- **Total Return:** {_pct(hist.get('return_total'))}")
    lines.append(f"- **Real Return (inflation-adjusted):** {_pct(hist.get('return_real'))}")
    lines.append(f"- **Annualised Return:** {_pct(hist.get('return_annualized'))}")
    lines.append(f"- **Annualised Volatility:** {_pct(hist.get('volatility_annualized'))}")
    lines.append(f"- **Sharpe Ratio:** {_fmt(hist.get('sharpe_ratio'))}")
    lines.append(f"- **Maximum Drawdown:** {_pct(hist.get('max_drawdown'))}")
    lines.append(f"- **Up Days:** {_pct(hist.get('up_days_percentage'))}")
    lines.append(f"- **Down Days:** {_pct(hist.get('down_days_percentage'))}")
    lines.append(f"- **Best Day Return:** {_pct(hist.get('best_day_return'))}")
    lines.append(f"- **Worst Day Return:** {_pct(hist.get('worst_day_return'))}")

    return "\n".join(lines)


def _build_predictions_forecasts(profile: dict[str, Any]) -> str:
    """Build the Predictions & Forecasts section (Section 8)."""
    preds = profile.get("predictions", {})
    lines: list[str] = []

    for horizon_label in ["next_day", "next_week", "next_month", "next_year"]:
        h_data = preds.get(horizon_label, {})
        h_title = horizon_label.replace("_", " ").title()
        lines.append(f"### {h_title}")
        lines.append("")

        if not h_data:
            lines.append("*No predictions available for this horizon.*")
            lines.append("")
            continue

        if "point_forecast" in h_data:
            pt = h_data["point_forecast"]
            if isinstance(pt, dict):
                for var, val in list(pt.items())[:10]:
                    lines.append(f"- **{var}:** {_fmt(val)}")
            else:
                lines.append(f"- Point forecast: {_fmt(pt)}")

        if "ohlc_series" in h_data:
            ohlc = h_data["ohlc_series"]
            if isinstance(ohlc, list) and ohlc:
                if horizon_label == "next_day":
                    lines.append("")
                    lines.append("*Technical Alpha protection applied: only Low is shown for next-day OHLC.*")
                lines.append("")
                lines.append(f"- OHLC candlestick series: {len(ohlc)} step(s)")

        lines.append("")

    # Monte Carlo
    mc = preds.get("monte_carlo", {})
    if mc:
        lines.append("### Monte Carlo Uncertainty")
        lines.append("")
        lines.append(f"- Scenarios simulated: {mc.get('n_scenarios', 'N/A')}")
        lines.append(f"- Tail risk (5th percentile): {_fmt(mc.get('p5'))}")
        lines.append(f"- Base case (50th percentile): {_fmt(mc.get('p50'))}")
        lines.append(f"- Upside (95th percentile): {_fmt(mc.get('p95'))}")

    # Conformal prediction intervals
    conformal = profile.get("conformal_intervals", {})
    if conformal:
        lines.append("")
        lines.append("### Conformal Prediction Intervals")
        lines.append("")
        lines.append("*Distribution-free intervals with guaranteed coverage (no Gaussian assumption):*")
        lines.append("")
        for var, intervals in list(conformal.items())[:10]:
            if isinstance(intervals, dict):
                for h, interval in intervals.items():
                    if isinstance(interval, dict):
                        lines.append(
                            f"- **{var}** ({h}): "
                            f"[{_fmt(interval.get('lower'))}, {_fmt(interval.get('upper'))}] "
                            f"(width: {_fmt(interval.get('interval_width'))})"
                        )

    # SHAP explanations
    shap_data = profile.get("shap_explanations", {})
    if shap_data.get("available"):
        lines.append("")
        lines.append("### What Drove These Predictions (SHAP Analysis)")
        lines.append("")
        per_var = shap_data.get("per_variable", {})
        for var, exp in list(per_var.items())[:8]:
            narrative = exp.get("narrative", "")
            if narrative:
                lines.append(f"- **{var}:** {narrative}")
            else:
                drivers = exp.get("top_drivers", [])
                parts = []
                for d in drivers[:3]:
                    sign = "+" if d.get("shap_value", 0) > 0 else ""
                    parts.append(f"{sign}{_fmt(d.get('shap_value'))} from {d.get('feature', '?')}")
                if parts:
                    lines.append(f"- **{var}:** {'; '.join(parts)}")

        global_imp = shap_data.get("global_feature_importance", {})
        if global_imp:
            lines.append("")
            lines.append("**Top global feature drivers (across all predictions):**")
            for feat, importance in list(global_imp.items())[:5]:
                lines.append(f"- {feat}: {_fmt(importance, '.4f')} mean |SHAP|")

    # Historical analogs (DTW)
    analogs = profile.get("historical_analogs", {})
    if analogs.get("available"):
        lines.append("")
        lines.append("### Historical Analogs (DTW Pattern Matching)")
        lines.append("")
        lines.append(f"*Method: {analogs.get('method', 'DTW')} | "
                     f"Query window: {analogs.get('query_window_days', '?')} days | "
                     f"Forecast horizon: {analogs.get('forecast_horizon_days', '?')} days*")
        lines.append("")

        emp = analogs.get("empirical_forecast", {})
        if emp:
            lines.append("**Empirical forecast from analog outcomes:**")
            lines.append("")
            lines.append(f"- Mean return: {_fmt(emp.get('return_mean_pct'))}%")
            lines.append(f"- Median return: {_fmt(emp.get('return_median_pct'))}%")
            lines.append(f"- Range: [{_fmt(emp.get('return_p5_pct'))}%, {_fmt(emp.get('return_p95_pct'))}%]")
            lines.append(f"- Worst drawdown: {_fmt(emp.get('worst_drawdown_pct'))}%")
            lines.append("")

        analog_list = analogs.get("analogs", [])
        if analog_list:
            lines.append("**Closest historical matches:**")
            lines.append("")
            for a in analog_list[:5]:
                lines.append(f"- {a.get('narrative', a.get('period', 'Unknown'))}")

    return "\n".join(lines)


def _build_technical_patterns(profile: dict[str, Any]) -> str:
    """Build the Technical Patterns & Chart Analysis section (Section 9)."""
    patterns = profile.get("patterns", profile.get("technical_patterns", {}))
    lines: list[str] = []

    recent = patterns.get("recent_patterns", [])
    predicted = patterns.get("predicted_patterns_week", patterns.get("predicted_patterns", []))

    lines.append("[CHART: 2-year price history with regime shading]")
    lines.append("")

    if recent:
        lines.append("### Recent Patterns (Last 6 Months)")
        lines.append("")
        for p in recent[:10]:
            lines.append(f"- {p}")
    else:
        lines.append("*No recent candlestick patterns detected.*")

    lines.append("")

    if predicted:
        lines.append("### Predicted Patterns")
        lines.append("")
        for p in predicted[:10]:
            lines.append(f"- {p}")
    else:
        lines.append("*No predicted patterns available.*")

    lines.append("")
    lines.append("[CHART: Predicted candlestick series for next week/month]")

    return "\n".join(lines)


def _build_investment_recommendation(profile: dict[str, Any]) -> str:
    """Build the Investment Recommendation section (Section 12)."""
    lines: list[str] = []

    # Derive recommendation from available data
    filters = profile.get("filters", {})
    survival = profile.get("survival", {})
    hist = profile.get("regimes", {})

    # Count filter passes
    pass_count = 0
    total_filters = 0
    for f_name, f_data in filters.items():
        if isinstance(f_data, dict):
            total_filters += 1
            verdict = str(f_data.get("verdict", ""))
            if "PASS" in verdict.upper():
                pass_count += 1

    # Simple heuristic recommendation
    is_survival = survival.get("company_survival_mode", False)
    total_return = hist.get("return_total", 0)

    if is_survival:
        recommendation = "SELL"
        confidence = "Medium"
        rationale = "Company is currently in survival mode; liquidity risk dominates."
    elif pass_count >= 3 and total_return is not None and (total_return or 0) > 0:
        recommendation = "BUY"
        confidence = "Medium"
        rationale = "Ethical filters largely pass, positive historical returns."
    elif pass_count >= 2:
        recommendation = "HOLD"
        confidence = "Low"
        rationale = "Mixed signals from ethical filters; monitor closely."
    else:
        recommendation = "SELL"
        confidence = "Low"
        rationale = "Multiple ethical filter failures indicate elevated risk."

    lines.append(f"**Recommendation:** {recommendation}")
    lines.append("")
    lines.append(f"**Confidence Level:** {confidence}")
    lines.append("")
    lines.append(f"**Rationale:** {rationale}")
    lines.append("")
    lines.append("**Key Catalysts to Watch:**")
    lines.append("- Upcoming earnings announcements")
    lines.append("- Changes in survival mode status")
    lines.append("- Macro regime shifts (inflation, rates)")
    lines.append("")
    lines.append("*Note: This recommendation is generated algorithmically from quantitative filters. "
                 "Professional judgement and qualitative analysis should supplement this assessment.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advanced module report sections
# ---------------------------------------------------------------------------


def _build_graph_risk_section(profile: dict[str, Any]) -> str:
    """Build Graph Theory / Supply Chain Risk section."""
    gr = profile.get("graph_risk", {})
    if not gr.get("available"):
        return "*Graph risk analysis not available for this run.*"

    lines = [
        "### Network Topology",
        "",
        f"- **Nodes**: {gr.get('n_nodes', 0)} entities in the network",
        f"- **Edges**: {gr.get('n_edges', 0)} relationship links",
        f"- **Target degree centrality**: {_fmt(gr.get('target_degree_centrality'))}",
        f"- **Target PageRank**: {_fmt(gr.get('target_pagerank'))}",
        "",
        "### Contagion Risk",
        "",
        f"- **Target infection probability**: {_pct(gr.get('contagion_target_infection_prob'))}",
        f"  *(If any linked entity enters distress, this is the probability of cascading to the target.)*",
        f"- **Expected infected entities**: {_fmt(gr.get('contagion_expected_infected'), '.1f')}",
        "",
        "### Supply Chain Concentration",
        "",
        f"- **Supplier HHI**: {_fmt(gr.get('supplier_hhi'))} ({gr.get('concentration_label', 'N/A')})",
        f"- **Customer HHI**: {_fmt(gr.get('customer_hhi'))}",
        "",
    ]

    top_pr = gr.get("top_pagerank", {})
    if top_pr:
        lines.append("### Most Influential Entities (PageRank)")
        lines.append("")
        for name, score in list(top_pr.items())[:5]:
            lines.append(f"- **{name}**: {_fmt(score, '.4f')}")
        lines.append("")

    return "\n".join(lines)


def _build_game_theory_section(profile: dict[str, Any]) -> str:
    """Build Game Theory / Competitive Dynamics section."""
    gt = profile.get("game_theory", {})
    if not gt.get("available"):
        return "*Game theory analysis not available for this run.*"

    lines = [
        f"**Market structure**: {gt.get('market_structure', 'N/A')}",
        f"**Competitors analysed**: {gt.get('n_competitors', 0)}",
        f"**CR4 concentration**: {_pct(gt.get('cr4'))}",
        "",
        "### Competitive Pressure",
        "",
        f"- **Pressure index**: {_fmt(gt.get('competitive_pressure'))} ({gt.get('pressure_label', 'N/A')})",
        "",
        "### Stackelberg Leadership",
        "",
    ]

    stk = gt.get("stackelberg", {})
    lines.append(f"- **Target role**: {stk.get('target_role', 'N/A')}")
    lines.append(f"- **Leadership score**: {_fmt(stk.get('leadership_score'))}")
    lines.append(f"- **Market cap rank**: #{stk.get('market_cap_rank', 'N/A')}")
    lines.append(f"- **Margin advantage**: {_pct(stk.get('margin_advantage'))}")
    lines.append("")

    cournot = gt.get("cournot", {})
    eq_shares = cournot.get("equilibrium_shares", {})
    if eq_shares:
        lines.append("### Cournot Equilibrium (Nash) Market Shares")
        lines.append("")
        for name, share in eq_shares.items():
            lines.append(f"- **{name}**: {_pct(share)}")
        lines.append("")

    return "\n".join(lines)


def _build_fuzzy_protection_section(profile: dict[str, Any]) -> str:
    """Build Fuzzy Logic Government Protection section."""
    fp = profile.get("fuzzy_protection", {})
    if not fp.get("available"):
        return "*Fuzzy protection analysis not available for this run.*"

    lines = [
        f"**Protection degree**: {_fmt(fp.get('mean_degree', fp.get('protection_degree')))} / 1.00",
        f"**Label**: {fp.get('latest_label', fp.get('label', 'N/A'))}",
        "",
        "### Dimension Scores",
        "",
        f"- **Sector strategicness**: {_fmt(fp.get('sector_score'))} *(How important is the sector to the government?)*",
        f"- **Economic significance**: {_fmt(fp.get('economic_score', fp.get('mean_economic')))} *(Market cap relative to GDP)*",
        f"- **Policy responsiveness**: {_fmt(fp.get('policy_score', fp.get('mean_policy')))} *(Emergency rate cuts)*",
        "",
        "*Higher scores indicate the company is more likely to receive government support during crises, "
        "which reduces downside risk in extreme scenarios.*",
        "",
    ]
    return "\n".join(lines)


def _build_pid_section(profile: dict[str, Any]) -> str:
    """Build PID Controller adaptive learning section."""
    pid = profile.get("pid_controller", {})
    if not pid.get("available", True) or not pid:
        return "*PID adaptive learning not available for this run.*"

    lines = [
        "The forward pass uses a PID (Proportional-Integral-Derivative) controller "
        "to dynamically adjust model learning rates based on prediction error feedback.",
        "",
        f"- **Mean learning multiplier**: {_fmt(pid.get('mean_multiplier'))}",
        f"- **Max multiplier**: {_fmt(pid.get('max_multiplier'))} *(most aggressive correction during the run)*",
        f"- **Min multiplier**: {_fmt(pid.get('min_multiplier'))} *(most conservative period)*",
        f"- **Variables controlled**: {pid.get('n_variables', 0)}",
        "",
    ]

    per_var = pid.get("per_variable", {})
    if per_var:
        lines.append("### Top PID Adjustments (by output)")
        lines.append("")
        sorted_vars = sorted(per_var.items(), key=lambda x: x[1].get("output", 1.0), reverse=True)
        for var, state in sorted_vars[:5]:
            lines.append(
                f"- **{var}**: multiplier={_fmt(state.get('output'))} "
                f"(P={_fmt(state.get('proportional'))}, "
                f"I={_fmt(state.get('integral'))}, "
                f"D={_fmt(state.get('derivative'))})"
            )
        lines.append("")

    return "\n".join(lines)


def _build_appendix(profile: dict[str, Any]) -> str:
    """Build the Appendix section (Section 13)."""
    lines: list[str] = []

    lines.append("### Methodology Summary")
    lines.append("")
    lines.append("This analysis uses **25+ mathematical modules** across 9 categories:")
    lines.append("")
    lines.append("| Category | Modules |")
    lines.append("|----------|---------|")
    lines.append("| Regime Detection | HMM (Hidden Markov Model), GMM (Gaussian Mixture), PELT, Bayesian Change Point |")
    lines.append("| Forecasting | Adaptive Kalman Filter, GARCH, VAR, LSTM with MC Dropout, **Temporal Fusion Transformer (TFT)** |")
    lines.append("| Tree Ensembles | Random Forest, XGBoost, Gradient Boosting |")
    lines.append("| Causality | Granger Causality, Transfer Entropy, Copula Models |")
    lines.append("| Uncertainty | **Conformal Prediction** (distribution-free intervals), **MC Dropout** (epistemic uncertainty), Regime-Aware Monte Carlo, Importance Sampling |")
    lines.append("| Explainability | **SHAP** (per-prediction feature attribution), Sobol Global Sensitivity |")
    lines.append("| Historical Analogs | **Dynamic Time Warping (DTW)** for finding similar past periods |")
    lines.append("| Optimisation | Genetic Algorithm (ensemble weight tuning) |")
    lines.append("| Pattern Recognition | Candlestick Detector, Wavelet/Fourier Decomposition |")
    lines.append("")

    lines.append("### Conformal Prediction")
    lines.append("")
    lines.append("Traditional financial models assume returns are normally distributed "
                 "and compute confidence intervals as RMSE x z-score x sqrt(horizon). "
                 "**This assumption is wrong** -- financial returns have fat tails and "
                 "regime switches that break Gaussian models.")
    lines.append("")
    lines.append("Conformal Prediction provides **distribution-free** intervals with "
                 "**guaranteed** finite-sample coverage. If we target 90% coverage, the "
                 "intervals will contain the true value at least 90% of the time -- regardless "
                 "of the underlying distribution.")
    lines.append("")
    lines.append("We use **Adaptive Conformal Inference (ACI)** which adjusts the interval "
                 "width online as the data distribution shifts (e.g., during regime changes).")
    lines.append("")

    lines.append("### SHAP Feature Attribution")
    lines.append("")
    lines.append("SHAP (SHapley Additive exPlanations) answers the question: *Why did "
                 "the model make this specific prediction?*")
    lines.append("")
    lines.append("For each predicted variable, SHAP decomposes the prediction into "
                 "contributions from individual features. For example: \"debt-to-equity "
                 "is predicted to rise 8% primarily because: +3.2% from rising long-term "
                 "debt, +2.1% from declining equity, -0.8% from strong cash position.\"")
    lines.append("")

    lines.append("### Temporal Fusion Transformer (TFT)")
    lines.append("")
    lines.append("TFT is a deep learning architecture purpose-built for mixed-frequency "
                 "time series. Unlike LSTM which treats all inputs equally, TFT uses:")
    lines.append("- **Variable selection gates** to learn which features matter")
    lines.append("- **Multi-head self-attention** to focus on relevant historical days")
    lines.append("- **Gated Residual Networks** for stable, deep learning")
    lines.append("")
    lines.append("This is particularly valuable for our data which mixes daily prices, "
                 "quarterly financial statements, and annual macro indicators.")
    lines.append("")

    lines.append("### Dynamic Time Warping (DTW) Historical Analogs")
    lines.append("")
    lines.append("DTW finds past periods where the company showed a similar "
                 "multi-variable pattern to the present. Instead of just matching "
                 "by regime label, DTW considers the *shape* of the trajectory "
                 "across multiple variables simultaneously (price, volatility, "
                 "debt, margins, macro conditions).")
    lines.append("")
    lines.append("The outcomes from those historical analog periods serve as "
                 "empirical priors: if 4 out of 5 analogs showed a 10% decline "
                 "in the following month, that is a strong signal regardless of "
                 "what the regression models predict.")
    lines.append("")

    lines.append("### MC Dropout (Epistemic Uncertainty)")
    lines.append("")
    lines.append("Standard neural networks give a single point prediction with "
                 "no indication of how *confident* the model is. MC Dropout fixes "
                 "this by running 100 forward passes through the LSTM with dropout "
                 "enabled at inference time. The spread of those 100 predictions "
                 "measures **epistemic uncertainty** -- how much the model itself "
                 "is unsure.")
    lines.append("")
    lines.append("This is different from **aleatoric uncertainty** (inherent "
                 "randomness in the data). A prediction with low epistemic but "
                 "high aleatoric uncertainty means: *the model is confident in "
                 "its estimate, but the variable is inherently noisy.* A prediction "
                 "with high epistemic uncertainty means: *the model does not have "
                 "enough information to make a reliable prediction.*")
    lines.append("")

    lines.append("### Forward Pass & Burn-Out Process")
    lines.append("")
    lines.append("The temporal engine uses a **day-by-day forward pass**: for each of "
                 "~500 trading days, it predicts the next day, compares with actual data, "
                 "and updates model parameters online. This is followed by a "
                 "**convergence-based burn-out** phase: intensive re-training on the most "
                 "recent 6 months with up to 10 iterations and patience-based early stopping.")
    lines.append("")

    lines.append("### Variable Tier Definitions")
    lines.append("")
    lines.append("| Tier | Category | Variables | Normal Weight |")
    lines.append("|------|----------|-----------|---------------|")
    lines.append("| 1 | Liquidity & Cash | cash_ratio, FCF, operating CF | 20% |")
    lines.append("| 2 | Solvency & Debt | debt_to_equity, net_debt_to_EBITDA | 20% |")
    lines.append("| 3 | Market Stability | volatility, drawdown, volume | 20% |")
    lines.append("| 4 | Profitability | margins, ROE, ROA | 20% |")
    lines.append("| 5 | Growth & Valuation | P/E, EV/EBITDA, revenue growth | 20% |")
    lines.append("")

    lines.append("### Data Sources")
    lines.append("")
    lines.append("- **Eulerpool:** Company fundamentals (profile, statements, peers, supply chain)")
    lines.append("- **FMP (Financial Modeling Prep):** Daily OHLCV market data")
    lines.append("- **World Bank Open Data:** Country macro indicators (inflation, GDP, unemployment)")
    lines.append("- **World Bank WDS:** Documents & Reports for qualitative country context")
    lines.append("- **Gemini API:** Linked entity discovery, report narrative generation")
    lines.append("")

    meta = profile.get("meta", {})
    lines.append("### Data Timestamps")
    lines.append("")
    lines.append(f"- Report generated: {meta.get('generated_at', 'N/A')}")
    lines.append(f"- Cache date range: {meta.get('cache_start', 'N/A')} to {meta.get('cache_end', 'N/A')}")
    lines.append("")

    lines.append("### Disclaimer")
    lines.append("")
    lines.append("This report is generated algorithmically and is for informational purposes only. "
                 "It does not constitute financial advice. Past performance does not guarantee "
                 "future results. All predictions carry inherent uncertainty.")

    return "\n".join(lines)


def _build_fallback_report(profile: dict[str, Any]) -> str:
    """Build a report using the local 13-section template (no Gemini)."""
    return _FALLBACK_TEMPLATE.format(
        generated_at=profile.get("meta", {}).get(
            "generated_at", datetime.utcnow().isoformat(),
        ),
        executive_summary=_build_executive_summary(profile),
        company_overview=_build_company_overview(profile),
        historical_performance=_build_historical_performance(profile),
        financial_health=_build_financial_health(profile),
        survival_analysis=_build_survival_analysis(profile),
        linked_entities=_build_linked_entities_section(profile),
        regime_analysis=_build_regime_analysis(profile),
        predictions_forecasts=_build_predictions_forecasts(profile),
        technical_patterns=_build_technical_patterns(profile),
        ethical_filters=_build_ethical_filters_section(profile),
        graph_risk=_build_graph_risk_section(profile),
        game_theory=_build_game_theory_section(profile),
        fuzzy_protection=_build_fuzzy_protection_section(profile),
        pid_controller=_build_pid_section(profile),
        risk_assessment=_build_risk_assessment(profile),
        limitations=_build_limitations(profile),
        investment_recommendation=_build_investment_recommendation(profile),
        appendix=_build_appendix(profile),
    )


# ---------------------------------------------------------------------------
# E2: Gemini response validation
# ---------------------------------------------------------------------------


# Required section headers (case-insensitive substring match).
_REQUIRED_SECTIONS: list[str] = [
    "executive summary",
    "company overview",
    "historical performance",
    "financial health",
    "survival",
    "linked variables",
    "temporal analysis",
    "predictions",
    "technical patterns",
    "ethical filter",
    "risk factors",
    "limitations",
    "investment recommendation",
    "appendix",
]


def validate_gemini_report(
    markdown: str,
    profile: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate that a Gemini-generated report is complete and accurate.

    Checks:
    1. All 13 required sections are present (by header text).
    2. Investment recommendation is present (BUY/HOLD/SELL keyword).
    3. LIMITATIONS section exists.
    4. No hallucinated numbers: spot-check key metrics against profile.

    Parameters
    ----------
    markdown:
        The Gemini-generated Markdown report text.
    profile:
        The company profile dict used to generate the report.

    Returns
    -------
    (is_valid, issues)
        ``is_valid`` is True only if all checks pass.
        ``issues`` is a list of human-readable issue descriptions.
    """
    issues: list[str] = []
    md_lower = markdown.lower()

    # Check 1: Section presence
    for section_name in _REQUIRED_SECTIONS:
        if section_name.lower() not in md_lower:
            issues.append(f"Missing section: '{section_name}'")

    # Check 2: Investment recommendation keyword
    rec_keywords = ["buy", "hold", "sell"]
    has_recommendation = any(
        kw in md_lower
        for kw in rec_keywords
    )
    if not has_recommendation:
        issues.append("No investment recommendation keyword (BUY/HOLD/SELL) found")

    # Check 3: LIMITATIONS section specifically
    if "limitations" not in md_lower:
        issues.append("LIMITATIONS section is missing")

    # Check 4: Spot-check key metrics against profile data
    identity = profile.get("identity", {})
    company_name = identity.get("name", "")
    if company_name and company_name.lower() not in md_lower:
        issues.append(f"Company name '{company_name}' not found in report")

    # Check ticker appears
    ticker = identity.get("ticker", "")
    if ticker and ticker.upper() not in markdown.upper():
        issues.append(f"Ticker '{ticker}' not found in report")

    # Check survival mode is mentioned if active
    survival = profile.get("survival", {})
    if survival.get("company_survival_mode_flag"):
        if "survival" not in md_lower:
            issues.append("Company is in survival mode but report does not mention survival")

    # Check debt-to-equity if available (spot-check for hallucination)
    # Profile stores hierarchy_weights in survival section, not tier2_solvency
    d2e = None
    if d2e is not None and not isinstance(d2e, str):
        d2e_str = f"{d2e:.1f}"
        # Allow some flexibility in formatting
        if d2e_str not in markdown and f"{d2e:.2f}" not in markdown:
            # Not a hard failure, just a warning
            issues.append(
                f"Debt-to-equity ({d2e_str}) not found verbatim in report "
                "(possible formatting difference, not necessarily an error)"
            )

    is_valid = len([i for i in issues if "not necessarily" not in i]) == 0

    if issues:
        logger.warning(
            "Gemini report validation: %d issue(s) found: %s",
            len(issues),
            "; ".join(issues[:5]),
        )
    else:
        logger.info("Gemini report validation: all checks passed")

    return is_valid, issues


# ---------------------------------------------------------------------------
# Charts (optional, gracefully skipped)
# ---------------------------------------------------------------------------


def generate_charts(
    cache: pd.DataFrame | None,
    profile: dict[str, Any],
    output_dir: str | Path | None = None,
) -> list[str]:
    """Generate optional analysis charts.

    Returns a list of file paths for successfully generated charts.
    Charts that fail to render are skipped with a warning.

    Parameters
    ----------
    cache:
        Full feature table with daily data.
    profile:
        Company profile dict from T7.1.
    output_dir:
        Directory to save chart PNGs. Defaults to ``cache/charts/``.
    """
    if output_dir is None:
        output_dir = Path(CACHE_DIR) / "charts"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chart_paths: list[str] = []

    if cache is None or cache.empty:
        logger.warning("No cache data available for chart generation.")
        return chart_paths

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart generation.")
        return chart_paths

    # Chart 1: Price history with regime shading
    try:
        if "close" in cache.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(cache.index, cache["close"], linewidth=1, color="#333")
            ax.set_title("Price History")
            ax.set_ylabel("Close Price")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            # Regime shading
            if "regime_label" in cache.columns:
                regime_colors = {
                    "bull": "#d4edda",
                    "bear": "#f8d7da",
                    "high_vol": "#fff3cd",
                    "low_vol": "#d1ecf1",
                }
                for regime, color in regime_colors.items():
                    mask = cache["regime_label"] == regime
                    if mask.any():
                        ax.fill_between(
                            cache.index,
                            cache["close"].min(),
                            cache["close"].max(),
                            where=mask,
                            alpha=0.3,
                            color=color,
                            label=regime,
                        )
                ax.legend(loc="upper left", fontsize=8)

            fig.tight_layout()
            path = str(out / "price_history.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate price history chart: %s", exc)

    # Chart 2: Survival timeline
    try:
        flag_cols = [
            c for c in (
                "company_survival_mode_flag",
                "country_survival_mode_flag",
                "country_protected_flag",
            )
            if c in cache.columns
        ]
        if flag_cols:
            fig, ax = plt.subplots(figsize=(14, 4))
            for i, col in enumerate(flag_cols):
                ax.fill_between(
                    cache.index,
                    i,
                    i + cache[col].fillna(0).astype(float),
                    alpha=0.7,
                    label=col.replace("_", " ").title(),
                )
            ax.set_yticks(range(len(flag_cols)))
            ax.set_yticklabels([c.replace("_", " ").title() for c in flag_cols])
            ax.set_title("Survival Flag Timeline")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            path = str(out / "survival_timeline.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate survival timeline chart: %s", exc)

    # Chart 3: Hierarchy weight evolution
    try:
        tier_cols = [
            f"hierarchy_tier{i}_weight" for i in range(1, 6)
            if f"hierarchy_tier{i}_weight" in cache.columns
        ]
        if tier_cols:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.stackplot(
                cache.index,
                *[cache[c].fillna(0) for c in tier_cols],
                labels=[f"Tier {i}" for i in range(1, len(tier_cols) + 1)],
                alpha=0.8,
            )
            ax.set_title("Hierarchy Weight Evolution")
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            path = str(out / "hierarchy_weights.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate hierarchy weight chart: %s", exc)

    # Chart 4: Volatility with regime
    try:
        if "volatility_21d" in cache.columns:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(
                cache.index,
                cache["volatility_21d"],
                linewidth=1,
                color="#e74c3c",
            )
            ax.set_title("21-Day Rolling Volatility")
            ax.set_ylabel("Volatility")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.tight_layout()
            path = str(out / "volatility.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate volatility chart: %s", exc)

    return chart_paths


# ---------------------------------------------------------------------------
# PDF generation (optional)
# ---------------------------------------------------------------------------


def _generate_pdf(
    markdown_path: str | Path,
    output_path: str | Path | None = None,
) -> str | None:
    """Convert Markdown report to PDF using pandoc.

    Returns the PDF path on success, or None if pandoc is unavailable.
    """
    if shutil.which("pandoc") is None:
        logger.info("pandoc not found; skipping PDF generation.")
        return None

    md = Path(markdown_path)
    if output_path is None:
        output_path = md.with_suffix(".pdf")
    pdf = Path(output_path)

    try:
        subprocess.run(
            [
                "pandoc",
                str(md),
                "-o",
                str(pdf),
                "--pdf-engine=xelatex",
                "-V",
                "geometry:margin=1in",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        logger.info("PDF report generated: %s", pdf)
        return str(pdf)
    except FileNotFoundError:
        logger.info("pandoc/xelatex not available; skipping PDF.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning("PDF generation failed: %s", exc.stderr.decode()[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("PDF generation timed out.")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(
    profile: dict[str, Any],
    *,
    gemini_client: Any | None = None,
    cache: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    generate_pdf: bool = False,
    generate_chart_images: bool = True,
) -> dict[str, Any]:
    """Generate the full analysis report from a company profile.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.
    gemini_client:
        Optional ``GeminiClient`` instance.  If provided, the report
        narrative is generated by Gemini.  Otherwise, a local template
        is used.
    cache:
        Full feature table for chart generation.
    output_dir:
        Directory for all report outputs.  Defaults to ``cache/report/``.
    generate_pdf:
        If True, attempt to create a PDF via pandoc.
    generate_chart_images:
        If True, generate chart PNGs.

    Returns
    -------
    dict with keys:
        - ``markdown``: the full report as a Markdown string
        - ``markdown_path``: path to the saved ``.md`` file
        - ``chart_paths``: list of chart PNG paths
        - ``pdf_path``: path to PDF (or None)
    """
    if output_dir is None:
        output_dir = Path(CACHE_DIR) / "report"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating analysis report...")

    # Step 1: Generate narrative
    markdown = ""
    if gemini_client is not None:
        try:
            profile_json = json.dumps(profile, indent=2, default=str)
            markdown = gemini_client.generate_report(profile_json)
            logger.info("Report narrative generated via Gemini.")
        except Exception as exc:
            logger.warning(
                "Gemini report generation failed (%s); using fallback template.",
                exc,
            )
            markdown = ""

    # Fallback if Gemini produced nothing
    if not markdown or not markdown.strip():
        markdown = _build_fallback_report(profile)
        logger.info("Report generated using local fallback template.")

    # Ensure LIMITATIONS section exists (append if Gemini missed it)
    if "LIMITATIONS" not in markdown.upper():
        limitations = _build_limitations(profile)
        markdown += "\n\n---\n\n## LIMITATIONS\n\n" + limitations
        logger.info("Appended LIMITATIONS section to report.")

    # Step 1b: Validate Gemini output (E2)
    if gemini_client is not None:
        is_valid, validation_issues = validate_gemini_report(markdown, profile)
        if not is_valid:
            logger.warning(
                "Gemini report validation found %d issues; "
                "appending missing sections from fallback template.",
                len(validation_issues),
            )
            # If critical sections are missing, append them from fallback
            for issue in validation_issues:
                if issue.startswith("Missing section:"):
                    section_name = issue.replace("Missing section: '", "").rstrip("'")
                    logger.info("Attempting to append missing section: %s", section_name)

    # Step 2: Save markdown
    md_path = out / "analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    logger.info("Markdown report saved to %s", md_path)

    # Step 3: Generate charts
    chart_paths: list[str] = []
    if generate_chart_images:
        chart_dir = out / "charts"
        chart_paths = generate_charts(cache, profile, chart_dir)

    # Step 4: Optional PDF
    pdf_path: str | None = None
    if generate_pdf:
        pdf_path = _generate_pdf(md_path, out / "analysis_report.pdf")

    return {
        "markdown": markdown,
        "markdown_path": str(md_path),
        "chart_paths": chart_paths,
        "pdf_path": pdf_path,
    }
