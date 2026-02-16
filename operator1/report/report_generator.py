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

## 3. Financial Health Assessment

{financial_health}

---

## 4. Survival Analysis

{survival_analysis}

---

## 5. Linked Entities & Relative Positioning

{linked_entities}

---

## 6. Macro Environment Impact

{macro_environment}

---

## 7. Regime Analysis & Forecasts

{regime_analysis}

---

## 8. Risk Assessment

{risk_assessment}

---

## 9. LIMITATIONS

{limitations}
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
            h_preds = horizons.get(h_label, [])
            if h_preds:
                lines.append(f"**{h_label} horizon:**")
                for p in h_preds[:5]:  # Top 5 for brevity
                    var = p.get("variable", "?")
                    pf = _fmt(p.get("point_forecast"))
                    ci_lo = _fmt(p.get("lower_ci"))
                    ci_hi = _fmt(p.get("upper_ci"))
                    lines.append(
                        f"- {var}: {pf} [{ci_lo}, {ci_hi}]"
                    )
                lines.append("")
    else:
        lines.append("Forecast data unavailable.")

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


def _build_fallback_report(profile: dict[str, Any]) -> str:
    """Build a report using the local template (no Gemini)."""
    return _FALLBACK_TEMPLATE.format(
        generated_at=profile.get("meta", {}).get(
            "generated_at", datetime.utcnow().isoformat(),
        ),
        executive_summary=_build_executive_summary(profile),
        company_overview=_build_company_overview(profile),
        financial_health=_build_financial_health(profile),
        survival_analysis=_build_survival_analysis(profile),
        linked_entities=_build_linked_entities_section(profile),
        macro_environment=_build_macro_section(profile),
        regime_analysis=_build_regime_analysis(profile),
        risk_assessment=_build_risk_assessment(profile),
        limitations=_build_limitations(profile),
    )


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
