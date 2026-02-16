"""Google Gemini API client.

Used for linked entity discovery, World Bank mapping suggestions,
and report generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from operator1.constants import GEMINI_BASE_URL
from operator1.config_loader import get_global_config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around the Gemini generative-language API.

    All methods are wrapped in try/except and return sensible fallbacks
    on failure -- Gemini is never a hard dependency for the pipeline.

    Parameters
    ----------
    api_key:
        Gemini API key (from secrets).
    base_url:
        Override for testing.
    model:
        Model name to use for generation.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = GEMINI_BASE_URL,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        cfg = get_global_config()
        timeout = cfg.get("timeout_s", 30)

        url = (
            f"{self._base_url}/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        # Navigate Gemini response structure
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Best-effort extraction of JSON from an LLM response.

        Handles responses that wrap JSON in markdown code fences.
        """
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini JSON response; returning raw text")
            return None

    # ------------------------------------------------------------------
    # Linked entity discovery (Sec 5)
    # ------------------------------------------------------------------

    _LINKED_ENTITIES_PROMPT = """\
You are a financial analyst. Given the following company profile, suggest
related entities grouped by relationship type.

Company profile:
{profile_json}

Sector hints: {sector_hints}

Return a JSON object with these keys, each containing a list of company
names (not tickers) that are publicly traded:
- competitors: direct competitors in the same industry
- suppliers: known major suppliers
- customers: known major customers
- financial_institutions: primary banks or lenders
- logistics: key logistics or distribution partners
- regulators: relevant regulatory bodies (if publicly listed)

Only include companies you are reasonably confident about.
Return valid JSON only, no markdown.
"""

    def propose_linked_entities(
        self,
        target_profile: dict[str, Any],
        sector_hints: str = "",
    ) -> dict[str, list[str]]:
        """Ask Gemini to propose linked entities for a target company.

        Returns dict mapping relationship_group -> list of company names.
        Returns empty dict on any failure.
        """
        try:
            prompt = self._LINKED_ENTITIES_PROMPT.format(
                profile_json=json.dumps(target_profile, indent=2),
                sector_hints=sector_hints or "none",
            )
            text = self._generate(prompt)
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                # Ensure values are lists of strings
                return {
                    k: [str(v) for v in vs]
                    for k, vs in parsed.items()
                    if isinstance(vs, list)
                }
            return {}
        except Exception as exc:
            logger.warning("Gemini linked-entity proposal failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # World Bank mapping suggestions (Sec 4)
    # ------------------------------------------------------------------

    _WB_MAPPING_PROMPT = """\
You are a macroeconomic data analyst. For the country "{country}" and
sector "{sector}", suggest the most relevant World Bank indicator codes
for these canonical variables:

- inflation_rate_yoy
- cpi_index
- unemployment_rate
- gdp_growth
- gdp_current_usd
- official_exchange_rate_lcu_per_usd
- current_account_balance_pct_gdp
- reserves_months_of_imports
- real_interest_rate
- lending_interest_rate
- deposit_interest_rate

Return a JSON object mapping each variable name to a World Bank indicator
code string (e.g. "FP.CPI.TOTL.ZG"). If unsure, omit the key.
Return valid JSON only, no markdown.
"""

    def propose_world_bank_mappings(
        self,
        country: str,
        sector: str = "",
    ) -> dict[str, str]:
        """Ask Gemini to suggest World Bank indicator mappings.

        These are *suggestions only* -- not used at runtime without
        human review.  Returns empty dict on failure.
        """
        try:
            prompt = self._WB_MAPPING_PROMPT.format(
                country=country,
                sector=sector or "general",
            )
            text = self._generate(prompt)
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
            return {}
        except Exception as exc:
            logger.warning("Gemini WB mapping proposal failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Report generation (Sec 18) -- Full 13-section prompt (Phase E1)
    # ------------------------------------------------------------------

    _REPORT_PROMPT = """\
You are a Bloomberg-style financial analyst specializing in comprehensive equity research.

You have been provided with a complete company profile that includes:
- 2 years of historical financial and market data
- Advanced temporal analysis using 25+ mathematical models (HMM, Kalman, \
GARCH, VAR, LSTM with MC Dropout, Temporal Fusion Transformer, Random Forest, XGBoost, etc.)
- Survival mode analysis (company + country)
- Ethical filter assessments (Purchasing Power, Solvency, Gharar, Cash is King)
- Multi-horizon predictions with Conformal Prediction intervals (distribution-free, \
guaranteed coverage -- not Gaussian assumptions)
- SHAP feature attribution explaining what drove each prediction
- MC Dropout epistemic uncertainty (separates "model does not know" from "inherent randomness")
- Dynamic Time Warping (DTW) historical analogs ("the last time this pattern occurred was...")
- Regime detection and structural break analysis
- Linked variables (sector, industry, competitors, macro from World Bank)

Your task is to generate a professional investment report that synthesizes \
this information into actionable insights for sophisticated investors.

---

REPORT STRUCTURE (MUST INCLUDE ALL 13 SECTIONS):

1. EXECUTIVE SUMMARY
   - 3 bullet points summarizing key findings
   - Clear investment recommendation: BUY / HOLD / SELL with confidence level (High/Medium/Low)
   - 12-month target price with rationale

2. COMPANY OVERVIEW
   - Company identity and classification
   - Current market position and capitalization
   - Sector and industry context

3. HISTORICAL PERFORMANCE ANALYSIS (2 Years)
   - Total return vs real return (Purchasing Power filter applied)
   - Risk-adjusted performance: Sharpe ratio, maximum drawdown
   - Regime breakdown: time spent in bull/bear/high-volatility regimes
   - Structural breaks and major market events detected
   - Up days vs down days distribution

4. CURRENT FINANCIAL HEALTH (Tier-by-Tier Breakdown)

   **Explain the Tier Hierarchy:**
   - Why the 5-tier system matters
   - How weights change in different survival regimes
   - Current hierarchy weights and what they mean

   **Tier 1: Liquidity & Cash** -- Cash and equivalents, cash ratio, free cash flow; Cash is King filter results
   **Tier 2: Solvency & Debt** -- Debt-to-equity, net debt to EBITDA, interest coverage; Solvency filter results
   **Tier 3: Market Stability** -- Volatility, drawdown, volume; Gharar filter results
   **Tier 4: Profitability** -- Margins (gross, operating, net), ROE, ROA
   **Tier 5: Growth & Valuation** -- Revenue/earnings growth, P/E, EV/EBITDA

5. SURVIVAL MODE ANALYSIS
   - Current survival status (company, country, protection)
   - Historical survival episodes (count and duration)
   - Vanity expenditure analysis and interpretation
   - What vanity spending reveals about management discipline

6. LINKED VARIABLES & MARKET CONTEXT
   - Sector performance: relative strength vs sector median
   - Industry positioning: valuation premium/discount vs industry
   - Competitor health assessment: how does the company compare?
   - Supply chain risk analysis (if applicable)
   - Macro environment from World Bank indicators (inflation, GDP, unemployment, FX)

7. TEMPORAL ANALYSIS & MODEL INSIGHTS
   - Current market regime and regime distribution over 2 years
   - Regime transitions and what they signal
   - Structural breaks detected
   - Model performance summary by tier (accuracy percentages)
   - Best performing module (from 23+ model ensemble including Kalman, GARCH, \
VAR, LSTM, Temporal Fusion Transformer, tree ensembles)
   - Conformal prediction calibration quality (empirical coverage vs target)
   - SHAP global feature importance: which variables matter most across all predictions
   - Confidence levels in predictions

8. PREDICTIONS & FORECASTS
   **Next Day:** NOTE: Due to Technical Alpha protection, only Low price \
is shown for next-day OHLC. Include Tier 1-5 variable predictions with \
Conformal Prediction intervals (distribution-free, guaranteed 90% coverage). \
Survival probability for next day.
   **Next Week:** Expected return and volatility, full OHLC candlestick series, predicted technical patterns.
   **Next Month:** Price target range (conformal 5th-95th percentile bands), key events to watch, regime shift predictions.
   **Next Year:** Annual outlook with widening uncertainty, predicted regime changes by quarter, long-term trajectory.
   **Monte Carlo Uncertainty:** Tail risk scenarios (worst 5%), base case (50th percentile), upside scenarios (top 5%).
   **SHAP Feature Drivers:** For key predictions, explain WHAT drove the forecast \
(e.g. "next-day cash_ratio prediction driven primarily by: +0.03 from declining \
short-term debt, -0.01 from rising volatility"). Use the SHAP narratives from the profile data.
   **Historical Analogs (DTW):** If available, describe the closest historical analog \
periods found via Dynamic Time Warping. Example: "The last time this company showed \
a similar pattern of rising debt + falling margins + high macro stress was [date]. \
In the following month, the stock [outcome]." Include the empirical return distribution \
from analog outcomes.
   **Epistemic vs Aleatoric Uncertainty:** Where MC Dropout data is available, \
distinguish between model uncertainty ("the model is unsure about this prediction") \
and inherent randomness ("even a perfect model would see variance here"). This helps \
investors understand the *quality* of each prediction.

9. TECHNICAL PATTERNS & CHART ANALYSIS
   - Describe the 2-year price chart with regime shading
   - Historical candlestick patterns detected (last 6 months)
   - Predicted patterns for next week/month
   - Support and resistance levels

10. ETHICAL FILTER ASSESSMENT
    **Purchasing Power Filter:** Verdict, nominal vs real return, inflation impact.
    **Solvency Filter:** Verdict, debt-to-equity ratio, threshold.
    **Gharar Filter (Uncertainty/Speculation):** Verdict, volatility, stability score.
    **Cash is King Filter:** Verdict, FCF yield.
    **Overall Ethical Score:** Combine all filters; is this investment suitable for ethical/Islamic investors?
    Universal lessons: Why these filters matter for ALL investors.

11. RISK FACTORS & LIMITATIONS
    - Model assumptions and their limitations
    - Key risks: company-specific, industry/sector, macro/country
    - Scenarios that could invalidate predictions
    - Black swan events not captured by models
    11.1 LIMITATIONS (SHORT, REQUIRED):
    Provide 5-10 bullets covering: data window, OHLCV source (FMP) caveats, \
macro frequency reality (World Bank often annual, aligned daily via as-of logic), \
missingness summary, any modules that failed and how the report compensated. \
Must be easy for a non-technical client to understand.

12. INVESTMENT RECOMMENDATION
    **Recommendation:** [BUY / HOLD / SELL]
    **Confidence Level:** [High / Medium / Low]
    **12-Month Target Price:** with rationale
    **Key Catalysts to Watch:** events or metrics that would change the recommendation
    **Entry Strategy:** recommended entry price or conditions
    **Exit Strategy:** price targets for profits and stop-loss levels
    **Position Sizing:** suggested portfolio allocation based on risk profile

13. APPENDIX
    - Methodology summary: all 23+ temporal modules used:
      * Regime Detection: HMM, GMM, PELT, Bayesian Change Point
      * Forecasting: Adaptive Kalman, GARCH, VAR, LSTM, Temporal Fusion Transformer (TFT)
      * Tree Ensembles: Random Forest, XGBoost, Gradient Boosting
      * Causality: Granger Causality, Transfer Entropy, Copula Models
      * Uncertainty: Conformal Prediction (distribution-free intervals), \
MC Dropout (epistemic uncertainty), Regime-Aware Monte Carlo, Importance Sampling
      * Explainability: SHAP (per-prediction feature attribution), Sobol Sensitivity
      * Historical Analogs: Dynamic Time Warping (DTW) for finding similar past periods
      * Optimisation: Genetic Algorithm for ensemble weights
      * Pattern Recognition: Candlestick Detector, Wavelet/Fourier Decomposition
    - Forward pass + burn-out process explanation (day-by-day predict-compare-update)
    - Conformal Prediction explanation: why distribution-free intervals are more \
reliable than Gaussian assumptions for financial data
    - SHAP explanation: how per-prediction drivers are computed
    - Ensemble weighting approach (inverse-RMSE + GA optimisation)
    - Variable tier definitions (Tier 1-5 explained)
    - Glossary of technical terms (including Conformal Prediction, SHAP, TFT, \
regime, structural break, nonconformity score)
    - Data sources: Eulerpool (financials), FMP (OHLCV), World Bank (macro + WDS documents), Gemini (relationships)
    - Data timestamps and coverage
    - Disclaimer and limitations

---

FORMATTING REQUIREMENTS:
- Use markdown formatting with clear section headers (##, ###)
- Use tables for financial data where appropriate
- Use bullet points and numbered lists for clarity
- Bold key metrics and verdicts
- Italicize interpretive commentary
- Include placeholders for charts: [CHART: Description]
- Keep language professional but accessible
- Explain technical concepts when first introduced
- Total length: aim for 8,000-12,000 words (comprehensive but readable)

---

COMPLETE COMPANY PROFILE DATA:
{profile_json}

---

Generate the complete Bloomberg-style investment report now.
"""

    def generate_report(
        self,
        company_profile_json: str,
        *,
        max_output_tokens: int = 16000,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str:
        """Generate a Bloomberg-style analysis report from profile data.

        Parameters
        ----------
        company_profile_json:
            JSON string of the full company profile.
        max_output_tokens:
            Maximum tokens for the Gemini response (default 16000 for
            comprehensive reports).
        temperature:
            Sampling temperature (lower = more factual).
        timeout:
            Request timeout in seconds (report generation is slow).

        Returns
        -------
        Markdown report string, or a fallback message on failure.
        """
        try:
            prompt = self._REPORT_PROMPT.format(
                profile_json=company_profile_json,
            )
            url = (
                f"{self._base_url}/models/{self._model}:generateContent"
                f"?key={self._api_key}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            }
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "")

        except Exception as exc:
            logger.error("Gemini report generation failed: %s", exc)
            return (
                "# Report Generation Failed\n\n"
                "The automated report could not be generated. "
                "Please review the raw data in the cache artifacts.\n\n"
                f"Error: {exc}\n"
            )
