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
    # Report generation (Sec 18)
    # ------------------------------------------------------------------

    _REPORT_PROMPT = """\
You are a senior financial analyst at a Bloomberg-style research firm.
Generate a comprehensive company analysis report in Markdown format
based on the following company profile data.

{profile_json}

The report MUST include:
1. Executive Summary
2. Company Overview
3. Financial Health Assessment (liquidity, solvency, profitability)
4. Survival Analysis (if applicable)
5. Linked Entities & Relative Positioning
6. Macro Environment Impact
7. Regime Analysis & Forecasts (if available)
8. Risk Assessment
9. **LIMITATIONS** -- a short, plain-language section covering:
   - Data window and frequency limitations
   - OHLCV source caveats
   - Macro data frequency and alignment
   - Data missingness summary
   - Any failed modules and mitigations applied

Write for a professional but non-technical audience. Be factual and
cite specific numbers from the data. Use Markdown formatting with
headers, bullet points, and bold text for emphasis.
"""

    def generate_report(self, company_profile_json: str) -> str:
        """Generate a Bloomberg-style analysis report from profile data.

        Parameters
        ----------
        company_profile_json:
            JSON string of the full company profile.

        Returns
        -------
        Markdown report string, or a fallback message on failure.
        """
        try:
            prompt = self._REPORT_PROMPT.format(
                profile_json=company_profile_json,
            )
            return self._generate(prompt)
        except Exception as exc:
            logger.error("Gemini report generation failed: %s", exc)
            return (
                "# Report Generation Failed\n\n"
                "The automated report could not be generated. "
                "Please review the raw data in the cache artifacts.\n\n"
                f"Error: {exc}\n"
            )
