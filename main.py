#!/usr/bin/env python3
"""Operator 1 -- CLI entry point for local execution.

Usage:
    python main.py --isin US0378331005 --symbol AAPL
    python main.py --isin US0378331005 --symbol AAPL --skip-linked --skip-models
    python main.py --help

This script runs the same pipeline as the Kaggle notebook but from a
standard Python environment.  API keys are loaded from a ``.env`` file
in the project root (copy ``.env.example`` to ``.env`` and fill in your keys).

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Early setup: configure logging before any operator1 imports
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("operator1.main")


def main() -> int:
    """Run the Operator 1 pipeline end-to-end."""

    parser = argparse.ArgumentParser(
        description="Operator 1 -- Financial analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --isin US0378331005 --symbol AAPL
  python main.py --isin DE0007164600 --symbol SAP --skip-models
  python main.py --isin US0378331005 --symbol AAPL --report-only
""",
    )
    parser.add_argument(
        "--isin", required=True,
        help="Eulerpool ISIN for the target company (e.g. US0378331005)",
    )
    parser.add_argument(
        "--symbol", required=True,
        help="FMP trading symbol for OHLCV data (e.g. AAPL)",
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Lookback window in years (default: 2.0)",
    )
    parser.add_argument(
        "--skip-linked", action="store_true",
        help="Skip linked entity discovery (faster, target-only analysis)",
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Skip temporal modeling / forecasting (cache + features only)",
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate report from existing cache/profile",
    )
    parser.add_argument(
        "--output-dir", type=str, default="cache",
        help="Output directory for all artifacts (default: cache/)",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Generate PDF report (requires pandoc)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Step 0: Load secrets
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("OPERATOR 1 -- Financial Analysis Pipeline")
    logger.info("=" * 60)
    logger.info("Target ISIN: %s", args.isin)
    logger.info("FMP Symbol: %s", args.symbol)

    try:
        from operator1.secrets_loader import load_secrets
        secrets = load_secrets()
    except SystemExit as exc:
        logger.error("Failed to load API keys: %s", exc)
        logger.error("Create a .env file from .env.example with your API keys")
        return 1

    eulerpool_key = secrets["EULERPOOL_API_KEY"]
    fmp_key = secrets["FMP_API_KEY"]
    gemini_key = secrets["GEMINI_API_KEY"]

    # ------------------------------------------------------------------
    # Step 1: Verify identifiers
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Verifying identifiers...")

    from operator1.clients.eulerpool import EulerportClient
    from operator1.clients.fmp import FMPClient
    from operator1.steps.verify_identifiers import verify_identifiers

    eulerpool_client = EulerportClient(api_key=eulerpool_key)
    fmp_client = FMPClient(api_key=fmp_key)

    try:
        verified = verify_identifiers(
            target_isin=args.isin,
            fmp_symbol=args.symbol,
            eulerpool_client=eulerpool_client,
            fmp_client=fmp_client,
        )
        logger.info("Verification passed: %s (%s)", verified.name or "?", verified.country or "?")
    except Exception as exc:
        logger.error("Identifier verification failed: %s", exc)
        logger.error("Check that your ISIN and FMP symbol are correct.")
        return 1

    # Convert dataclass to dict for downstream compatibility
    from dataclasses import asdict
    target_profile = asdict(verified)

    # ------------------------------------------------------------------
    # Report-only mode
    # ------------------------------------------------------------------
    if args.report_only:
        return _generate_report_only(args, secrets, target_profile)

    # ------------------------------------------------------------------
    # Step 2: Fetch macro data
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Fetching macro data from World Bank...")

    from operator1.clients.world_bank import WorldBankClient
    from operator1.steps.macro_mapping import fetch_macro_data

    wb_client = WorldBankClient()
    country_code = target_profile.get("country", "US")

    try:
        macro_data = fetch_macro_data(
            country_iso2=country_code,
            wb_client=wb_client,
        )
        logger.info("Macro data fetched: %d indicators", len(macro_data) if macro_data is not None else 0)
    except Exception as exc:
        logger.warning("Macro data fetch failed (continuing without): %s", exc)
        macro_data = None

    # Fetch WDS documents for qualitative context
    try:
        wds_context = wb_client.get_country_context(
            country=country_code,
            sector=target_profile.get("sector"),
        )
        logger.info("WDS context: %d documents found", wds_context.get("document_count", 0))
    except Exception as exc:
        logger.warning("WDS document search failed (continuing without): %s", exc)
        wds_context = {}

    # ------------------------------------------------------------------
    # Step 3: Discover linked entities (optional)
    # ------------------------------------------------------------------
    relationships = {}
    if not args.skip_linked:
        logger.info("")
        logger.info("Step 3: Discovering linked entities via Gemini...")

        from operator1.clients.gemini import GeminiClient
        from operator1.steps.entity_discovery import discover_linked_entities

        gemini_client = GeminiClient(api_key=gemini_key)

        try:
            relationships = discover_linked_entities(
                target_profile=target_profile,
                gemini_client=gemini_client,
                eulerpool_client=eulerpool_client,
            )
            total_linked = sum(len(v) for v in relationships.values() if isinstance(v, list))
            logger.info("Linked entities discovered: %d", total_linked)
        except Exception as exc:
            logger.warning("Entity discovery failed (continuing without): %s", exc)
    else:
        logger.info("Step 3: Skipped (--skip-linked)")

    # ------------------------------------------------------------------
    # Step 4: Extract data and build cache
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Building 2-year daily cache...")

    from operator1.steps.data_extraction import extract_target_data
    from operator1.steps.cache_builder import build_daily_cache

    lookback_days = int(args.years * 365)
    date_start = (date.today() - timedelta(days=lookback_days)).isoformat()
    date_end = date.today().isoformat()

    try:
        raw_data = extract_target_data(
            isin=args.isin,
            fmp_symbol=args.symbol,
            eulerpool_client=eulerpool_client,
            fmp_client=fmp_client,
            start_date=date_start,
            end_date=date_end,
        )
        logger.info("Raw data extracted")
    except Exception as exc:
        logger.error("Data extraction failed: %s", exc)
        return 1

    try:
        cache = build_daily_cache(
            target_data=raw_data,
            macro_data=macro_data,
        )
        logger.info("Cache built: %d rows x %d columns", len(cache), len(cache.columns))
    except Exception as exc:
        logger.error("Cache building failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Feature engineering
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Computing derived features...")

    from operator1.features.derived_variables import compute_derived_features
    from operator1.analysis.survival_mode import detect_company_survival_mode
    from operator1.analysis.hierarchy_weights import select_hierarchy_weights

    try:
        cache = compute_derived_features(cache)
        logger.info("Features computed: %d columns", len(cache.columns))
    except Exception as exc:
        logger.warning("Feature engineering partially failed: %s", exc)

    # Survival mode
    try:
        cache["company_survival_mode_flag"] = detect_company_survival_mode(cache)
        weights = select_hierarchy_weights(
            company_survival=bool(cache["company_survival_mode_flag"].iloc[-1]),
            country_survival=False,
            country_protected=False,
            vanity_pct=0.0,
        )
        logger.info("Survival mode: %d days flagged", cache["company_survival_mode_flag"].sum())
    except Exception as exc:
        logger.warning("Survival mode detection failed: %s", exc)
        weights = {f"tier{i}": 20.0 for i in range(1, 6)}

    # ------------------------------------------------------------------
    # Step 6: Temporal modeling (optional)
    # ------------------------------------------------------------------
    forecast_result = None
    forward_pass_result = None
    burnout_result = None

    if not args.skip_models:
        logger.info("")
        logger.info("Step 6: Running temporal models...")

        from operator1.models.regime_detector import detect_regimes
        from operator1.models.forecasting import (
            run_forecasting,
            run_forward_pass,
            run_burnout,
        )
        from operator1.models.monte_carlo import run_monte_carlo
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        # Regime detection
        try:
            cache = detect_regimes(cache)
            logger.info("Regimes detected")
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)

        # Standard forecasting (initial model fitting)
        try:
            cache, forecast_result = run_forecasting(cache)
            logger.info("Forecasting complete")
        except Exception as exc:
            logger.warning("Forecasting failed: %s", exc)

        # Forward pass (D1)
        regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None
        try:
            forward_pass_result = run_forward_pass(
                cache,
                hierarchy_weights=weights,
                regime_labels=regime_labels,
            )
            logger.info("Forward pass complete: %d steps", forward_pass_result.total_days)
        except Exception as exc:
            logger.warning("Forward pass failed: %s", exc)

        # Burn-out (D4)
        try:
            burnout_result = run_burnout(
                cache,
                hierarchy_weights=weights,
                regime_labels=regime_labels,
            )
            logger.info(
                "Burn-out complete: %d iterations, converged=%s",
                burnout_result.iterations_completed,
                burnout_result.converged,
            )
        except Exception as exc:
            logger.warning("Burn-out failed: %s", exc)

        # Monte Carlo
        mc_result = None
        try:
            mc_result = run_monte_carlo(cache)
            logger.info("Monte Carlo simulation complete")
        except Exception as exc:
            logger.warning("Monte Carlo failed: %s", exc)

        # Prediction aggregation
        if forecast_result is not None:
            try:
                pred_result = run_prediction_aggregation(
                    cache, forecast_result, mc_result,
                )
                logger.info("Predictions aggregated: %d variables", len(pred_result.variables_predicted))
            except Exception as exc:
                logger.warning("Prediction aggregation failed: %s", exc)
    else:
        logger.info("Step 6: Skipped (--skip-models)")

    # ------------------------------------------------------------------
    # Step 7: Build company profile
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 7: Building company profile...")

    from operator1.report.profile_builder import build_company_profile

    try:
        profile = build_company_profile(
            cache=cache,
            target_info=target_profile,
            relationships=relationships,
        )
        # Save profile
        profile_path = Path(args.output_dir) / "company_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        logger.info("Profile saved: %s", profile_path)
    except Exception as exc:
        logger.error("Profile building failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 8: Generate report (optional)
    # ------------------------------------------------------------------
    if not args.skip_report:
        logger.info("")
        logger.info("Step 8: Generating report...")

        from operator1.clients.gemini import GeminiClient
        from operator1.report.report_generator import generate_report

        gemini_client = GeminiClient(api_key=gemini_key)

        try:
            report_output = generate_report(
                profile=profile,
                gemini_client=gemini_client,
                cache=cache,
                output_dir=Path(args.output_dir) / "report",
                generate_pdf=args.pdf,
            )
            logger.info("Report saved: %s", report_output.get("markdown_path"))
            if report_output.get("pdf_path"):
                logger.info("PDF saved: %s", report_output["pdf_path"])
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)
    else:
        logger.info("Step 8: Skipped (--skip-report)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Output directory: %s", args.output_dir)

    return 0


def _generate_report_only(args: argparse.Namespace, secrets: dict, target_profile: dict) -> int:
    """Generate report from an existing profile JSON."""
    profile_path = Path(args.output_dir) / "company_profile.json"
    if not profile_path.exists():
        logger.error("No existing profile found at %s. Run full pipeline first.", profile_path)
        return 1

    with open(profile_path, "r", encoding="utf-8") as fh:
        profile = json.load(fh)

    from operator1.clients.gemini import GeminiClient
    from operator1.report.report_generator import generate_report

    gemini_client = GeminiClient(api_key=secrets["GEMINI_API_KEY"])

    report_output = generate_report(
        profile=profile,
        gemini_client=gemini_client,
        output_dir=Path(args.output_dir) / "report",
        generate_pdf=args.pdf,
    )
    logger.info("Report saved: %s", report_output.get("markdown_path"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
