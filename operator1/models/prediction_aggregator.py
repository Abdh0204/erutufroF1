"""T6.4 -- Prediction aggregation with ensemble weighting.

Consumes outputs from T6.1 (regime detection), T6.2 (forecasting models),
and T6.3 (Monte Carlo simulations) to produce unified, multi-horizon
predictions with uncertainty bands and Technical Alpha protection.

**Key features:**

1. **Ensemble weighting**: inverse-RMSE weighting across all models that
   successfully fitted for each variable.  Models with lower validation
   error receive higher weight.

2. **Multi-horizon predictions**: aggregated point forecasts for 1d, 5d,
   21d, and 252d horizons.

3. **Uncertainty bands**: confidence intervals derived from model RMSE
   scaled by the square root of horizon (standard financial assumption),
   optionally widened when Monte Carlo survival probability is low.

4. **Technical Alpha protection**: next-day OHLC predictions are masked
   (set to ``None``) except for the Low estimate, per Sec 17 of the spec.

5. **Persistence**: final predictions are stored in
   ``cache/predictions.parquet`` and a summary JSON in
   ``cache/prediction_summary.json``.

Top-level entry point:
  ``run_prediction_aggregation(cache, forecast_result, mc_result)``

Spec refs: Sec 17
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR
from operator1.models.forecasting import (
    HORIZONS,
    ForecastResult,
    ModelMetrics,
)
from operator1.models.monte_carlo import MonteCarloResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CI z-score for 90% interval (5th to 95th percentile).
Z_SCORE_90: float = 1.645

# Default confidence level.
DEFAULT_CONFIDENCE_LEVEL: float = 0.90

# Survival risk multiplier: how much to widen bands when survival is low.
DEFAULT_SURVIVAL_RISK_MULTIPLIER: float = 2.0

# Intraday low estimation factor (multiples of daily vol below last close).
INTRADAY_LOW_FACTOR: float = 1.5

# Minimum RMSE to avoid division by zero in inverse weighting.
MIN_RMSE_FOR_WEIGHTING: float = 1e-10

# Default volatility when none is available in the cache.
DEFAULT_VOLATILITY: float = 0.02


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class HorizonPrediction:
    """Single prediction for one variable at one horizon."""

    variable: str = ""
    horizon: str = ""  # "1d", "5d", "21d", "252d"
    point_forecast: float = float("nan")
    lower_ci: float = float("nan")  # 5th percentile
    upper_ci: float = float("nan")  # 95th percentile
    confidence: float = float("nan")  # 0-1 overall confidence score
    model_used: str = ""  # which model produced the forecast
    ensemble_weight: float = 0.0  # weight this model got
    survival_adjusted: bool = False  # whether survival prob was factored in


@dataclass
class TechnicalAlphaMask:
    """Masked OHLC predictions for Technical Alpha protection.

    Per Sec 17: mask next-day OHLC except Low.
    """

    next_day_open: float | None = None  # MASKED
    next_day_high: float | None = None  # MASKED
    next_day_low: float = float("nan")  # VISIBLE
    next_day_close: float | None = None  # MASKED
    mask_applied: bool = True


@dataclass
class PredictionAggregatorResult:
    """Container for all prediction aggregation outputs."""

    # Per-variable, per-horizon predictions.
    # {variable: {horizon: HorizonPrediction}}
    predictions: dict[str, dict[str, HorizonPrediction]] = field(
        default_factory=dict,
    )

    # Technical Alpha masked OHLC.
    technical_alpha: TechnicalAlphaMask = field(
        default_factory=TechnicalAlphaMask,
    )

    # Ensemble weights per model type.
    # {model_name: weight}
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    # Model availability.
    n_models_available: int = 0
    n_models_failed: int = 0

    # Survival probability summary (pass-through from MC).
    survival_probability_mean: float = float("nan")
    survival_probability_p5: float = float("nan")
    survival_probability_p95: float = float("nan")

    # Current regime at prediction time.
    current_regime: str = ""

    # Metadata.
    prediction_date: str = ""
    variables_predicted: list[str] = field(default_factory=list)
    horizons: list[str] = field(default_factory=list)

    # Error info.
    error: str | None = None
    fitted: bool = False


# ---------------------------------------------------------------------------
# Ensemble weight computation
# ---------------------------------------------------------------------------


def compute_ensemble_weights(
    metrics: list[ModelMetrics],
) -> dict[str, float]:
    """Compute inverse-RMSE ensemble weights from model metrics.

    Parameters
    ----------
    metrics:
        List of ``ModelMetrics`` from ``ForecastResult.metrics``.
        Only entries with ``fitted=True`` and finite RMSE are used.

    Returns
    -------
    Dict mapping ``model_name`` to normalised weight (sum = 1.0).
    Models with lower RMSE receive higher weight.
    """
    # Collect best RMSE per unique model name.
    model_rmse: dict[str, float] = {}

    for m in metrics:
        if not m.fitted:
            continue
        if math.isnan(m.rmse) or m.rmse < MIN_RMSE_FOR_WEIGHTING:
            continue

        name = m.model_name
        if name not in model_rmse or m.rmse < model_rmse[name]:
            model_rmse[name] = m.rmse

    if not model_rmse:
        # No valid RMSE available -- return empty (caller handles this).
        return {}

    # Inverse-RMSE weights.
    inv_rmse = {name: 1.0 / rmse for name, rmse in model_rmse.items()}
    total = sum(inv_rmse.values())

    if total <= 0:
        # Uniform fallback.
        n = len(inv_rmse)
        return {name: 1.0 / n for name in inv_rmse}

    return {name: w / total for name, w in inv_rmse.items()}


# ---------------------------------------------------------------------------
# Forecast aggregation
# ---------------------------------------------------------------------------


def aggregate_forecasts(
    forecast_result: ForecastResult,
) -> dict[str, dict[str, float]]:
    """Extract aggregated point forecasts per variable per horizon.

    In the current architecture, ``ForecastResult.forecasts`` already
    contains the best model's prediction for each variable (selected by
    the fallback chain in ``run_forecasting``).  This function passes
    them through with validation.

    Parameters
    ----------
    forecast_result:
        Output from ``run_forecasting``.

    Returns
    -------
    ``{variable: {horizon_label: point_forecast}}``
    """
    aggregated: dict[str, dict[str, float]] = {}

    for var_name, horizons in forecast_result.forecasts.items():
        var_forecasts: dict[str, float] = {}

        for h_label, value in horizons.items():
            if not math.isnan(value):
                var_forecasts[h_label] = value

        if var_forecasts:
            aggregated[var_name] = var_forecasts

    return aggregated


# ---------------------------------------------------------------------------
# Uncertainty bands
# ---------------------------------------------------------------------------


def _get_best_rmse_for_variable(
    variable: str,
    metrics: list[ModelMetrics],
) -> float:
    """Return the RMSE of the best fitted model for a variable."""
    best = float("nan")

    for m in metrics:
        if m.variable == variable and m.fitted and not math.isnan(m.rmse):
            if math.isnan(best) or m.rmse < best:
                best = m.rmse

    return best


def compute_uncertainty_bands(
    point_forecast: float,
    rmse: float,
    horizon_days: int,
    *,
    survival_probability: float = 1.0,
    survival_risk_multiplier: float = DEFAULT_SURVIVAL_RISK_MULTIPLIER,
    z_score: float = Z_SCORE_90,
) -> tuple[float, float]:
    """Compute confidence interval bounds for a single prediction.

    The base interval is derived from the model's RMSE scaled by
    ``sqrt(horizon_days)`` (random-walk scaling).  If the Monte Carlo
    survival probability is below 1.0, the interval is widened by a
    risk factor proportional to the survival shortfall.

    Parameters
    ----------
    point_forecast:
        Central prediction value.
    rmse:
        Best model's root mean squared error on validation data.
    horizon_days:
        Forecast horizon in business days.
    survival_probability:
        Monte Carlo survival probability for this horizon (0-1).
        Defaults to 1.0 (no widening).
    survival_risk_multiplier:
        How aggressively to widen bands when survival prob is low.
    z_score:
        z-score for the desired confidence level (default 1.645 = 90%).

    Returns
    -------
    (lower_ci, upper_ci)
    """
    if math.isnan(point_forecast):
        return float("nan"), float("nan")

    if math.isnan(rmse) or rmse < MIN_RMSE_FOR_WEIGHTING:
        # No valid RMSE -- use a conservative 10% of point forecast.
        base_spread = abs(point_forecast) * 0.10
    else:
        base_spread = z_score * rmse * math.sqrt(max(horizon_days, 1))

    # Survival-weighted widening.
    surv_prob = max(0.0, min(1.0, survival_probability))
    risk_factor = 1.0 + (1.0 - surv_prob) * survival_risk_multiplier

    adjusted_spread = base_spread * risk_factor

    lower_ci = point_forecast - adjusted_spread
    upper_ci = point_forecast + adjusted_spread

    return lower_ci, upper_ci


def compute_confidence_score(
    rmse: float,
    rmse_reference: float,
    survival_probability: float = 1.0,
) -> float:
    """Compute a 0-1 confidence score for a prediction.

    Combines model quality (normalised RMSE) with survival probability.

    Parameters
    ----------
    rmse:
        Model's RMSE on validation data.
    rmse_reference:
        A reference RMSE to normalise against (e.g. median RMSE across
        all models/variables).  If 0, model quality = 0.5 default.
    survival_probability:
        Monte Carlo survival probability (0-1).

    Returns
    -------
    Confidence score in [0, 1].
    """
    # Model quality score: 1.0 when RMSE is 0, decays as RMSE grows.
    if math.isnan(rmse) or math.isnan(rmse_reference):
        model_quality = 0.5  # default when no info
    elif rmse_reference <= MIN_RMSE_FOR_WEIGHTING:
        model_quality = 0.5
    else:
        # Exponential decay: quality = exp(-rmse / reference).
        model_quality = math.exp(-rmse / rmse_reference)
        model_quality = max(0.0, min(1.0, model_quality))

    surv_prob = max(0.0, min(1.0, survival_probability))
    if math.isnan(surv_prob):
        surv_prob = 1.0

    # Geometric mean of model quality and survival probability.
    confidence = math.sqrt(model_quality * surv_prob)

    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Technical Alpha masking
# ---------------------------------------------------------------------------


def apply_technical_alpha_mask(
    cache: pd.DataFrame,
    forecasts: dict[str, dict[str, float]],
) -> TechnicalAlphaMask:
    """Apply Technical Alpha protection to next-day OHLC predictions.

    Per Sec 17: mask next-day OHLC except Low.  The Low estimate is
    derived from the last observed close minus a volatility-based buffer.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with ``close`` and optionally
        ``volatility_21d`` columns.
    forecasts:
        Aggregated forecasts dict (may contain ``close`` 1d forecast).

    Returns
    -------
    ``TechnicalAlphaMask`` with open/high/close masked (None) and
    low set to an estimated value.
    """
    mask = TechnicalAlphaMask(mask_applied=True)

    # Get last observed close.
    if "close" in cache.columns:
        close_series = cache["close"].dropna()
        if len(close_series) > 0:
            last_close = float(close_series.iloc[-1])
        else:
            last_close = float("nan")
    else:
        last_close = float("nan")

    if math.isnan(last_close):
        # Cannot estimate -- return mask with NaN low.
        logger.warning(
            "No close price available for Technical Alpha mask"
        )
        return mask

    # Get volatility estimate.
    vol = DEFAULT_VOLATILITY
    if "volatility_21d" in cache.columns:
        vol_series = cache["volatility_21d"].dropna()
        if len(vol_series) > 0:
            vol_val = float(vol_series.iloc[-1])
            if not math.isnan(vol_val) and vol_val > 0:
                vol = vol_val

    # Estimate next-day low.
    # Low is typically last_close minus some fraction of daily vol.
    mask.next_day_low = last_close * (1.0 - vol * INTRADAY_LOW_FACTOR)

    # Masked fields stay None.
    mask.next_day_open = None
    mask.next_day_high = None
    mask.next_day_close = None

    logger.info(
        "Technical Alpha mask: last_close=%.4f, vol=%.6f, "
        "estimated_low=%.4f",
        last_close,
        vol,
        mask.next_day_low,
    )

    return mask


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_predictions(
    result: PredictionAggregatorResult,
    cache_dir: str = CACHE_DIR,
) -> tuple[Path, Path]:
    """Save predictions to parquet and summary to JSON.

    Parameters
    ----------
    result:
        The aggregation result to persist.
    cache_dir:
        Directory to write files into.

    Returns
    -------
    (parquet_path, json_path)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # -- Parquet: flattened predictions table --
    rows: list[dict[str, Any]] = []
    for var_name, horizons in result.predictions.items():
        for h_label, pred in horizons.items():
            rows.append({
                "variable": pred.variable,
                "horizon": pred.horizon,
                "point_forecast": pred.point_forecast,
                "lower_ci": pred.lower_ci,
                "upper_ci": pred.upper_ci,
                "confidence": pred.confidence,
                "model_used": pred.model_used,
                "ensemble_weight": pred.ensemble_weight,
                "survival_adjusted": pred.survival_adjusted,
            })

    parquet_path = cache_path / "predictions.parquet"
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)
    else:
        # Write empty parquet with correct schema.
        df = pd.DataFrame(columns=[
            "variable", "horizon", "point_forecast", "lower_ci",
            "upper_ci", "confidence", "model_used", "ensemble_weight",
            "survival_adjusted",
        ])
        df.to_parquet(parquet_path, index=False)

    logger.info("Saved predictions to %s (%d rows)", parquet_path, len(rows))

    # -- JSON: summary metadata --
    ta = result.technical_alpha
    summary = {
        "prediction_date": result.prediction_date,
        "current_regime": result.current_regime,
        "variables_predicted": result.variables_predicted,
        "horizons": result.horizons,
        "n_models_available": result.n_models_available,
        "n_models_failed": result.n_models_failed,
        "ensemble_weights": result.ensemble_weights,
        "survival_probability_mean": _safe_float(
            result.survival_probability_mean,
        ),
        "survival_probability_p5": _safe_float(
            result.survival_probability_p5,
        ),
        "survival_probability_p95": _safe_float(
            result.survival_probability_p95,
        ),
        "technical_alpha": {
            "mask_applied": ta.mask_applied,
            "next_day_low": _safe_float(ta.next_day_low),
            "next_day_open": ta.next_day_open,
            "next_day_high": ta.next_day_high,
            "next_day_close": ta.next_day_close,
        },
        "fitted": result.fitted,
        "error": result.error,
    }

    json_path = cache_path / "prediction_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved prediction summary to %s", json_path)

    return parquet_path, json_path


def _safe_float(val: float) -> float | None:
    """Convert NaN to None for JSON serialisation."""
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def run_prediction_aggregation(
    cache: pd.DataFrame,
    forecast_result: ForecastResult,
    mc_result: MonteCarloResult | None = None,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    survival_risk_multiplier: float = DEFAULT_SURVIVAL_RISK_MULTIPLIER,
    save_to_cache: bool = True,
    cache_dir: str = CACHE_DIR,
) -> PredictionAggregatorResult:
    """Run the full prediction aggregation pipeline.

    Combines forecasting results with Monte Carlo survival estimates
    to produce final predictions with uncertainty bands and Technical
    Alpha protection.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (with ``close``, ``volatility_21d``,
        ``regime_label`` columns expected).
    forecast_result:
        Output from ``run_forecasting`` (T6.2).
    mc_result:
        Output from ``run_monte_carlo`` (T6.3).  Optional -- if
        ``None``, survival adjustment is skipped.
    confidence_level:
        Desired confidence level for uncertainty bands (default 0.90).
    survival_risk_multiplier:
        How aggressively to widen bands when survival prob is low.
    save_to_cache:
        If ``True``, write predictions.parquet and prediction_summary.json.
    cache_dir:
        Directory for cache files.

    Returns
    -------
    ``PredictionAggregatorResult`` with all predictions, TA mask, and
    metadata.
    """
    logger.info("Starting prediction aggregation pipeline...")

    result = PredictionAggregatorResult()
    result.prediction_date = date.today().isoformat()
    result.horizons = sorted(HORIZONS.keys(), key=lambda h: HORIZONS[h])

    # ------------------------------------------------------------------
    # Current regime
    # ------------------------------------------------------------------
    if "regime_label" in cache.columns:
        labels = cache["regime_label"].dropna()
        if len(labels) > 0:
            result.current_regime = str(labels.iloc[-1])
        else:
            result.current_regime = "unknown"
    else:
        result.current_regime = "unknown"

    # ------------------------------------------------------------------
    # Ensemble weights
    # ------------------------------------------------------------------
    result.ensemble_weights = compute_ensemble_weights(
        forecast_result.metrics,
    )

    # Count model availability.
    failed_flags = [
        forecast_result.model_failed_kalman,
        forecast_result.model_failed_garch,
        forecast_result.model_failed_var,
        forecast_result.model_failed_lstm,
        forecast_result.model_failed_tree,
    ]
    result.n_models_failed = sum(failed_flags)
    result.n_models_available = 5 - result.n_models_failed  # 5 model types

    # ------------------------------------------------------------------
    # Aggregate forecasts
    # ------------------------------------------------------------------
    aggregated = aggregate_forecasts(forecast_result)

    if not aggregated:
        result.error = "No forecasts available to aggregate"
        logger.warning(result.error)
        # Still apply TA mask and save.
        result.technical_alpha = apply_technical_alpha_mask(cache, {})
        if save_to_cache:
            save_predictions(result, cache_dir)
        return result

    # ------------------------------------------------------------------
    # Survival probabilities from Monte Carlo
    # ------------------------------------------------------------------
    mc_survival_by_horizon: dict[str, float] = {}
    if mc_result is not None and mc_result.fitted:
        mc_survival_by_horizon = mc_result.survival_probability
        result.survival_probability_mean = mc_result.survival_probability_mean
        result.survival_probability_p5 = mc_result.survival_probability_p5
        result.survival_probability_p95 = mc_result.survival_probability_p95

    # ------------------------------------------------------------------
    # Reference RMSE for confidence scoring
    # ------------------------------------------------------------------
    valid_rmse_values = [
        m.rmse
        for m in forecast_result.metrics
        if m.fitted and not math.isnan(m.rmse)
    ]
    if valid_rmse_values:
        rmse_reference = float(np.median(valid_rmse_values))
    else:
        rmse_reference = float("nan")

    # ------------------------------------------------------------------
    # Z-score for requested confidence level
    # ------------------------------------------------------------------
    # Map common levels; fallback to 1.645 for 90%.
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576, 0.80: 1.282}
    z_score = z_map.get(confidence_level, Z_SCORE_90)

    # ------------------------------------------------------------------
    # Build predictions per variable per horizon
    # ------------------------------------------------------------------
    for var_name, var_forecasts in aggregated.items():
        var_rmse = _get_best_rmse_for_variable(
            var_name, forecast_result.metrics,
        )
        model_name = forecast_result.model_used.get(var_name, "unknown")
        model_weight = result.ensemble_weights.get(model_name, 0.0)

        horizon_preds: dict[str, HorizonPrediction] = {}

        for h_label in result.horizons:
            point = var_forecasts.get(h_label, float("nan"))
            horizon_days = HORIZONS.get(h_label, 1)

            # Survival probability for this horizon.
            surv_prob = mc_survival_by_horizon.get(h_label, 1.0)
            survival_adjusted = surv_prob < 1.0

            # Uncertainty bands.
            lower, upper = compute_uncertainty_bands(
                point,
                var_rmse,
                horizon_days,
                survival_probability=surv_prob,
                survival_risk_multiplier=survival_risk_multiplier,
                z_score=z_score,
            )

            # Confidence score.
            confidence = compute_confidence_score(
                var_rmse, rmse_reference, surv_prob,
            )

            pred = HorizonPrediction(
                variable=var_name,
                horizon=h_label,
                point_forecast=point,
                lower_ci=lower,
                upper_ci=upper,
                confidence=confidence,
                model_used=model_name,
                ensemble_weight=model_weight,
                survival_adjusted=survival_adjusted,
            )
            horizon_preds[h_label] = pred

        result.predictions[var_name] = horizon_preds

    result.variables_predicted = sorted(result.predictions.keys())

    # ------------------------------------------------------------------
    # Technical Alpha mask
    # ------------------------------------------------------------------
    result.technical_alpha = apply_technical_alpha_mask(cache, aggregated)

    # ------------------------------------------------------------------
    # Mark as fitted
    # ------------------------------------------------------------------
    result.fitted = True

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if save_to_cache:
        try:
            save_predictions(result, cache_dir)
        except Exception as exc:
            logger.warning("Failed to save predictions: %s", exc)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    logger.info(
        "Prediction aggregation complete: %d variables, %d horizons, "
        "%d models available (%d failed), "
        "survival_mean=%.4f, regime='%s'",
        len(result.variables_predicted),
        len(result.horizons),
        result.n_models_available,
        result.n_models_failed,
        result.survival_probability_mean
        if not math.isnan(result.survival_probability_mean)
        else -1.0,
        result.current_regime,
    )

    return result
