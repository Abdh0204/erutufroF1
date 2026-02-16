"""T6.2 -- Forecasting models with fallback chains.

Provides a suite of time-series forecasting models for predicting
financial variables at multiple horizons.  Each model is wrapped in
``try/except`` so that a missing optional dependency logs a warning
and falls through to the next model in the chain.

**Model hierarchy (in order of attempted fit):**

1. **Kalman filter** -- for Tier 1-2 liquidity/solvency variables.
   Uses a local-level state-space model (statsmodels).
2. **GARCH** -- for volatility forecasting (``arch`` library).
3. **VAR** -- multivariate vector autoregression (statsmodels).
   Fallback: univariate AR(1).
4. **LSTM** -- nonlinear sequence model (PyTorch).
   Fallback: GradientBoosting or LinearRegression.
5. **RF / GBM / XGB** -- tree ensemble for tabular features
   (sklearn / xgboost).
6. **Baseline** -- last-value carry-forward or exponential moving
   average.  Always succeeds.

Each model produces:
  - Point forecast per horizon (1d, 5d, 21d, 252d).
  - ``model_failed_<name>`` flag (bool) if the model could not fit.
  - Error metrics (MAE, RMSE) on a held-out validation fold.

Top-level entry point:
  ``run_forecasting(cache, tier_variables, regime_labels)``

Spec refs: Sec 17
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from operator1.config_loader import load_config
from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forecast horizons (business days).
HORIZONS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "21d": 21,
    "252d": 252,
}

# Minimum observations required for each model type.
_MIN_OBS_KALMAN: int = 30
_MIN_OBS_GARCH: int = 60
_MIN_OBS_VAR: int = 50
_MIN_OBS_LSTM: int = 100
_MIN_OBS_TREE: int = 30
_MIN_OBS_BASELINE: int = 1

# Burn-out phase: retrain on last N days for refinement.
_BURNOUT_WINDOW: int = 126  # ~6 months of trading days
_EARLY_STOP_PATIENCE: int = 3  # stop if no improvement for N iterations

# LSTM defaults.
_LSTM_HIDDEN: int = 32
_LSTM_LAYERS: int = 1
_LSTM_LOOKBACK: int = 21
_LSTM_EPOCHS: int = 50
_LSTM_LR: float = 0.001

# VAR max lag selection cap.
_VAR_MAX_LAG: int = 10


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Error metrics for a single model on a single variable."""

    model_name: str = ""
    variable: str = ""
    mae: float = float("nan")
    rmse: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    fitted: bool = False
    error: str | None = None


@dataclass
class ForecastResult:
    """Container for all forecasting outputs."""

    # Per-variable, per-horizon point forecasts.
    # {variable: {horizon_label: value}}
    forecasts: dict[str, dict[str, float]] = field(default_factory=dict)

    # Model failure flags.
    model_failed_kalman: bool = False
    model_failed_garch: bool = False
    model_failed_var: bool = False
    model_failed_lstm: bool = False
    model_failed_tree: bool = False
    # Baseline never fails.

    # Error messages.
    kalman_error: str | None = None
    garch_error: str | None = None
    var_error: str | None = None
    lstm_error: str | None = None
    tree_error: str | None = None

    # Per-model, per-variable metrics.
    metrics: list[ModelMetrics] = field(default_factory=list)

    # Which model was used for each variable.
    # {variable: model_name}
    model_used: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: error metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Return (MAE, RMSE) for non-NaN aligned pairs."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan"), float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return mae, rmse


def _split_train_test(
    series: np.ndarray,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1-D series into train/test (no shuffle -- temporal)."""
    n = len(series)
    split = max(1, int(n * (1 - test_frac)))
    return series[:split], series[split:]


# ---------------------------------------------------------------------------
# Tier variable lookup
# ---------------------------------------------------------------------------


def _load_tier_variables() -> dict[str, list[str]]:
    """Load tier -> variable list from survival_hierarchy config."""
    try:
        cfg = load_config("survival_hierarchy")
    except FileNotFoundError:
        logger.warning("survival_hierarchy config not found")
        return {}
    tiers = cfg.get("tiers", {})
    result: dict[str, list[str]] = {}
    for tier_key, tier_data in tiers.items():
        result[tier_key] = tier_data.get("variables", [])
    return result


def _get_tier_for_variable(
    variable: str,
    tier_map: dict[str, list[str]],
) -> str:
    """Return the tier key a variable belongs to, or 'unknown'."""
    for tier_key, vars_list in tier_map.items():
        if variable in vars_list:
            return tier_key
    return "unknown"


# ===========================================================================
# Model implementations
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Kalman filter (local-level state-space model)
# ---------------------------------------------------------------------------


def fit_kalman(
    series: np.ndarray,
    n_forecast: int = 1,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a local-level Kalman filter and produce forecasts.

    Parameters
    ----------
    series:
        1-D array of observed values (may contain NaN -- the Kalman
        filter handles missing observations natively).
    n_forecast:
        Number of steps ahead to forecast.

    Returns
    -------
    (forecasts, metrics)
        ``forecasts`` is an array of length ``n_forecast``, or None on
        failure.
    """
    metrics = ModelMetrics(model_name="kalman")

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- skipping Kalman filter"
        logger.warning(metrics.error)
        return None, metrics

    clean = series[~np.isnan(series)]
    if len(clean) < _MIN_OBS_KALMAN:
        metrics.error = (
            f"Insufficient observations for Kalman ({len(clean)} < {_MIN_OBS_KALMAN})"
        )
        logger.warning(metrics.error)
        return None, metrics

    try:
        train, test = _split_train_test(clean)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                train,
                level="local level",
            )
            result = model.fit(disp=False, maxiter=200)

        # In-sample predictions for validation.
        if len(test) > 0:
            forecast_obj = result.get_forecast(steps=len(test))
            preds = forecast_obj.predicted_mean
            mae, rmse = _compute_metrics(test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit for final forecast.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = UnobservedComponents(clean, level="local level")
            full_result = full_model.fit(disp=False, maxiter=200)

        forecast_obj = full_result.get_forecast(steps=n_forecast)
        forecasts = forecast_obj.predicted_mean

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info(
            "Kalman fit: %d train, %d test, MAE=%.6f, RMSE=%.6f",
            len(train), len(test), mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Kalman fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 2. GARCH (volatility forecasting)
# ---------------------------------------------------------------------------


def fit_garch(
    returns: np.ndarray,
    n_forecast: int = 1,
    p: int = 1,
    q: int = 1,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a GARCH(p,q) model for conditional volatility forecasting.

    Parameters
    ----------
    returns:
        1-D array of daily returns (not prices).
    n_forecast:
        Number of steps ahead.
    p, q:
        GARCH order parameters.

    Returns
    -------
    (volatility_forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="garch")

    try:
        from arch import arch_model  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "arch library not installed -- skipping GARCH"
        logger.warning(metrics.error)
        return None, metrics

    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS_GARCH:
        metrics.error = (
            f"Insufficient observations for GARCH ({len(clean)} < {_MIN_OBS_GARCH})"
        )
        logger.warning(metrics.error)
        return None, metrics

    try:
        # Scale returns to percentage for numerical stability.
        scaled = clean * 100.0
        train, test = _split_train_test(scaled)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                train,
                vol="Garch",
                p=p,
                q=q,
                mean="Constant",
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)

        # Validation forecasts.
        if len(test) > 0:
            forecast_obj = result.forecast(horizon=len(test))
            # Variance forecast -> std dev.
            var_forecast = forecast_obj.variance.iloc[-1].values
            vol_pred = np.sqrt(var_forecast) / 100.0  # back to decimal
            vol_actual = np.abs(test) / 100.0
            mae, rmse = _compute_metrics(
                vol_actual[: len(vol_pred)],
                vol_pred[: len(vol_actual)],
            )
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = arch_model(
                scaled,
                vol="Garch",
                p=p,
                q=q,
                mean="Constant",
                rescale=False,
            )
            full_result = full_model.fit(disp="off", show_warning=False)

        forecast_obj = full_result.forecast(horizon=n_forecast)
        var_fcast = forecast_obj.variance.iloc[-1].values[:n_forecast]
        vol_forecasts = np.sqrt(var_fcast) / 100.0  # decimal volatility

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info(
            "GARCH(%d,%d) fit: %d train, %d test, MAE=%.6f, RMSE=%.6f",
            p, q, len(train), len(test), mae, rmse,
        )
        return vol_forecasts, metrics

    except Exception as exc:
        metrics.error = f"GARCH fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 3. VAR (multivariate) with AR(1) fallback
# ---------------------------------------------------------------------------


def fit_var(
    data: pd.DataFrame,
    target_col: str,
    n_forecast: int = 1,
    max_lag: int = _VAR_MAX_LAG,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a VAR model and forecast the target variable.

    Falls back to univariate AR(1) if VAR fails (e.g. singular matrix,
    too few variables).

    Parameters
    ----------
    data:
        DataFrame with multiple numeric columns (including ``target_col``).
    target_col:
        The variable to extract forecasts for.
    n_forecast:
        Steps ahead.
    max_lag:
        Maximum lag order for AIC selection.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="var")

    try:
        from statsmodels.tsa.api import VAR as VARModel  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- skipping VAR"
        logger.warning(metrics.error)
        return None, metrics

    clean = data.dropna()
    if len(clean) < _MIN_OBS_VAR:
        metrics.error = (
            f"Insufficient observations for VAR ({len(clean)} < {_MIN_OBS_VAR})"
        )
        logger.warning(metrics.error)
        # Try AR(1) fallback.
        return _fit_ar1_fallback(clean, target_col, n_forecast, metrics)

    if target_col not in clean.columns:
        metrics.error = f"Target column '{target_col}' not in data"
        logger.warning(metrics.error)
        return None, metrics

    try:
        train_n = max(1, int(len(clean) * 0.85))
        train = clean.iloc[:train_n]
        test = clean.iloc[train_n:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VARModel(train)
            # Select optimal lag via AIC, capped at max_lag.
            lag_order = min(max_lag, len(train) // 3)
            result = model.fit(maxlags=max(1, lag_order), ic="aic")

        selected_lag = result.k_ar

        # Validation.
        if len(test) > 0:
            forecast_arr = result.forecast(
                train.values[-selected_lag:], steps=len(test)
            )
            col_idx = list(clean.columns).index(target_col)
            preds = forecast_arr[:, col_idx]
            actual = test[target_col].values
            mae, rmse = _compute_metrics(actual, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit for final forecast.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = VARModel(clean)
            full_result = full_model.fit(maxlags=max(1, lag_order), ic="aic")

        full_lag = full_result.k_ar
        forecast_arr = full_result.forecast(
            clean.values[-full_lag:], steps=n_forecast
        )
        col_idx = list(clean.columns).index(target_col)
        forecasts = forecast_arr[:, col_idx]

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True
        metrics.model_name = f"var(lag={full_lag})"

        logger.info(
            "VAR fit: lag=%d, %d vars, %d train, MAE=%.6f, RMSE=%.6f",
            full_lag, len(clean.columns), len(train), mae, rmse,
        )
        return forecasts, metrics

    except Exception as exc:
        msg = f"VAR fitting failed: {exc}"
        logger.warning(msg + " -- falling back to AR(1)")
        metrics.error = msg
        return _fit_ar1_fallback(clean, target_col, n_forecast, metrics)


def _fit_ar1_fallback(
    data: pd.DataFrame,
    target_col: str,
    n_forecast: int,
    parent_metrics: ModelMetrics,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Univariate AR(1) fallback for VAR."""
    metrics = ModelMetrics(model_name="ar1")

    if target_col not in data.columns:
        metrics.error = f"Target '{target_col}' not in data for AR(1)"
        return None, metrics

    try:
        from statsmodels.tsa.ar_model import AutoReg  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- AR(1) fallback unavailable"
        return None, metrics

    series = data[target_col].dropna().values
    if len(series) < 10:
        metrics.error = f"Insufficient data for AR(1) ({len(series)} < 10)"
        return None, metrics

    try:
        train, test = _split_train_test(series)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoReg(train, lags=1)
            result = model.fit()

        if len(test) > 0:
            preds = result.predict(start=len(train), end=len(train) + len(test) - 1)
            mae, rmse = _compute_metrics(test, preds.values)
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = AutoReg(series, lags=1)
            full_result = full_model.fit()

        preds_final = full_result.predict(
            start=len(series),
            end=len(series) + n_forecast - 1,
        )

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info("AR(1) fallback fit: MAE=%.6f, RMSE=%.6f", mae, rmse)
        return preds_final.values, metrics

    except Exception as exc:
        metrics.error = f"AR(1) fallback failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 4. LSTM with tree/linear fallback
# ---------------------------------------------------------------------------


def fit_lstm(
    series: np.ndarray,
    n_forecast: int = 1,
    lookback: int = _LSTM_LOOKBACK,
    hidden_size: int = _LSTM_HIDDEN,
    num_layers: int = _LSTM_LAYERS,
    epochs: int = _LSTM_EPOCHS,
    lr: float = _LSTM_LR,
    random_state: int = 42,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit an LSTM for nonlinear pattern forecasting.

    Falls back to GradientBoosting or LinearRegression if PyTorch
    is unavailable.

    Parameters
    ----------
    series:
        1-D array of observed values.
    n_forecast:
        Steps ahead.
    lookback:
        Number of past observations used as input features.
    hidden_size:
        LSTM hidden dimension.
    num_layers:
        Number of stacked LSTM layers.
    epochs:
        Training epochs.
    lr:
        Learning rate.
    random_state:
        Random seed.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="lstm")

    try:
        import torch  # type: ignore[import-untyped]
        import torch.nn as nn  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "PyTorch not installed -- falling back to tree/linear"
        logger.warning(metrics.error)
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)

    clean = series[~np.isnan(series)]
    if len(clean) < _MIN_OBS_LSTM:
        metrics.error = (
            f"Insufficient observations for LSTM ({len(clean)} < {_MIN_OBS_LSTM})"
        )
        logger.warning(metrics.error + " -- falling back to tree/linear")
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)

    try:
        torch.manual_seed(random_state)

        # Normalise.
        mean_val = float(np.mean(clean))
        std_val = float(np.std(clean))
        if std_val < 1e-12:
            std_val = 1.0
        normed = (clean - mean_val) / std_val

        # Create sequences.
        X_list, y_list = [], []
        for i in range(lookback, len(normed)):
            X_list.append(normed[i - lookback: i])
            y_list.append(normed[i])

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        # Train/test split.
        split = max(1, int(len(X_arr) * 0.85))
        X_train, X_test = X_arr[:split], X_arr[split:]
        y_train, y_test = y_arr[:split], y_arr[split:]

        X_train_t = torch.from_numpy(X_train).unsqueeze(-1)
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test).unsqueeze(-1)

        # Simple LSTM model.
        class _SimpleLSTM(nn.Module):
            def __init__(self, inp: int, hid: int, layers: int):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True)
                self.fc = nn.Linear(hid, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = _SimpleLSTM(1, hidden_size, num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training with early stop.
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - 1e-6:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= _EARLY_STOP_PATIENCE * 3:
                    break

        # Validation.
        model.eval()
        with torch.no_grad():
            if len(X_test) > 0:
                preds_test = model(X_test_t).numpy()
                preds_test_denorm = preds_test * std_val + mean_val
                y_test_denorm = y_test * std_val + mean_val
                mae, rmse = _compute_metrics(y_test_denorm, preds_test_denorm)
            else:
                mae, rmse = float("nan"), float("nan")

        # Multi-step forecast via autoregressive roll-forward.
        last_seq = torch.from_numpy(
            normed[-lookback:].astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1)

        forecasts_list = []
        current_seq = last_seq.clone()
        for _ in range(n_forecast):
            with torch.no_grad():
                next_val = model(current_seq).item()
            forecasts_list.append(next_val * std_val + mean_val)
            # Roll the window.
            new_entry = torch.tensor([[[next_val]]], dtype=torch.float32)
            current_seq = torch.cat([current_seq[:, 1:, :], new_entry], dim=1)

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "LSTM fit: lookback=%d, %d epochs, MAE=%.6f, RMSE=%.6f",
            lookback, epochs, mae, rmse,
        )
        return np.array(forecasts_list), metrics

    except Exception as exc:
        metrics.error = f"LSTM fitting failed: {exc}"
        logger.warning(metrics.error + " -- falling back to tree/linear")
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)


def _fit_linear_fallback(
    series: np.ndarray,
    n_forecast: int,
    lookback: int,
    random_state: int,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """GradientBoosting or LinearRegression fallback for LSTM."""
    metrics = ModelMetrics(model_name="gradient_boosting")

    clean = series[~np.isnan(series)]
    if len(clean) < lookback + 5:
        metrics.error = f"Insufficient data for linear fallback ({len(clean)})"
        return None, metrics

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
        model_cls = GradientBoostingRegressor
        model_kwargs: dict[str, Any] = {
            "n_estimators": 50,
            "max_depth": 3,
            "random_state": random_state,
        }
    except ImportError:
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
            model_cls = LinearRegression  # type: ignore[assignment]
            model_kwargs = {}
            metrics.model_name = "linear_regression"
        except ImportError:
            metrics.error = "sklearn not installed -- tree/linear fallback unavailable"
            return None, metrics

    try:
        # Build lagged features.
        X_list, y_list = [], []
        for i in range(lookback, len(clean)):
            X_list.append(clean[i - lookback: i])
            y_list.append(clean[i])

        X = np.array(X_list)
        y = np.array(y_list)

        split = max(1, int(len(X) * 0.85))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        if len(X_test) > 0:
            preds = model.predict(X_test)
            mae, rmse = _compute_metrics(y_test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Refit on all data.
        model.fit(X, y)

        # Multi-step autoregressive.
        current = clean[-lookback:].copy()
        forecasts = []
        for _ in range(n_forecast):
            next_val = float(model.predict(current.reshape(1, -1))[0])
            forecasts.append(next_val)
            current = np.append(current[1:], next_val)

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "%s fallback fit: MAE=%.6f, RMSE=%.6f",
            metrics.model_name, mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Tree/linear fallback failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 5. RF / GBM / XGB (tree ensembles for tabular features)
# ---------------------------------------------------------------------------


def fit_tree_ensemble(
    features: pd.DataFrame,
    target_col: str,
    n_forecast: int = 1,
    random_state: int = 42,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a tree ensemble (XGBoost > GBM > RF) for tabular forecasting.

    Parameters
    ----------
    features:
        DataFrame with feature columns and ``target_col``.
    target_col:
        Column to forecast.
    n_forecast:
        Steps ahead (autoregressive roll-forward).
    random_state:
        Random seed.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="xgboost")

    # Try XGBoost first, then sklearn GBM, then RF.
    model_obj = _try_load_tree_model(random_state, metrics)
    if model_obj is None:
        return None, metrics

    clean = features.dropna()
    if len(clean) < _MIN_OBS_TREE:
        metrics.error = (
            f"Insufficient observations for tree ensemble "
            f"({len(clean)} < {_MIN_OBS_TREE})"
        )
        logger.warning(metrics.error)
        return None, metrics

    if target_col not in clean.columns:
        metrics.error = f"Target column '{target_col}' not in features"
        logger.warning(metrics.error)
        return None, metrics

    try:
        feature_cols = [c for c in clean.columns if c != target_col]
        if not feature_cols:
            metrics.error = "No feature columns available for tree ensemble"
            return None, metrics

        X = clean[feature_cols].values
        y = clean[target_col].values

        split = max(1, int(len(X) * 0.85))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_obj.fit(X_train, y_train)

        if len(X_test) > 0:
            preds = model_obj.predict(X_test)
            mae, rmse = _compute_metrics(y_test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Refit on all data.
        model_obj.fit(X, y)

        # For multi-step: use last row's features as starting point.
        last_features = X[-1:].copy()
        forecasts = []
        for _ in range(n_forecast):
            next_val = float(model_obj.predict(last_features)[0])
            forecasts.append(next_val)
            # Shift features (simple carry-forward for tabular).
            # In production, features would be updated properly.

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "%s fit: %d features, %d train, MAE=%.6f, RMSE=%.6f",
            metrics.model_name, len(feature_cols), len(X_train), mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Tree ensemble fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


def _try_load_tree_model(
    random_state: int,
    metrics: ModelMetrics,
) -> Any | None:
    """Try loading XGBoost > GBM > RF, return first available."""
    try:
        from xgboost import XGBRegressor  # type: ignore[import-untyped]
        metrics.model_name = "xgboost"
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
        metrics.model_name = "gradient_boosting"
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        metrics.model_name = "random_forest"
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    except ImportError:
        pass

    metrics.error = "No tree ensemble library available (xgboost/sklearn)"
    logger.warning(metrics.error)
    return None


# ---------------------------------------------------------------------------
# 6. Baseline (always succeeds)
# ---------------------------------------------------------------------------


def fit_baseline(
    series: np.ndarray,
    n_forecast: int = 1,
    method: str = "ema",
    ema_span: int = 21,
) -> tuple[np.ndarray, ModelMetrics]:
    """Baseline forecaster: last-value or exponential moving average.

    This model **always** succeeds.  It is the final fallback.

    Parameters
    ----------
    series:
        1-D array of observed values.
    n_forecast:
        Steps ahead.
    method:
        ``"last"`` for last-value carry-forward, ``"ema"`` for
        exponential moving average.
    ema_span:
        EMA span in periods.

    Returns
    -------
    (forecasts, metrics)
        ``forecasts`` is always a valid array.
    """
    metrics = ModelMetrics(model_name=f"baseline_{method}", fitted=True)

    clean = series[~np.isnan(series)]

    if len(clean) == 0:
        # Absolute fallback: return zeros.
        metrics.model_name = "baseline_zero"
        return np.zeros(n_forecast), metrics

    if method == "ema" and len(clean) >= 3:
        # Compute EMA.
        alpha = 2.0 / (ema_span + 1)
        ema = clean[0]
        for val in clean[1:]:
            ema = alpha * val + (1 - alpha) * ema
        forecast_val = float(ema)
    else:
        forecast_val = float(clean[-1])

    forecasts = np.full(n_forecast, forecast_val)

    # Simple validation: use last 15% as test.
    if len(clean) > 10:
        split = max(1, int(len(clean) * 0.85))
        test = clean[split:]
        preds = np.full(len(test), float(clean[split - 1]))  # last-value
        mae, rmse = _compute_metrics(test, preds)
        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = split
        metrics.n_test = len(test)

    logger.info(
        "Baseline (%s) forecast: %.6f (n_forecast=%d)",
        method, forecast_val, n_forecast,
    )
    return forecasts, metrics


# ===========================================================================
# Burn-out phase
# ===========================================================================


def _burnout_refit(
    series: np.ndarray,
    fit_fn: Any,
    n_forecast: int,
    window: int = _BURNOUT_WINDOW,
    patience: int = _EARLY_STOP_PATIENCE,
    **kwargs: Any,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Intensive retraining on the most recent window.

    Repeatedly shrinks the window and refits.  Stops early if the
    validation error does not improve for ``patience`` iterations.

    Parameters
    ----------
    series:
        Full 1-D series.
    fit_fn:
        One of the fit_* functions above.
    n_forecast:
        Steps to forecast.
    window:
        Maximum recent-data window.
    patience:
        Early-stop patience.
    **kwargs:
        Extra args forwarded to ``fit_fn``.

    Returns
    -------
    Best (forecasts, metrics) from the burnout phase, or (None, metrics)
    if no improvement was found.
    """
    clean = series[~np.isnan(series)]
    if len(clean) < 30:
        return None, ModelMetrics(
            model_name="burnout",
            error="Insufficient data for burnout refit",
        )

    best_rmse = float("inf")
    best_result: tuple[np.ndarray | None, ModelMetrics] = (
        None,
        ModelMetrics(model_name="burnout"),
    )
    no_improve = 0

    for frac in [1.0, 0.75, 0.5]:
        win = max(30, int(min(window, len(clean)) * frac))
        subset = clean[-win:]
        forecasts, met = fit_fn(subset, n_forecast=n_forecast, **kwargs)
        if forecasts is not None and not np.isnan(met.rmse) and met.rmse < best_rmse:
            best_rmse = met.rmse
            best_result = (forecasts, met)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_result


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def run_forecasting(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    random_state: int = 42,
    enable_burnout: bool = True,
) -> tuple[pd.DataFrame, ForecastResult]:
    """Run the full forecasting pipeline on the daily cache.

    Applies each model in the fallback chain for each variable.
    Adds forecast columns to the cache and returns the result
    container with metrics.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex) with feature columns.
    variables:
        List of variable names to forecast.  If ``None``, all tier
        variables from the survival hierarchy are used.
    random_state:
        Random seed for reproducible models.
    enable_burnout:
        If True, run burnout refinement phase on best model.

    Returns
    -------
    (cache, result)
        The (possibly augmented) cache and the ``ForecastResult``.
    """
    logger.info("Starting forecasting pipeline...")

    result = ForecastResult()
    tier_map = _load_tier_variables()

    if variables is None:
        variables = []
        for tier_vars in tier_map.values():
            variables.extend(tier_vars)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_vars: list[str] = []
        for v in variables:
            if v not in seen:
                seen.add(v)
                unique_vars.append(v)
        variables = unique_vars

    # Filter to columns actually present in cache.
    available_vars = [v for v in variables if v in cache.columns]
    missing_vars = set(variables) - set(available_vars)
    if missing_vars:
        logger.info(
            "Forecasting: %d variables not in cache, skipping: %s",
            len(missing_vars),
            sorted(missing_vars)[:5],
        )

    # Also get returns/volatility for specialised models.
    has_returns = "return_1d" in cache.columns
    has_volatility = "volatility_21d" in cache.columns

    # ------------------------------------------------------------------
    # GARCH on volatility (special case)
    # ------------------------------------------------------------------
    if has_returns:
        returns = cache["return_1d"].values
        max_horizon = max(HORIZONS.values())
        garch_fcast, garch_met = fit_garch(
            returns, n_forecast=max_horizon,
        )
        garch_met.variable = "volatility_21d"
        result.metrics.append(garch_met)
        if garch_fcast is None:
            result.model_failed_garch = True
            result.garch_error = garch_met.error
        else:
            result.forecasts["volatility_garch"] = {
                label: float(garch_fcast[min(h - 1, len(garch_fcast) - 1)])
                for label, h in HORIZONS.items()
            }
            result.model_used["volatility_garch"] = "garch"

    # ------------------------------------------------------------------
    # Per-variable forecasting
    # ------------------------------------------------------------------
    kalman_attempted = False
    var_attempted = False
    lstm_attempted = False
    tree_attempted = False

    for var_name in available_vars:
        series = cache[var_name].values
        tier = _get_tier_for_variable(var_name, tier_map)
        max_horizon = max(HORIZONS.values())
        best_forecast: np.ndarray | None = None
        best_model_name = ""
        best_metrics: ModelMetrics | None = None

        # --- Kalman (preferred for tier1/tier2) ---
        if tier in ("tier1", "tier2"):
            fcast, met = fit_kalman(series, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = "kalman"
                best_metrics = met
            else:
                if not kalman_attempted:
                    result.model_failed_kalman = True
                    result.kalman_error = met.error
            kalman_attempted = True

        # --- VAR (if multiple variables available) ---
        if best_forecast is None and len(available_vars) >= 2:
            # Build a small multivariate frame from available vars.
            var_subset_cols = [
                c for c in available_vars
                if c in cache.columns
            ][:10]  # Cap at 10 for stability.
            if var_name not in var_subset_cols:
                var_subset_cols = [var_name] + var_subset_cols[:9]
            var_df = cache[var_subset_cols].copy()

            fcast, met = fit_var(var_df, var_name, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = met.model_name
                best_metrics = met
            else:
                if not var_attempted:
                    result.model_failed_var = True
                    result.var_error = met.error
            var_attempted = True

        # --- LSTM / tree-linear fallback ---
        if best_forecast is None:
            fcast, met = fit_lstm(
                series,
                n_forecast=max_horizon,
                random_state=random_state,
            )
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = met.model_name
                best_metrics = met
            else:
                if not lstm_attempted:
                    result.model_failed_lstm = True
                    result.lstm_error = met.error
            lstm_attempted = True

        # --- Tree ensemble on tabular features ---
        if best_forecast is None:
            feature_cols = [
                c for c in cache.columns
                if c != var_name
                and cache[c].dtype in (np.float64, np.float32, np.int64)
            ][:15]
            if feature_cols:
                feat_df = cache[feature_cols + [var_name]].copy()
                fcast, met = fit_tree_ensemble(
                    feat_df,
                    var_name,
                    n_forecast=max_horizon,
                    random_state=random_state,
                )
                met.variable = var_name
                result.metrics.append(met)
                if fcast is not None:
                    best_forecast = fcast
                    best_model_name = met.model_name
                    best_metrics = met
                else:
                    if not tree_attempted:
                        result.model_failed_tree = True
                        result.tree_error = met.error
                tree_attempted = True

        # --- Baseline (always succeeds) ---
        if best_forecast is None:
            fcast, met = fit_baseline(series, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met

        # --- Burnout refinement ---
        if enable_burnout and best_forecast is not None and best_model_name == "kalman":
            burnout_fcast, burnout_met = _burnout_refit(
                series, fit_kalman, n_forecast=max_horizon
            )
            if (
                burnout_fcast is not None
                and not np.isnan(burnout_met.rmse)
                and (
                    best_metrics is None
                    or np.isnan(best_metrics.rmse)
                    or burnout_met.rmse < best_metrics.rmse
                )
            ):
                best_forecast = burnout_fcast
                best_model_name = f"{best_model_name}_burnout"
                best_metrics = burnout_met

        # Store results.
        if best_forecast is not None:
            result.forecasts[var_name] = {
                label: float(best_forecast[min(h - 1, len(best_forecast) - 1)])
                for label, h in HORIZONS.items()
            }
            result.model_used[var_name] = best_model_name

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_forecasted = len(result.forecasts)
    n_failed = sum([
        result.model_failed_kalman,
        result.model_failed_garch,
        result.model_failed_var,
        result.model_failed_lstm,
        result.model_failed_tree,
    ])
    models_used = {}
    for var, model in result.model_used.items():
        models_used[model] = models_used.get(model, 0) + 1

    logger.info(
        "Forecasting complete: %d variables forecasted, %d model types failed, "
        "model distribution: %s",
        n_forecasted,
        n_failed,
        models_used,
    )

    return cache, result
