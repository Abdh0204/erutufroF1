"""Variational Autoencoder (VAE) imputer for the Sudoku inference step.

An optional replacement for the BayesianRidge rolling imputer in Pass 2
of the estimation engine.  The VAE captures nonlinear relationships
between financial variables, which can improve imputation quality when
the feature space has complex interactions (e.g. solvency ratios that
depend nonlinearly on multiple balance-sheet items).

Architecture
------------
A lightweight conditional VAE with:
  - Encoder: observed features -> latent mean + log-variance
  - Decoder: latent sample + observed features -> reconstructed targets
  - Loss: reconstruction (MSE) + KL divergence (standard VAE ELBO)

The model is trained **only on fully-observed rows** up to day ``t``
(no look-ahead), then used to impute missing values for rows where
some variables are absent.

Dependencies
------------
Requires ``torch >= 2.0`` (already in requirements.txt for LSTM
forecasting).  Falls back gracefully if torch is unavailable.

Usage
-----
This module is called from ``estimator.py`` when the imputer method
is set to ``"vae"`` (via ``global_config.yml`` or function argument).
The default remains ``"bayesian_ridge"`` for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum rows of fully-observed data needed to train the VAE
_MIN_TRAIN_ROWS = 30

# Training hyperparameters (conservative defaults for financial data)
_DEFAULT_LATENT_DIM = 8
_DEFAULT_HIDDEN_DIM = 32
_DEFAULT_EPOCHS = 80
_DEFAULT_LR = 1e-3
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_KL_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# VAE PyTorch module
# ---------------------------------------------------------------------------

def _check_torch_available() -> bool:
    """Return True if torch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _build_vae(
    input_dim: int,
    latent_dim: int = _DEFAULT_LATENT_DIM,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
):
    """Construct the VAE encoder and decoder as torch Modules.

    Returns ``(vae_model, torch_module)`` -- the model instance and the
    torch module reference for downstream use.
    """
    import torch
    import torch.nn as nn

    class VAEEncoder(nn.Module):
        """Maps input features to latent distribution parameters."""

        def __init__(self, in_dim: int, h_dim: int, z_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(in_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, h_dim)
            self.fc_mu = nn.Linear(h_dim, z_dim)
            self.fc_logvar = nn.Linear(h_dim, z_dim)
            self.relu = nn.ReLU()
            # LayerNorm instead of BatchNorm1d to avoid single-sample
            # batch crashes and to behave consistently during MC-Dropout
            # inference (BatchNorm switches stats mode with train/eval).
            self.ln1 = nn.LayerNorm(h_dim)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor):
            h = self.relu(self.ln1(self.fc1(x)))
            h = self.dropout(self.relu(self.fc2(h)))
            return self.fc_mu(h), self.fc_logvar(h)

    class VAEDecoder(nn.Module):
        """Maps latent sample back to reconstructed features."""

        def __init__(self, z_dim: int, h_dim: int, out_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(z_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, h_dim)
            self.fc_out = nn.Linear(h_dim, out_dim)
            self.relu = nn.ReLU()
            self.ln1 = nn.LayerNorm(h_dim)
            self.dropout = nn.Dropout(0.1)

        def forward(self, z: torch.Tensor):
            h = self.relu(self.ln1(self.fc1(z)))
            h = self.dropout(self.relu(self.fc2(h)))
            return self.fc_out(h)

    class VAE(nn.Module):
        """Variational Autoencoder for multivariate imputation."""

        def __init__(self, in_dim: int, h_dim: int, z_dim: int) -> None:
            super().__init__()
            self.encoder = VAEEncoder(in_dim, h_dim, z_dim)
            self.decoder = VAEDecoder(z_dim, h_dim, in_dim)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
            """Sample z = mu + eps * std using the reparameterization trick."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x: torch.Tensor):
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar

    model = VAE(input_dim, hidden_dim, latent_dim)
    return model


def _vae_loss(recon_x, x, mu, logvar, kl_weight: float = _DEFAULT_KL_WEIGHT):
    """Compute the VAE ELBO loss: reconstruction + KL divergence.

    Parameters
    ----------
    recon_x : torch.Tensor
        Reconstructed output from the decoder.
    x : torch.Tensor
        Original input.
    mu : torch.Tensor
        Latent mean from encoder.
    logvar : torch.Tensor
        Latent log-variance from encoder.
    kl_weight : float
        Weighting factor for KL term (beta-VAE style).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    import torch
    import torch.nn.functional as F

    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


# ---------------------------------------------------------------------------
# Training and imputation
# ---------------------------------------------------------------------------

@dataclass
class VAEImputerResult:
    """Result container for the VAE imputation step."""

    estimated_values: dict[str, pd.Series]   # var -> estimated series
    confidence_scores: dict[str, pd.Series]  # var -> confidence series
    train_loss_final: float = 0.0
    n_train_rows: int = 0
    n_features_used: int = 0
    fallback_used: bool = False


def _normalize_data(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns to zero mean and unit variance.

    Returns (X_normalized, means, stds).
    """
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds < 1e-10] = 1.0  # avoid division by zero
    X_norm = (X - means) / stds
    return X_norm, means, stds


def _denormalize(
    X_norm: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Reverse standardization."""
    return X_norm * stds + means


def train_and_impute_vae(
    df: pd.DataFrame,
    target_vars: list[str],
    feature_cols: list[str],
    tier_weights: dict[str, pd.Series] | None = None,
    latent_dim: int = _DEFAULT_LATENT_DIM,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    epochs: int = _DEFAULT_EPOCHS,
    lr: float = _DEFAULT_LR,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    kl_weight: float = _DEFAULT_KL_WEIGHT,
) -> VAEImputerResult:
    """Train a VAE on observed data and impute missing values.

    The VAE is trained on rows where ALL ``target_vars`` are observed,
    using ``feature_cols`` + ``target_vars`` as the joint input space.
    After training, rows with missing target values are imputed by
    encoding the available features and decoding the reconstruction.

    Parameters
    ----------
    df : pd.DataFrame
        The full feature table.
    target_vars : list[str]
        Variables to impute (subset of columns in df).
    feature_cols : list[str]
        Predictor columns (numeric, non-flag).
    tier_weights : dict[str, pd.Series] | None
        Optional per-variable tier weight series for weighted training.
    latent_dim : int
        Dimension of the VAE latent space.
    hidden_dim : int
        Hidden layer width.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    kl_weight : float
        Beta weight for KL divergence term.

    Returns
    -------
    VAEImputerResult
        Contains estimated values and confidence scores per variable.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    result = VAEImputerResult(
        estimated_values={v: pd.Series(np.nan, index=df.index) for v in target_vars},
        confidence_scores={v: pd.Series(np.nan, index=df.index) for v in target_vars},
    )

    # Build the joint feature matrix: feature_cols + target_vars
    all_cols = feature_cols + [v for v in target_vars if v not in feature_cols]

    # Extract numeric data, forward-fill features
    X_raw = df[all_cols].copy()
    X_raw[feature_cols] = X_raw[feature_cols].ffill().bfill()

    # Find fully-observed rows (all target vars present) for training
    target_observed_mask = df[target_vars].notna().all(axis=1)
    # Also require features to be non-NaN after ffill
    feature_valid_mask = X_raw[feature_cols].notna().all(axis=1)
    train_mask = target_observed_mask & feature_valid_mask

    n_train = train_mask.sum()
    if n_train < _MIN_TRAIN_ROWS:
        logger.warning(
            "VAE imputer: only %d fully-observed rows (need %d) -- skipping",
            n_train, _MIN_TRAIN_ROWS,
        )
        result.fallback_used = True
        return result

    result.n_train_rows = int(n_train)
    result.n_features_used = len(all_cols)

    # Normalize
    X_train_raw = X_raw.loc[train_mask].values.astype(np.float64)
    X_norm, means, stds = _normalize_data(X_train_raw)

    # Convert to torch tensors
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(X_tensor)),
        shuffle=True,
        drop_last=False,
    )

    # Build and train model
    input_dim = len(all_cols)
    model = _build_vae(input_dim, latent_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    model.train()
    final_loss = float("inf")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon, mu, logvar = model(batch_x)
                loss = _vae_loss(recon, batch_x, mu, logvar, kl_weight)
                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)
            final_loss = avg_loss

    result.train_loss_final = final_loss
    logger.info(
        "VAE training complete: %d epochs, final loss %.6f, "
        "%d train rows, %d features",
        epochs, final_loss, n_train, input_dim,
    )

    # --- Save trained model to cache for reuse ---
    _save_vae_model(model, means, stds, all_cols, input_dim, latent_dim, hidden_dim)

    # --- Imputation pass ---
    # Find rows that need imputation: at least one target var missing
    any_missing_mask = df[target_vars].isna().any(axis=1) & feature_valid_mask

    if not any_missing_mask.any():
        logger.info("VAE imputer: no rows need imputation")
        return result

    # Prepare input for missing rows: use observed features, fill missing
    # targets with the column mean (less biased than zero after normalization)
    X_impute_raw = X_raw.loc[any_missing_mask].copy()
    for v in target_vars:
        col_mean = df[v].mean() if df[v].notna().any() else 0.0
        X_impute_raw[v] = X_impute_raw[v].fillna(col_mean)

    X_impute_vals = X_impute_raw.values.astype(np.float64)
    X_impute_norm = (X_impute_vals - means) / stds
    X_impute_tensor = torch.tensor(X_impute_norm, dtype=torch.float32)

    with torch.no_grad():
        # MC-Dropout: enable dropout layers only (not LayerNorm)
        # by manually setting each dropout submodule to train mode.
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        n_mc_samples = 20
        predictions = []
        for _ in range(n_mc_samples):
            recon, _, _ = model(X_impute_tensor)
            predictions.append(recon.numpy())

    predictions = np.stack(predictions, axis=0)  # (n_mc, n_rows, n_features)
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    # Denormalize predictions
    pred_mean_denorm = _denormalize(pred_mean, means, stds)
    pred_std_denorm = pred_std * stds  # std scales linearly

    # Extract target variable estimates
    impute_indices = df.index[any_missing_mask]

    for var in target_vars:
        if var not in all_cols:
            continue

        col_idx = all_cols.index(var)
        var_missing = df.loc[any_missing_mask, var].isna()

        if not var_missing.any():
            continue

        missing_in_impute = var_missing[var_missing].index

        for idx in missing_in_impute:
            row_pos = impute_indices.get_loc(idx)
            est_val = pred_mean_denorm[row_pos, col_idx]
            est_std = pred_std_denorm[row_pos, col_idx]

            result.estimated_values[var].loc[idx] = est_val

            # Confidence: inverse of relative uncertainty, clipped to [0.1, 0.9]
            # Lower std relative to the variable's overall std means higher confidence
            var_std = stds[col_idx]
            if var_std > 1e-10 and est_std > 0:
                rel_uncertainty = est_std / var_std
                conf = max(0.1, min(0.9, 1.0 - rel_uncertainty))
            else:
                conf = 0.5  # default when uncertainty can't be computed
            result.confidence_scores[var].loc[idx] = conf

    n_imputed = sum(
        result.estimated_values[v].notna().sum() for v in target_vars
    )
    logger.info(
        "VAE imputation complete: %d values imputed across %d variables",
        n_imputed, len(target_vars),
    )

    return result


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------

_VAE_CACHE_DIR = os.path.join("cache", "vae_model")


def _save_vae_model(
    model,
    means: np.ndarray,
    stds: np.ndarray,
    columns: list[str],
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
) -> None:
    """Persist the trained VAE and normalization params to disk.

    Saved to ``cache/vae_model/`` so that subsequent pipeline runs
    with ``FORCE_REBUILD=false`` can reload instead of retraining.
    """
    try:
        import torch

        os.makedirs(_VAE_CACHE_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(_VAE_CACHE_DIR, "vae_weights.pt"))
        np.savez(
            os.path.join(_VAE_CACHE_DIR, "vae_norm.npz"),
            means=means,
            stds=stds,
            input_dim=np.array([input_dim]),
            latent_dim=np.array([latent_dim]),
            hidden_dim=np.array([hidden_dim]),
        )
        # Save column order for validation on reload
        with open(os.path.join(_VAE_CACHE_DIR, "vae_columns.txt"), "w") as f:
            f.write("\n".join(columns))
        logger.info("VAE model cached to %s", _VAE_CACHE_DIR)
    except Exception as exc:
        logger.debug("Failed to cache VAE model: %s", exc)


def load_cached_vae_model(
    expected_columns: list[str] | None = None,
):
    """Load a previously cached VAE model if available.

    Parameters
    ----------
    expected_columns : list[str] | None
        If provided, validates that the cached model was trained on
        the same column set.  Returns None on mismatch.

    Returns
    -------
    tuple or None
        ``(model, means, stds, columns)`` if cache is valid, else None.
    """
    try:
        import torch
    except ImportError:
        return None

    weights_path = os.path.join(_VAE_CACHE_DIR, "vae_weights.pt")
    norm_path = os.path.join(_VAE_CACHE_DIR, "vae_norm.npz")
    cols_path = os.path.join(_VAE_CACHE_DIR, "vae_columns.txt")

    if not (os.path.exists(weights_path) and os.path.exists(norm_path)):
        return None

    try:
        norm_data = np.load(norm_path)
        means = norm_data["means"]
        stds = norm_data["stds"]
        input_dim = int(norm_data["input_dim"][0])
        latent_dim = int(norm_data["latent_dim"][0])
        hidden_dim = int(norm_data["hidden_dim"][0])

        # Validate columns if provided
        if os.path.exists(cols_path) and expected_columns is not None:
            with open(cols_path) as f:
                cached_cols = f.read().strip().split("\n")
            if cached_cols != expected_columns:
                logger.debug("VAE cache column mismatch -- retraining")
                return None

        model = _build_vae(input_dim, latent_dim, hidden_dim)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        cached_cols = []
        if os.path.exists(cols_path):
            with open(cols_path) as f:
                cached_cols = f.read().strip().split("\n")

        logger.info("Loaded cached VAE model from %s", _VAE_CACHE_DIR)
        return model, means, stds, cached_cols

    except Exception as exc:
        logger.debug("Failed to load cached VAE: %s", exc)
        return None
