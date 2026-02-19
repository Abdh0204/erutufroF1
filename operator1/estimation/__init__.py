"""Post-cache estimation and imputation engine.

Supports two Pass 2 backends:
  - ``bayesian_ridge``: Per-variable BayesianRidge (default, linear)
  - ``vae``: Variational Autoencoder (nonlinear, requires torch)
"""
