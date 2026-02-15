"""Load API secrets from Kaggle or local environment.

On Kaggle, keys are stored via the Kaggle Secrets client.  For local
development, falls back to environment variables.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Required keys -- pipeline will halt if any are missing.
_REQUIRED_KEYS = ("EULERPOOL_API_KEY", "FMP_API_KEY", "GEMINI_API_KEY")

# Optional keys -- logged as warning if absent.
_OPTIONAL_KEYS = ("WORLD_BANK_API_KEY",)


def _load_from_kaggle() -> dict[str, str]:
    """Attempt to read secrets via Kaggle UserSecretsClient."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]
        client = UserSecretsClient()
        secrets: dict[str, str] = {}
        for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS):
            try:
                value = client.get_secret(key)
                if value:
                    secrets[key] = value
            except Exception:
                pass
        return secrets
    except ImportError:
        return {}


def _load_from_env() -> dict[str, str]:
    """Fallback: read from OS environment variables."""
    secrets: dict[str, str] = {}
    for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS):
        value = os.environ.get(key)
        if value:
            secrets[key] = value
    return secrets


def load_secrets() -> dict[str, str]:
    """Return a dict of API secrets.

    Tries Kaggle secrets first, then falls back to environment variables.
    Raises ``SystemExit`` with a clear message if any required key is missing.
    """
    secrets = _load_from_kaggle()
    if not secrets:
        logger.info("Kaggle secrets client unavailable; falling back to env vars")
        secrets = _load_from_env()

    # Validate required keys
    missing = [k for k in _REQUIRED_KEYS if k not in secrets]
    if missing:
        msg = (
            "Missing required API keys: %s. "
            "Store them in Kaggle Secrets or set as environment variables."
        )
        logger.critical(msg, ", ".join(missing))
        raise SystemExit(msg % ", ".join(missing))

    # Warn about optional keys
    for key in _OPTIONAL_KEYS:
        if key not in secrets:
            logger.warning("Optional key %s not found -- some features may be limited", key)

    return secrets
