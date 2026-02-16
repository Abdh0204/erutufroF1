"""Load API secrets from Kaggle, .env file, or environment variables.

On Kaggle, keys are stored via the Kaggle Secrets client.  For local
development, falls back to a ``.env`` file in the project root, then
to environment variables.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to load .env file for local development
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load variables from .env file if python-dotenv is available."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
    except ImportError:
        # Manual fallback: parse simple KEY=VALUE lines
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)
            logger.info("Loaded .env file manually (python-dotenv not installed)")
        except Exception as exc:
            logger.warning("Failed to parse .env file: %s", exc)

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

    Priority order:
    1. Kaggle Secrets client (if running on Kaggle)
    2. ``.env`` file in project root
    3. OS environment variables

    Raises ``SystemExit`` with a clear message if any required key is missing.
    """
    # Try .env file first (for local development)
    _load_dotenv()

    secrets = _load_from_kaggle()
    if not secrets:
        logger.info("Kaggle secrets client unavailable; falling back to env vars / .env")
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
