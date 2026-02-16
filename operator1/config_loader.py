"""Load YAML configuration files from the config/ directory."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_cache: dict[str, dict[str, Any]] = {}


def load_config(name: str, *, reload: bool = False) -> dict[str, Any]:
    """Load a YAML config by name (without extension).

    Parameters
    ----------
    name:
        Config filename stem, e.g. ``"global_config"`` loads
        ``config/global_config.yml``.
    reload:
        If *True*, bypass the in-memory cache and re-read from disk.

    Returns
    -------
    dict
        Parsed YAML as a Python dict.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not reload and name in _cache:
        return _cache[name]

    path = _CONFIG_DIR / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.debug("Loaded config %s from %s", name, path)
    _cache[name] = data
    return data


def get_global_config() -> dict[str, Any]:
    """Convenience accessor for ``config/global_config.yml``."""
    return load_config("global_config")
