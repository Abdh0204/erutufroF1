"""Shared HTTP utilities with disk caching, retries, and request logging.

Every API client in the pipeline routes requests through this module to get
consistent retry logic, on-disk caching, and audit logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from operator1.config_loader import get_global_config
from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# Module-level request log (populated during pipeline run).
_request_log: list[dict[str, Any]] = []


class HTTPError(Exception):
    """Raised when an HTTP request fails after all retries."""

    def __init__(self, url: str, status_code: int, detail: str = "") -> None:
        self.url = url
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code} for {url}: {detail}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    """Deterministic hash for a request (URL + sorted params)."""
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def _read_cache(cache_dir: str, key: str, ttl_hours: float) -> dict | list | None:
    """Return cached response if it exists and is fresh, else None."""
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > ttl_hours:
        logger.debug("Cache expired (%.1fh > %.1fh): %s", age_hours, ttl_hours, path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt cache file, ignoring: %s", path)
        return None


def _write_cache(cache_dir: str, key: str, data: Any) -> None:
    """Persist JSON-serialisable response to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, key)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except (TypeError, OSError) as exc:
        logger.warning("Failed to write cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# API key injection
# ---------------------------------------------------------------------------

def inject_api_key(url: str, api_key: str) -> str:
    """Append ``apikey=<key>`` to *url* using the correct separator.

    If the URL already contains a ``?``, append with ``&``; otherwise ``?``.
    """
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}apikey={api_key}"


# ---------------------------------------------------------------------------
# Core GET with retries + caching
# ---------------------------------------------------------------------------

def cached_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    cache_dir: str | None = None,
    ttl_hours: float | None = None,
) -> Any:
    """HTTP GET with disk caching and exponential-backoff retries.

    Parameters
    ----------
    url:
        Full request URL (API key should already be injected if needed).
    params:
        Optional query parameters.
    headers:
        Optional request headers.
    cache_dir:
        Directory for disk cache.  Defaults to ``cache/http``.
    ttl_hours:
        Cache freshness threshold.  Defaults to config value.

    Returns
    -------
    Parsed JSON response (dict or list).

    Raises
    ------
    HTTPError
        After all retries are exhausted.
    """
    cfg = get_global_config()
    max_retries: int = cfg.get("max_retries", 3)
    backoff: float = cfg.get("backoff_factor", 2.0)
    timeout: int = cfg.get("timeout_s", 30)

    if cache_dir is None:
        cache_dir = os.path.join(CACHE_DIR, "http")
    if ttl_hours is None:
        ttl_hours = cfg.get("http_cache_ttl_hours", 24)

    key = _cache_key(url, params)
    cached = _read_cache(cache_dir, key, ttl_hours)
    if cached is not None:
        logger.debug("Cache HIT for %s", url)
        return cached

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=timeout,
            )
            elapsed = time.time() - t0

            # Log the request
            _request_log.append({
                "url": _sanitise_url(url),
                "status": resp.status_code,
                "elapsed_s": round(elapsed, 3),
                "attempt": attempt,
                "cached": False,
            })

            if resp.status_code == 200:
                data = resp.json()
                _write_cache(cache_dir, key, data)
                return data

            # Respect Retry-After header
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait = float(retry_after)
            else:
                wait = backoff ** attempt

            logger.warning(
                "HTTP %d on attempt %d/%d for %s -- retrying in %.1fs",
                resp.status_code, attempt, max_retries, _sanitise_url(url), wait,
            )
            last_exc = HTTPError(url, resp.status_code, resp.text[:200])
            time.sleep(wait)

        except requests.RequestException as exc:
            elapsed = time.time() - t0
            _request_log.append({
                "url": _sanitise_url(url),
                "status": None,
                "elapsed_s": round(elapsed, 3),
                "attempt": attempt,
                "cached": False,
                "error": str(exc),
            })
            last_exc = exc
            wait = backoff ** attempt
            logger.warning(
                "Request error on attempt %d/%d: %s -- retrying in %.1fs",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    raise HTTPError(
        url, getattr(last_exc, "status_code", 0),
        f"All {max_retries} retries exhausted: {last_exc}",
    )


# ---------------------------------------------------------------------------
# Request log accessors
# ---------------------------------------------------------------------------

def get_request_log() -> list[dict[str, Any]]:
    """Return the accumulated request log for metadata.json."""
    return list(_request_log)


def clear_request_log() -> None:
    """Reset the request log (useful between test runs)."""
    _request_log.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_url(url: str) -> str:
    """Strip API keys from a URL before logging."""
    # Remove apikey=... parameter value
    import re
    return re.sub(r"(apikey=)[^&]+", r"\1***", url)
