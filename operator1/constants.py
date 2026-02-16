"""Global constants for Operator 1 pipeline."""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Date window -- 2-year lookback from today
# ---------------------------------------------------------------------------
DATE_END: date = date.today()
DATE_START: date = DATE_END - timedelta(days=730)

# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------
EPSILON: float = 1e-9

# ---------------------------------------------------------------------------
# Eulerpool base URL
# ---------------------------------------------------------------------------
EULERPOOL_BASE_URL: str = "https://api.eulerpool.com"

# ---------------------------------------------------------------------------
# FMP base URL
# ---------------------------------------------------------------------------
FMP_BASE_URL: str = "https://financialmodelingprep.com/api/v3"

# ---------------------------------------------------------------------------
# World Bank base URL
# ---------------------------------------------------------------------------
WORLD_BANK_BASE_URL: str = "https://api.worldbank.org/v2"

# World Bank Documents & Reports Search API (WDS)
WB_WDS_BASE_URL: str = "https://search.worldbank.org/api/v3/wds"

# ---------------------------------------------------------------------------
# Gemini base URL
# ---------------------------------------------------------------------------
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"

# ---------------------------------------------------------------------------
# Match scoring thresholds (entity discovery)
# ---------------------------------------------------------------------------
MATCH_SCORE_THRESHOLD: int = 70
SECTOR_PEER_FALLBACK_COUNT: int = 5

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR: str = "cache"
RAW_CACHE_DIR: str = "cache/raw"
WB_CACHE_DIR: str = "cache/world_bank_raw"
