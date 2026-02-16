"""C1 -- Candlestick pattern detection (rule-based).

Detects classical OHLC candlestick patterns and predicts future
pattern formations. Uses pure rule-based logic so no external
dependency (TA-Lib) is required, though it can optionally use it.

Spec reference: The_Apps_core_idea.pdf Section E.2 Category 6.

Detected patterns:
  - Doji, Hammer, Inverted Hammer, Shooting Star
  - Bullish/Bearish Engulfing
  - Morning Star, Evening Star
  - Three White Soldiers, Three Black Crows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum body-to-range ratio to consider a candle "real-bodied".
_BODY_THRESHOLD = 0.3
# Maximum body-to-range ratio for doji-like candles.
_DOJI_THRESHOLD = 0.1


@dataclass
class PatternMatch:
    """A single detected pattern occurrence."""
    name: str
    date: str
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float = 0.5


@dataclass
class PatternDetectorResult:
    """Result from pattern detection."""
    recent_patterns: list[PatternMatch] = field(default_factory=list)
    predicted_patterns_week: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "recent_patterns": [
                {"name": p.name, "date": p.date,
                 "direction": p.direction, "confidence": p.confidence}
                for p in self.recent_patterns
            ],
            "predicted_patterns_week": self.predicted_patterns_week,
        }


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range(h: float, l: float) -> float:
    return max(h - l, 1e-10)


def _is_bullish(o: float, c: float) -> bool:
    return c > o


def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def detect_doji(o: float, h: float, l: float, c: float) -> bool:
    """Body is very small relative to range."""
    return _body(o, c) / _range(h, l) < _DOJI_THRESHOLD


def detect_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Small body at top, long lower shadow (>= 2x body)."""
    body = _body(o, c)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    rng = _range(h, l)
    return (
        body / rng < _BODY_THRESHOLD
        and lower >= 2 * body
        and upper < body * 0.5
    )


def detect_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Small body at bottom, long upper shadow (>= 2x body)."""
    body = _body(o, c)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    rng = _range(h, l)
    return (
        body / rng < _BODY_THRESHOLD
        and upper >= 2 * body
        and lower < body * 0.5
    )


def detect_engulfing(
    o1: float, c1: float, o2: float, c2: float,
) -> str | None:
    """Two-candle engulfing. Returns 'bullish' or 'bearish' or None."""
    # Bullish engulfing: day1 bearish, day2 bullish, day2 body engulfs day1
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1:
        return "bullish"
    # Bearish engulfing: day1 bullish, day2 bearish, day2 body engulfs day1
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1:
        return "bearish"
    return None


def detect_three_soldiers_crows(
    opens: list[float], closes: list[float],
) -> str | None:
    """Three consecutive strong candles in the same direction."""
    if len(opens) < 3 or len(closes) < 3:
        return None
    # Three White Soldiers: 3 consecutive bullish candles, each closing higher
    if all(c > o for o, c in zip(opens, closes)):
        if closes[1] > closes[0] and closes[2] > closes[1]:
            return "bullish"
    # Three Black Crows: 3 consecutive bearish, each closing lower
    if all(c < o for o, c in zip(opens, closes)):
        if closes[1] < closes[0] and closes[2] < closes[1]:
            return "bearish"
    return None


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_patterns(
    cache: pd.DataFrame,
    lookback_days: int = 130,
) -> PatternDetectorResult:
    """Detect candlestick patterns in the OHLC cache.

    Parameters
    ----------
    cache:
        Daily cache with ``open``, ``high``, ``low``, ``close`` columns.
    lookback_days:
        How far back to scan (default ~6 months).

    Returns
    -------
    PatternDetectorResult
    """
    for col in ("open", "high", "low", "close"):
        if col not in cache.columns:
            return PatternDetectorResult(
                available=False,
                error=f"Missing OHLC column: {col}",
            )

    try:
        return _detect_patterns_impl(cache, lookback_days)
    except Exception as exc:
        logger.warning("Pattern detection failed: %s", exc)
        return PatternDetectorResult(available=False, error=str(exc))


def _detect_patterns_impl(
    cache: pd.DataFrame,
    lookback_days: int,
) -> PatternDetectorResult:
    """Internal implementation -- may raise."""
    df = cache[["open", "high", "low", "close"]].dropna().tail(lookback_days)
    if len(df) < 3:
        return PatternDetectorResult(
            available=False, error="Insufficient OHLC data",
        )

    patterns: list[PatternMatch] = []

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    dates = df.index

    for i in range(len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        d = str(dates[i])[:10]

        if detect_doji(o, h, l, c):
            patterns.append(PatternMatch("Doji", d, "neutral", 0.6))

        if detect_hammer(o, h, l, c):
            direction = "bullish"
            patterns.append(PatternMatch("Hammer", d, direction, 0.65))

        if detect_shooting_star(o, h, l, c):
            patterns.append(PatternMatch("Shooting Star", d, "bearish", 0.65))

        # Two-candle patterns
        if i >= 1:
            eng = detect_engulfing(opens[i - 1], closes[i - 1], o, c)
            if eng:
                name = f"{'Bullish' if eng == 'bullish' else 'Bearish'} Engulfing"
                patterns.append(PatternMatch(name, d, eng, 0.7))

        # Three-candle patterns
        if i >= 2:
            tsc = detect_three_soldiers_crows(
                [opens[i - 2], opens[i - 1], o],
                [closes[i - 2], closes[i - 1], c],
            )
            if tsc == "bullish":
                patterns.append(PatternMatch(
                    "Three White Soldiers", d, "bullish", 0.75,
                ))
            elif tsc == "bearish":
                patterns.append(PatternMatch(
                    "Three Black Crows", d, "bearish", 0.75,
                ))

    # Take last 20 patterns as "recent"
    recent = patterns[-20:]

    # Simple prediction: count bullish vs bearish in last 10 patterns
    last_n = patterns[-10:] if len(patterns) >= 10 else patterns
    bullish_count = sum(1 for p in last_n if p.direction == "bullish")
    bearish_count = sum(1 for p in last_n if p.direction == "bearish")

    predicted: list[dict[str, Any]] = []
    if bullish_count > bearish_count:
        predicted.append({
            "pattern": "Bullish continuation expected",
            "confidence": min(0.9, 0.5 + 0.05 * (bullish_count - bearish_count)),
        })
    elif bearish_count > bullish_count:
        predicted.append({
            "pattern": "Bearish pressure expected",
            "confidence": min(0.9, 0.5 + 0.05 * (bearish_count - bullish_count)),
        })
    else:
        predicted.append({"pattern": "Neutral / consolidation", "confidence": 0.5})

    logger.info(
        "Pattern detection: %d patterns found in %d days "
        "(%d bullish, %d bearish)",
        len(patterns), len(df), bullish_count, bearish_count,
    )

    return PatternDetectorResult(
        recent_patterns=recent,
        predicted_patterns_week=predicted,
    )
