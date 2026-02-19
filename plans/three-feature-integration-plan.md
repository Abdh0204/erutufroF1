# Integration Plan: Sentiment, Peer Ranking, Macro Quadrant

## Overview

Three new modules that inject daily columns into the cache before Step 6,
following the same pattern as `financial_health.py`. No existing modules
are modified except for wiring in `main.py` and adding profile sections.

```
Pipeline order (updated):
  Step 4:  cache_builder
  Step 5:  derived_variables, survival_mode, fuzzy_protection
  Step 5d: financial_health          (already done)
  Step 5e: news_sentiment            (NEW)
  Step 5f: peer_percentile_ranking   (NEW)
  Step 5g: macro_quadrant            (NEW)
  Step 6:  regime_detector -> forecasting -> forward_pass -> burnout -> MC -> aggregator
  Step 7:  profile_builder
  Step 8:  report_generator
```

All three features share the same integration pattern:
1. New module in `operator1/features/` or `operator1/models/`
2. Injects daily columns into cache DataFrame
3. Returns a result dataclass with summary stats
4. Added to `main.py` between Steps 5d and 6
5. Added to `extra_variables` list for temporal model learning
6. New section in profile_builder.py
7. Tests in `tests/`

---

## Feature 1: News Sentiment Scoring

### File: `operator1/features/news_sentiment.py`

**Data source:** FMP stock news API (`/stock_news?tickers=AAPL&limit=50`)
- FMP provides historical stock news with title, text, publishedDate
- No new API key needed -- uses existing FMP client

**Approach:**
- Fetch news for the target symbol from FMP (last 2 years)
- Use Gemini to batch-score sentiment (send 10-20 headlines at once, get JSON scores)
- Fall back to keyword-based scoring if Gemini unavailable
- Align news sentiment to daily cache via as-of logic (each day gets the latest sentiment)

**Cache columns injected:**
- `sentiment_score` -- daily sentiment (-1.0 to +1.0)
- `sentiment_count` -- number of articles that day
- `sentiment_momentum_5d` -- 5-day rolling mean of sentiment
- `sentiment_momentum_21d` -- 21-day rolling mean
- `sentiment_volatility_21d` -- 21-day rolling std of sentiment
- `is_missing_sentiment` -- companion flag

**Result dataclass: `SentimentResult`**
- `n_articles_scored: int`
- `mean_sentiment: float`
- `latest_sentiment: float`
- `latest_label: str` (Bearish/Neutral/Bullish)
- `columns_added: list[str]`

**Changes to existing files:**

1. `operator1/clients/fmp.py` -- add `get_stock_news(symbol, limit)` method:
   ```python
   def get_stock_news(self, symbol: str, limit: int = 200) -> pd.DataFrame:
       path = f"/stock_news?tickers={symbol}&limit={limit}"
       ...
   ```

2. `operator1/clients/gemini.py` -- add `score_sentiment(headlines)` method:
   ```python
   def score_sentiment(self, headlines: list[str]) -> list[float]:
       prompt = "Score each headline sentiment from -1 (bearish) to +1 (bullish)..."
       ...
   ```

3. `main.py` -- add Step 5e between financial_health and Step 6:
   ```python
   # Step 5e: News sentiment scoring
   sentiment_result = None
   try:
       from operator1.features.news_sentiment import compute_news_sentiment
       cache, sentiment_result = compute_news_sentiment(
           cache, fmp_client=fmp_client, gemini_client=gemini_client,
           symbol=args.symbol,
       )
   except Exception as exc:
       logger.warning("News sentiment failed: %s", exc)
   ```

4. `operator1/report/profile_builder.py` -- add `_build_sentiment_section()`

5. `_fh_extra_vars` collection in main.py -- extend to also include `sentiment_*` columns

**Tests: `tests/test_news_sentiment.py`**
- Test with mock FMP news data
- Test Gemini scoring fallback
- Test as-of alignment to daily cache
- Test cache column injection
- Test empty news produces NaN gracefully

---

## Feature 2: Peer Percentile Ranking

### File: `operator1/features/peer_ranking.py`

**Data source:** Existing linked entity caches (already built in Step 4)
- Uses `cache_result.linked_caches` from `build_all_caches()`
- No new API calls needed

**Approach:**
- For each day, collect the target value and all peer values for each variable
- Compute percentile rank of target within the peer group
- Uses `sector_peers` and `industry_peers` relationship groups from linked_aggregates

**Cache columns injected (for each variable V in AGGREGATE_VARIABLES):**
- `peer_pctile_{V}` -- percentile rank 0-100 (50 = median peer)
- `peer_zscore_{V}` -- z-score vs peers (0 = at peer mean)
- `peer_rank_composite` -- weighted average of all percentile ranks
- `peer_rank_label` -- Laggard / Below Average / Average / Above Average / Leader
- `is_missing_peer_pctile_{V}` -- companion flag

**Result dataclass: `PeerRankingResult`**
- `n_peers: int`
- `n_variables_ranked: int`
- `latest_composite_rank: float`
- `latest_label: str`
- `columns_added: list[str]`

**Changes to existing files:**

1. `operator1/steps/cache_builder.py` -- expose `linked_caches` in `CacheResult`:
   - The linked entity daily caches are already built but need to be
     accessible. Check if `CacheResult.linked_daily` already returns them.

2. `main.py` -- add Step 5f:
   ```python
   # Step 5f: Peer percentile ranking
   peer_result = None
   try:
       from operator1.features.peer_ranking import compute_peer_ranking
       cache, peer_result = compute_peer_ranking(
           cache, linked_caches=cache_result.linked_daily,
       )
   except Exception as exc:
       logger.warning("Peer ranking failed: %s", exc)
   ```

3. `operator1/report/profile_builder.py` -- add `_build_peer_ranking_section()`

4. `_fh_extra_vars` in main.py -- extend to include `peer_*` columns

**Tests: `tests/test_peer_ranking.py`**
- Test with 5 synthetic peer caches
- Test single peer (degenerate case)
- Test no peers (all NaN)
- Test percentile rank correctness
- Test cache column injection

---

## Feature 3: Macro Quadrant Mapping

### File: `operator1/features/macro_quadrant.py`

**Data source:** Existing macro_data from Step 2 (World Bank)
- GDP growth rate and inflation rate already fetched
- No new API calls needed

**Approach:**
- Classify each day into one of 4 macro quadrants based on:
  - GDP growth: above/below trend (using 10-year median as trend)
  - Inflation: above/below target (using 2% or country-specific target)
- Quadrants:
  - **Goldilocks**: growth above trend, inflation below target
  - **Reflation**: growth above trend, inflation above target
  - **Stagflation**: growth below trend, inflation above target
  - **Deflation**: growth below trend, inflation below target

**Cache columns injected:**
- `macro_quadrant` -- categorical label (goldilocks/reflation/stagflation/deflation)
- `macro_quadrant_numeric` -- 0/1/2/3 encoding for models
- `macro_growth_vs_trend` -- GDP growth minus trend (positive = above)
- `macro_inflation_vs_target` -- inflation minus target (positive = above)
- `macro_quadrant_stability_21d` -- how many days in the same quadrant over last 21 days
- `is_missing_macro_quadrant` -- companion flag

**Result dataclass: `MacroQuadrantResult`**
- `current_quadrant: str`
- `quadrant_distribution: dict[str, float]` (% of days in each)
- `n_transitions: int` (number of quadrant changes)
- `columns_added: list[str]`

**Changes to existing files:**

1. `main.py` -- add Step 5g:
   ```python
   # Step 5g: Macro quadrant mapping
   macro_quadrant_result = None
   try:
       from operator1.features.macro_quadrant import compute_macro_quadrant
       cache, macro_quadrant_result = compute_macro_quadrant(
           cache, macro_data=macro_data,
       )
   except Exception as exc:
       logger.warning("Macro quadrant mapping failed: %s", exc)
   ```

2. `operator1/report/profile_builder.py` -- add `_build_macro_quadrant_section()`

3. `_fh_extra_vars` in main.py -- extend to include `macro_*` columns

**Tests: `tests/test_macro_quadrant.py`**
- Test all 4 quadrant classifications
- Test with missing GDP or inflation
- Test quadrant stability calculation
- Test cache column injection
- Test transition counting

---

## Changes to Shared Infrastructure

### `main.py` extra_variables collection (updated)

```python
# Collect ALL injected feature columns for temporal model learning
_extra_vars = [
    c for c in cache.columns
    if (c.startswith("fh_") or c.startswith("sentiment_")
        or c.startswith("peer_") or c.startswith("macro_"))
    and cache[c].dtype in ("float64", "float32", "int64")
]
```

### `operator1/report/profile_builder.py`

Add 3 new section builders and parameters:
- `_build_sentiment_section(cache, sentiment_result)`
- `_build_peer_ranking_section(cache, peer_result)`
- `_build_macro_quadrant_section(cache, macro_quadrant_result)`

Add to `build_company_profile()` signature:
```python
sentiment_result: dict[str, Any] | None = None,
peer_ranking_result: dict[str, Any] | None = None,
macro_quadrant_result: dict[str, Any] | None = None,
```

### `operator1/models/__init__.py`

Update docstring to include new features.

---

## File Summary

| Action | File | Description |
|--------|------|-------------|
| NEW | `operator1/features/news_sentiment.py` | Sentiment scoring module |
| NEW | `operator1/features/peer_ranking.py` | Peer percentile ranking |
| NEW | `operator1/features/macro_quadrant.py` | Macro quadrant classifier |
| NEW | `tests/test_news_sentiment.py` | Sentiment tests |
| NEW | `tests/test_peer_ranking.py` | Peer ranking tests |
| NEW | `tests/test_macro_quadrant.py` | Macro quadrant tests |
| MODIFY | `operator1/clients/fmp.py` | Add `get_stock_news()` method |
| MODIFY | `operator1/clients/gemini.py` | Add `score_sentiment()` method |
| MODIFY | `main.py` | Wire Steps 5e/5f/5g, extend extra_variables |
| MODIFY | `operator1/report/profile_builder.py` | Add 3 section builders |
| MODIFY | `tests/test_phase7_report.py` | Update expected profile keys |

**Zero changes to:**
- forecasting.py (already supports extra_variables)
- regime_detector.py
- monte_carlo.py
- prediction_aggregator.py
- Any existing analysis modules
- Any existing test logic (only additions)

---

## Implementation Order

1. **Macro Quadrant** (simplest -- no API calls, uses existing data)
2. **Peer Ranking** (no API calls, uses existing linked caches)
3. **News Sentiment** (requires FMP news endpoint + Gemini scoring)

Each feature is independently deployable. If one fails, the pipeline
continues with `available: False` in the profile.
