# Phase D & E -- Implementation Plan

> Detailed implementation guide for the final two phases of the
> [INTEGRATION-GAP-ANALYSIS](./INTEGRATION-GAP-ANALYSIS.md).
> Phases A, B, and C are already complete. This document covers
> Phase D (Temporal Engine Enhancements) and Phase E (Gemini
> Integration & Polish).

---

## Prerequisites

All work builds on the following branches (already merged or ready to merge):

| Branch | What it delivered |
|--------|-------------------|
| `feature/phase-a-spec-conformance` | Corrected tier hierarchy weights, tier variable assignments, vanity adjustment target |
| `feature/phase-b-core-features` | 4 ethical filters, `canonical_fields.yml`, `country_survival_rules.yml`, macro stale flags, Technical Alpha mask |
| `feature/phase-c-math-modules` | Transformer, Particle Filter, Copula, Transfer Entropy, Genetic Algorithm, Sobol Sensitivity, Candlestick Pattern Detector, Wavelet/Fourier Decomposition |

The primary target files live in:
- `operator1/models/forecasting.py` (~1,300 lines -- forecasting models + burn-out)
- `operator1/models/prediction_aggregator.py` (~600 lines -- ensemble + multi-horizon)
- `operator1/clients/gemini.py` (~260 lines -- Gemini API wrapper)
- `operator1/clients/world_bank.py` (~160 lines -- World Bank indicator API)
- `operator1/report/report_generator.py` (~830 lines -- fallback template + chart gen + PDF)
- `notebooks/operator1.ipynb` (~20 cells)

---

## Phase D -- Temporal Engine Enhancements

**Estimated effort:** 3-5 days  
**Spec references:** Core Idea Sections E, K, L, M; Coding Instructions Section 10.7

### Current state

The forecasting pipeline (`run_forecasting` in `forecasting.py`) currently:

1. Iterates over each variable independently.
2. Tries models in a fallback chain (Kalman -> GARCH -> VAR -> LSTM -> Tree -> Baseline).
3. Picks the best single model per variable based on held-out RMSE.
4. Runs an optional `_burnout_refit` that shrinks the training window (100% / 75% / 50%) and picks the best window -- but does **not** do day-by-day predict-compare-update.
5. Multi-horizon forecasts in `prediction_aggregator.py` use sqrt-of-time scaling on single-step RMSE to widen confidence intervals -- not iterative chaining.

### What the spec requires

The PDF describes a fundamentally different temporal loop:

1. **Day-by-day forward pass** (Day 1 through ~500): predict t+1 from state at t, compare with cached actual at t+1, update model parameters, advance.
2. **Regime-weighted training windows**: `w(τ) ∝ exp(-Δt/half_life) * similarity(regime(τ), regime(t))`.
3. **Online/incremental parameter updates** after each compare step.
4. **Convergence-based burn-out** with early stopping when rolling error plateaus.
5. **Iterative multi-step prediction**: predict Day+1, feed that prediction as input to predict Day+2, repeat for week/month/year.

---

### D1: Day-by-day forward pass with predict-compare-update loop

**File:** `operator1/models/forecasting.py`  
**Effort:** High (~1-1.5 days)  
**Spec:** Core Idea E.1-E.4, Coding Instructions 10.7

#### What to build

A new top-level function `run_forward_pass` that replaces the current batch-fit-then-predict approach with a streaming day-by-day loop.

#### Design

```python
def run_forward_pass(
    cache: pd.DataFrame,
    tier_variables: dict[str, list[str]],
    hierarchy_weights: dict[str, float],
    regime_labels: pd.Series,
    *,
    warmup_days: int = 60,
    log_interval: int = 50,
) -> ForwardPassResult:
    """Day-by-day temporal analysis: predict -> compare -> update.

    Parameters
    ----------
    cache:
        Full 2-year daily cache (DatetimeIndex, ~500 rows).
    tier_variables:
        Mapping of tier name -> list of variable names.
    hierarchy_weights:
        Current tier weights from survival mode analysis.
    regime_labels:
        Per-day regime labels from HMM/GMM.
    warmup_days:
        Number of initial days used for cold-start fitting (no
        predict-compare during warmup).
    log_interval:
        Print progress every N days.

    Returns
    -------
    ForwardPassResult containing:
        - errors_by_tier: dict[int, list[float]] -- per-tier daily errors
        - errors_by_regime: dict[str, list[float]] -- per-regime errors
        - model_states: dict[str, Any] -- final model parameter snapshots
        - predictions_log: pd.DataFrame -- day-by-day predictions vs actuals
    """
```

#### Implementation steps

1. **Cold-start warmup** (days 0 to `warmup_days`):
   - Fit all models on the warmup window using existing `_fit_*` functions.
   - No prediction errors logged during warmup.

2. **Forward loop** (days `warmup_days` to `len(cache) - 1`):
   - For each day `t`:
     - **Step A -- Regime classification**: read `regime_labels[t]`.
     - **Step B -- Multi-module prediction**: call each fitted model's predict method for `t+1`.
     - **Step C -- Ensemble fusion**: weight predictions using inverse-RMSE weights, adjusted by tier hierarchy weights. Formula:
       ```
       adjusted_weight[model] = base_weight[model] * (tier_weight / 20.0)
       ```
     - **Step D -- Reality check**: compute `error = actual[t+1] - predicted[t+1]`, weighted by tier hierarchy weights.
     - **Step E -- Online update**: call each model's `update()` method (see D3).
     - **Step F -- Log**: append to `predictions_log` DataFrame and `errors_by_tier`.

3. **Result packaging**: aggregate error statistics, store final model states.

#### Data structure

```python
@dataclass
class ForwardPassResult:
    errors_by_tier: dict[int, list[float]]
    errors_by_regime: dict[str, list[float]]
    model_states: dict[str, Any]
    predictions_log: pd.DataFrame  # columns: date, variable, predicted, actual, error, tier, regime
    total_days: int = 0
    warmup_days: int = 0
```

#### Integration point

After `run_forecasting` is called (which initialises models), call `run_forward_pass` before burn-out. The forward pass replaces the current single-fit approach as the primary training mechanism. Keep `run_forecasting` as the initial model fitting step.

#### Tests

- Synthetic cache with known linear trend: verify errors decrease over time.
- Verify warmup days produce no error entries.
- Verify tier-weighted error aggregation matches expected formula.
- Verify predictions_log has correct shape (n_days * n_variables rows).

---

### D2: Regime-weighted historical training windows

**File:** `operator1/models/forecasting.py`  
**Effort:** Medium (~0.5-1 day)  
**Spec:** Core Idea Section L.1

#### What to build

A sample-weighting scheme that gives higher training weight to historical days with similar regimes.

#### Design

```python
def compute_regime_sample_weights(
    regime_labels: pd.Series,
    current_day_idx: int,
    *,
    half_life_days: int = 126,  # ~6 months
    regime_similarity_boost: float = 2.0,
) -> np.ndarray:
    """Compute per-day training weights for regime-aware learning.

    w(τ) ∝ exp(-Δt / half_life) * similarity(regime(τ), regime(t))

    Parameters
    ----------
    regime_labels:
        Series of regime labels (str) indexed by day position.
    current_day_idx:
        Index of the current day t in the forward pass.
    half_life_days:
        Exponential decay half-life in trading days.
    regime_similarity_boost:
        Multiplier for days sharing the same regime as day t.

    Returns
    -------
    1-D array of non-negative weights for days [0, current_day_idx].
    """
```

#### Implementation steps

1. Compute temporal decay: `decay[τ] = exp(-(t - τ) / half_life)` for all `τ <= t`.
2. Compute regime similarity: `sim[τ] = regime_similarity_boost` if `regime[τ] == regime[t]`, else `1.0`.
3. Final weight: `w[τ] = decay[τ] * sim[τ]`, then normalise to sum to 1.
4. Pass weights to model fitting functions:
   - **Kalman**: not directly sample-weighted; instead use weighted initialisation of Q/R matrices from regime-filtered data.
   - **VAR**: use `sample_weight` parameter in the OLS fitting (or pre-filter to regime-matched data).
   - **LSTM**: use `sample_weight` in the PyTorch DataLoader via `WeightedRandomSampler`.
   - **Tree ensemble**: pass `sample_weight` directly to sklearn's `fit()`.
   - **GARCH**: weight the log-likelihood (or pre-filter to regime-matched residuals).

#### Integration

- Called inside the forward pass loop (D1) before each model update step.
- Also used during burn-out (D4) to weight the training window.

#### Tests

- Verify weights sum to 1.0.
- Verify same-regime days get higher weight.
- Verify recent days get higher weight than old days.
- Edge case: all days same regime -- weights should be purely temporal decay.

---

### D3: Online/incremental model parameter updates

**File:** `operator1/models/forecasting.py`  
**Effort:** High (~1 day)  
**Spec:** Core Idea Sections L.2, M

#### What to build

An `update()` method on each model wrapper that incrementally adjusts parameters after seeing a new observation, without full refit.

#### Design

Each model needs to expose a common interface:

```python
class ModelWrapper(Protocol):
    def predict(self, state_t: np.ndarray) -> np.ndarray: ...
    def update(self, state_t: np.ndarray, actual_t_plus_1: np.ndarray,
               hierarchy_weights: dict[str, float]) -> None: ...
```

#### Per-model update strategies

| Model | Update Strategy | Complexity |
|-------|----------------|------------|
| **Kalman** | Native: Kalman gain update is already online. Add `self.update(z, H)` after each observation. | Low -- Kalman is inherently online |
| **GARCH** | Append new return observation, update conditional variance recursively: `σ²[t+1] = ω + α*ε²[t] + β*σ²[t]`. No full refit needed. | Low |
| **VAR** | Rolling OLS update: add new row, drop oldest if window exceeded. Periodic partial refit every N steps. | Medium |
| **LSTM** | Single-step gradient update (1 epoch, 1 sample) with current learning rate. Clip gradients to prevent explosion. | Medium |
| **Tree ensemble** | Cannot be updated online. Periodic full refit every N steps (e.g., every 50 days) with regime-weighted window. | Low (just schedule refits) |
| **Baseline** | Update EMA: `ema = α * actual + (1-α) * ema`. | Trivial |

#### Implementation steps

1. Create `ModelWrapper` abstract base class with `predict()` and `update()`.
2. Wrap each existing `_fit_*` function's output in the appropriate `ModelWrapper` subclass.
3. Implement `update()` for each model type as described above.
4. In the forward pass loop (D1), after Step D (reality check), call:
   ```python
   for model in active_models:
       model.update(state_t, actual_t_plus_1, hierarchy_weights)
   ```
5. Store `model_failed_update_<name>` flags if any update throws.

#### Hierarchy-weighted loss integration

When updating, weight the error signal by tier:
```python
weighted_error = sum(
    tier_weight * mse(predicted[tier_vars], actual[tier_vars])
    for tier, tier_vars in tier_variables.items()
) / sum(hierarchy_weights.values())
```

For LSTM specifically, use this as the loss function during the single-step update.

#### Tests

- Kalman: verify state vector changes after update.
- GARCH: verify conditional variance changes after new observation.
- LSTM: verify model weights change after single gradient step.
- Tree: verify periodic refit triggers at correct intervals.
- Verify `model_failed_update_*` flag is set on deliberate error injection.

---

### D4: Convergence detection for burn-out

**File:** `operator1/models/forecasting.py`  
**Effort:** Small (~0.5 day)  
**Spec:** Core Idea Section L.3

#### What to build

Replace the current `_burnout_refit` (window-shrinking heuristic) with a proper convergence-detecting burn-out loop that uses the forward pass.

#### Design

```python
def run_burnout(
    cache: pd.DataFrame,
    tier_variables: dict[str, list[str]],
    hierarchy_weights: dict[str, float],
    regime_labels: pd.Series,
    model_states: dict[str, Any],
    *,
    burnout_window: int = 130,  # ~6 months of trading days
    max_iterations: int = 10,
    patience: int = 3,
    validation_days: int = 20,
    learning_rate_multiplier: float = 2.0,
) -> BurnoutResult:
    """Intensive re-training on recent data with convergence detection.

    Each iteration:
    1. Reset to burnout_window days ago.
    2. Initialise from model_states (pattern from previous iteration).
    3. Run forward pass with higher learning rates.
    4. Measure accuracy on last validation_days.
    5. If accuracy improves, save as new best pattern.
    6. Stop early if no improvement for patience iterations.
    """
```

#### Implementation steps

1. Extract the last `burnout_window` days from cache.
2. For each iteration (up to `max_iterations`):
   a. Clone model states from previous best.
   b. Run `run_forward_pass` on the burnout window with `learning_rate_multiplier` applied.
   c. Evaluate on the last `validation_days` using tier-weighted RMSE.
   d. Track improvement: if RMSE improved, save as `pattern_best`. If not, increment `no_improve_count`.
   e. If `no_improve_count >= patience`, break.
3. Return `BurnoutResult` with final model states, iteration count, convergence flag.

#### Data structure

```python
@dataclass
class BurnoutResult:
    model_states: dict[str, Any]
    iterations_completed: int
    converged: bool  # True if early-stopped due to patience
    best_rmse_by_tier: dict[int, float]
    rmse_history: list[float]  # per-iteration validation RMSE
    learning_rate_multiplier: float
```

#### Difference from current `_burnout_refit`

| Aspect | Current | New |
|--------|---------|-----|
| Strategy | Window shrinking (100%/75%/50%) | Iterative forward pass with online updates |
| Iterations | 3 (fixed) | Up to 10, with early stopping |
| Learning | Single batch refit | Day-by-day with higher learning rate |
| Convergence | Pick lowest RMSE across 3 windows | Track rolling RMSE, stop on plateau |
| Regime awareness | None | Regime-weighted sample weights (D2) |

#### Tests

- Verify early stopping triggers after `patience` non-improving iterations.
- Verify `converged=True` when early-stopped, `converged=False` when max iterations hit.
- Verify `rmse_history` has length equal to `iterations_completed`.
- Verify `learning_rate_multiplier` is applied (mock check on LSTM optimizer lr).

---

### D5: Iterative multi-step prediction (chain day predictions)

**File:** `operator1/models/prediction_aggregator.py`  
**Effort:** Medium (~1 day)  
**Spec:** Core Idea Section E.4, Phase 4

#### What to build

An `iterative_multi_step_predict` function that replaces the current sqrt-scaling approximation with true iterative chaining.

#### Current approach (to replace for horizons > 1d)

```python
# Current: scale CI by sqrt of horizon
ci_width = rmse * Z_SCORE_90 * math.sqrt(horizon_days)
```

#### New approach

```python
def iterative_multi_step_predict(
    models: dict[str, ModelWrapper],
    ensemble_weights: dict[str, float],
    initial_state: pd.Series,
    hierarchy_weights: dict[str, float],
    horizon_days: int,
    mc_scenarios: int = 1000,
) -> list[StepPrediction]:
    """Generate multi-step predictions by iterative chaining.

    For each step s in [1, horizon_days]:
    1. Use ensemble to predict state at step s from state at step s-1.
    2. Add noise sampled from regime-conditional distribution.
    3. Feed predicted state as input for step s+1.
    4. Collect all intermediate predictions.

    Parameters
    ----------
    models:
        Dict of model_name -> ModelWrapper (from forward pass / burn-out).
    ensemble_weights:
        Normalised weights per model.
    initial_state:
        The last observed day's state vector.
    hierarchy_weights:
        Current tier weights for weighting predictions.
    horizon_days:
        Number of days to predict forward.
    mc_scenarios:
        Number of Monte Carlo paths for uncertainty estimation.

    Returns
    -------
    List of StepPrediction, one per day in the horizon.
    """
```

#### Implementation steps

1. Create `StepPrediction` dataclass:
   ```python
   @dataclass
   class StepPrediction:
       step: int  # 1-indexed
       point_forecast: dict[str, float]  # variable -> value
       lower_ci: dict[str, float]  # 5th percentile
       upper_ci: dict[str, float]  # 95th percentile
       regime_probabilities: dict[str, float]
   ```

2. **Point forecast chain** (deterministic path):
   - `state_0 = initial_state`
   - For `s in range(1, horizon_days + 1)`:
     - `predicted_s = weighted_average(model.predict(state_{s-1}) for model in models)`
     - `state_s = predicted_s`
   - Store each step's prediction.

3. **Monte Carlo uncertainty** (stochastic paths):
   - Run `mc_scenarios` parallel chains.
   - At each step, add noise: `state_s = predicted_s + noise` where noise is sampled from regime-conditional return distribution.
   - Collect all terminal and intermediate states.
   - Compute percentiles (5th, 50th, 95th) per variable per step.

4. **Technical Alpha protection**:
   - For step 1 (next day): mask Open, High, Close; only expose Low.
   - For steps 2+ (week, month, year): expose full OHLC predictions.

5. **Integration**: Wire `iterative_multi_step_predict` into `run_prediction_aggregation`. Generate horizon-specific outputs:
   - `predictions_next_day`: step 1 only.
   - `predictions_next_week`: steps 1-5 as OHLC series.
   - `predictions_next_month`: steps 1-21 as OHLC series.
   - `predictions_next_year`: steps 1-252, aggregated monthly to reduce clutter.

#### Backward compatibility

Keep the existing sqrt-scaling as a fallback. If iterative chaining fails (e.g., model diverges), fall back to sqrt-scaling with a warning flag.

#### Tests

- Verify chain produces exactly `horizon_days` step predictions.
- Verify uncertainty bands widen over time (later steps have wider CIs).
- Verify Technical Alpha mask applies to step 1 only.
- Verify Monte Carlo paths produce sensible percentile distributions.
- Verify fallback to sqrt-scaling on deliberate model failure.

---

## Phase E -- Gemini Integration & Polish

**Estimated effort:** 1-2 days  
**Spec references:** Core Idea Section G; Coding Instructions Section 12, 13

### Current state

- `gemini.py` has a `generate_report()` method that sends a **9-section** prompt to Gemini.
- `report_generator.py` calls `gemini_client.generate_report()` if a client is provided, otherwise uses a local fallback template.
- The fallback template has 9 sections, not the 13 required by the PDF.
- No validation of Gemini output completeness.
- No World Bank WDS Documents API integration.
- The notebook has ~20 cells but does not follow the Section 13 cell arrangement exactly.

---

### E1: Full 13-section Gemini report prompt

**File:** `operator1/clients/gemini.py`  
**Effort:** Medium (~0.5-1 day)  
**Spec:** Coding Instructions Section 12.1, Core Idea Section G.2

#### What to change

Replace the current `_REPORT_PROMPT` (a simple 9-section request) with the full 13-section structured prompt from the PDF spec.

#### Current prompt (simplified)

```
You are a senior financial analyst...
The report MUST include:
1. Executive Summary
2. Company Overview
3. Financial Health Assessment
4. Survival Analysis
5. Linked Entities & Relative Positioning
6. Macro Environment Impact
7. Regime Analysis & Forecasts
8. Risk Assessment
9. LIMITATIONS
```

#### New prompt structure (from PDF Section 12.1)

The new prompt must include three parts:

**Part 1 -- Role & Context:**
- Bloomberg-style equity research analyst role.
- Mention the 20+ mathematical modules, survival mode, ethical filters, multi-horizon predictions.

**Part 2 -- 13 Required Sections:**

| # | Section | Key content |
|---|---------|-------------|
| 1 | Executive Summary | 3 bullet points, BUY/HOLD/SELL recommendation, 12-month target price |
| 2 | Company Overview | Identity, market position, sector context |
| 3 | Historical Performance Analysis | Total vs real return, Sharpe ratio, regime breakdown, structural breaks |
| 4 | Current Financial Health | **Tier-by-tier breakdown** with hierarchy explanation and current weights |
| 5 | Survival Mode Analysis | Current status, historical episodes, vanity analysis |
| 6 | Linked Variables & Market Context | Sector/industry/competitor positioning, macro environment |
| 7 | Temporal Analysis & Model Insights | Current regime, model performance by tier, best module |
| 8 | Predictions & Forecasts | Next day (masked OHLC), week, month, year with Monte Carlo uncertainty |
| 9 | Technical Patterns & Chart Analysis | Historical + predicted candlestick patterns |
| 10 | Ethical Filter Assessment | All 4 filters with verdicts + overall ethical score |
| 11 | Risk Factors & Limitations | Model assumptions, key risks, data quality, black swan caveat |
| 11.1 | Limitations (short, required) | Data window, OHLCV caveats, macro frequency, missingness, failed modules |
| 12 | Investment Recommendation | BUY/HOLD/SELL + confidence + target price + entry/exit strategy + position sizing |
| 13 | Appendix | Methodology summary, tier definitions, glossary, data sources |

**Part 3 -- Profile Data Injection:**
- Inject `company_profile` as JSON.
- Formatting requirements: markdown, tables, bold metrics, chart placeholders, 8,000-12,000 words target.

#### Implementation steps

1. Replace `_REPORT_PROMPT` class variable in `GeminiClient` with the full 13-section prompt.
2. Use f-string interpolation to inject dynamic values into the prompt (hierarchy weights, regime, survival flags, filter verdicts) -- the PDF shows these as inline references.
3. Update `generationConfig` to:
   ```python
   {
       "temperature": 0.3,
       "maxOutputTokens": 16000,  # up from current default
   }
   ```
4. Add timeout of 120 seconds for the Gemini API call (report generation takes longer than search/mapping calls).

#### Also update fallback template

Update `_FALLBACK_TEMPLATE` in `report_generator.py` to match the 13-section structure so that offline reports are structurally equivalent.

#### Tests

- Verify prompt contains all 13 section headers.
- Verify dynamic values are interpolated correctly.
- Verify `maxOutputTokens` is set to 16000.
- Verify timeout is 120s.

---

### E2: Gemini response validation

**File:** `operator1/report/report_generator.py`  
**Effort:** Small (~0.5 day)  
**Spec:** Core Idea Section G.3

#### What to build

A validation function that checks the Gemini output for completeness and data integrity.

#### Design

```python
def validate_gemini_report(
    markdown: str,
    profile: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate that a Gemini-generated report is complete and accurate.

    Checks:
    1. All 13 required sections are present (by header text).
    2. Investment recommendation is present (BUY/HOLD/SELL keyword).
    3. LIMITATIONS section exists.
    4. No hallucinated numbers: spot-check key metrics against profile.

    Returns
    -------
    (is_valid, list_of_issues)
    """
```

#### Implementation steps

1. **Section presence check**: scan for header patterns matching the 13 required sections (case-insensitive regex).
2. **Recommendation check**: search for "BUY", "HOLD", or "SELL" in the report text.
3. **LIMITATIONS check**: verify the word "LIMITATIONS" appears as a section header.
4. **Metric spot-check**: extract a few key numbers from the report (e.g., debt-to-equity, volatility, FCF yield) and compare against the profile dict. Flag if any differ by more than 10% (Gemini may round, so allow tolerance).
5. **Integration**: call `validate_gemini_report` after Gemini returns. If validation fails, log warnings but still use the report (append missing sections from fallback if needed).

#### Tests

- Provide a valid 13-section markdown: verify passes.
- Remove section 10: verify failure with clear message.
- Inject wrong debt-to-equity value: verify metric mismatch is flagged.

---

### E3: World Bank WDS Documents API integration

**File:** `operator1/clients/world_bank.py`  
**Effort:** Small (~0.5 day)  
**Spec:** Core Idea Section D.1 (Country enrichment)

#### What to build

A new method on `WorldBankClient` that queries the World Bank Documents & Reports API (WDS) to fetch country-specific documents for qualitative context.

#### Design

```python
def search_wds_documents(
    self,
    country_iso2: str,
    *,
    sector_keywords: list[str] | None = None,
    max_results: int = 10,
    recent_years: int = 3,
) -> list[dict[str, Any]]:
    """Search World Bank Documents & Reports API for country context.

    Base endpoint: https://search.worldbank.org/api/v3/wds

    Parameters
    ----------
    country_iso2:
        ISO-2 country code.
    sector_keywords:
        Optional sector/industry terms to narrow results.
    max_results:
        Maximum documents to return.
    recent_years:
        Only fetch documents from the last N years.

    Returns
    -------
    List of document metadata dicts with keys:
        - title, abstract, doc_type, date, url, country, sector
    """
```

#### Implementation steps

1. Build query URL:
   ```
   https://search.worldbank.org/api/v3/wds
     ?format=json
     &qterm={country_name}+{sector_keywords}
     &count_exact={country_iso2}
     &lang_exact=English
     &rows={max_results}
     &strdate={start_date}
     &fl=title,abstracts,docty,docdt,url,count
     &sort=docdt
     &order=desc
   ```
2. Parse JSON response, extract document metadata.
3. Store results as `cache/wds_documents.json` for profile builder consumption.
4. Add a lightweight feature extraction: count documents by type (e.g., "Economic Update", "Country Partnership Framework"), store as `wds_doc_count_by_type` in the profile's linked_variables.macro section.

#### Integration

- Call after macro indicators are fetched (in the data ingestion phase).
- Pass results to the profile builder as an additional `wds_context` field.
- The report template (both Gemini and fallback) can reference this for the "Macro Environment" section.

#### Error handling

- If WDS API fails, log warning and continue. This is additive context, not critical data.
- Respect rate limits (the WDS API is public and generally permissive).

#### Tests

- Mock WDS response: verify parsing extracts expected fields.
- Verify date filtering works (only recent documents).
- Verify graceful degradation on API failure.

---

### E4: Notebook cell arrangement per Section 13

**File:** `notebooks/operator1.ipynb`  
**Effort:** Medium (~0.5-1 day)  
**Spec:** Coding Instructions Section 13.1-13.3

#### What to change

The current notebook has ~20 cells. The PDF specifies a particular cell arrangement with:
1. One purpose per cell.
2. Idempotent cells (check for cached artifacts before re-running).
3. Hard boundaries between API calls and feature engineering.
4. Optional cells with explicit guards.
5. `RUN_OPTIONAL` dict for toggling optional sections.

#### Current cell layout (approximate)

```
Cell 0:  Intro markdown
Cell 1:  Inputs (ISIN, FMP symbol)
Cell 2:  Secrets
Cell 3:  Global config
Cell 4:  HTTP utils
Cell 5:  Eulerpool client
Cell 6:  FMP client
Cell 7:  Identifier verification
Cell 8:  World Bank macro
Cell 9:  Linked entity discovery (Gemini)
Cell 10: Linked entity resolution
Cell 11: Target OHLCV (FMP)
Cell 12: Target fundamentals (as-of)
Cell 13: Feature engineering
Cell 14: Linked caches
...
```

#### Required changes

1. **Add `RUN_OPTIONAL` guard cell** (new Cell 4.5):
   ```python
   RUN_OPTIONAL = {
       "linked_entities": True,
       "intraday": False,
       "heavy_models": True,
       "pdf": False,
   }
   ```

2. **Add idempotency checks** to all data-fetching cells:
   ```python
   if Path("cache/target_ohlcv.parquet").exists() and not FORCE_REBUILD:
       print("Loading cached OHLCV...")
       target_ohlcv = pd.read_parquet("cache/target_ohlcv.parquet")
   else:
       # ... fetch from FMP
   ```

3. **Add optional cell guards** around linked entity cells:
   ```python
   if not RUN_OPTIONAL["linked_entities"]:
       print("Skipping linked entities (disabled)")
   else:
       # ... discovery + resolution
   ```

4. **Add separate modeling cells** (one per model, wrapped in try/except):
   - Cell N+1: Regime detection (HMM/GMM/PELT/BCP)
   - Cell N+2: Forward pass + burn-out
   - Cell N+3: Prediction aggregation + iterative chaining
   - Cell N+4: Monte Carlo simulation
   - Each wrapped in:
     ```python
     try:
         result = run_model(...)
     except Exception as e:
         print(f"Model failed: {e}")
         model_failures.append(("model_name", str(e)))
     ```

5. **Add Gemini report cell** (with WDS integration):
   ```python
   # Cell: Report generation (Gemini)
   gemini_client = GeminiClient(api_key=GEMINI_API_KEY)
   report_result = generate_report(
       profile=company_profile,
       gemini_client=gemini_client,
       cache=target_cache,
       generate_pdf=RUN_OPTIONAL["pdf"],
   )
   ```

6. **Add final summary cell** with output file listing.

#### Tests

- Run notebook with `FORCE_REBUILD = False` and verify it loads from cache.
- Run with `RUN_OPTIONAL["linked_entities"] = False` and verify it skips gracefully.
- Verify all cells are idempotent (re-run doesn't corrupt state).

---

## Execution Order & Dependencies

```
D1 (forward pass)
 ├── D2 (regime-weighted windows) -- used inside D1's loop
 ├── D3 (online updates) -- called at each step of D1's loop
 └── D4 (convergence burn-out) -- runs after D1, uses D1's infrastructure
       └── D5 (iterative multi-step) -- uses model states from D4

E1 (Gemini prompt) -- independent
E2 (validation) -- depends on E1
E3 (WDS documents) -- independent
E4 (notebook) -- depends on all D tasks + E1-E3 being wired in
```

**Recommended implementation sequence:**

1. **D3** first (model wrappers with `update()`) -- foundational.
2. **D2** next (regime weights) -- self-contained utility.
3. **D1** (forward pass) -- combines D2 + D3.
4. **D4** (burn-out) -- builds on D1.
5. **D5** (iterative chaining) -- uses model states from D4.
6. **E1** (Gemini prompt) -- independent, can be done in parallel with D tasks.
7. **E3** (WDS) -- independent, small.
8. **E2** (validation) -- after E1.
9. **E4** (notebook) -- final assembly after everything is wired.

---

## Testing Strategy

| Task | Unit Tests | Integration Tests |
|------|-----------|-------------------|
| D1 | Synthetic cache forward pass, error tracking | Full pipeline on sample data |
| D2 | Weight computation, edge cases | Verify weights flow into D1 correctly |
| D3 | Per-model update methods | State changes verified in D1 loop |
| D4 | Early stopping, convergence flag | D4 output feeds into D5 |
| D5 | Chain length, CI widening, TA mask | Compare against sqrt-scaling baseline |
| E1 | Prompt contains all 13 sections | Gemini mock returns valid report |
| E2 | Section detection, metric spot-check | Validates against Gemini mock output |
| E3 | WDS response parsing, date filter | Additive -- verify no crash on failure |
| E4 | Cell idempotency, optional guards | Notebook kernel restart test |

**Test file naming:**
- `tests/test_phase_d_forward_pass.py` (D1-D4)
- `tests/test_phase_d_iterative_predict.py` (D5)
- `tests/test_phase_e_gemini.py` (E1-E2)
- `tests/test_phase_e_wds.py` (E3)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Forward pass is too slow (~500 days * 5 models) | Use lightweight online updates (not full refit). Profile and set a time budget (~10 min max for forward pass). |
| LSTM diverges during online update | Gradient clipping (`clip_grad_norm_`), learning rate cap, fallback to tree/baseline if loss becomes NaN. |
| Iterative chaining accumulates errors | Monte Carlo uncertainty widening naturally captures this. Add explicit divergence detection (if predicted values go out of 5-sigma range, clip). |
| Gemini rate limiting / token limit | Retry with exponential backoff. If profile is too large for token limit, summarise historical section before sending. |
| WDS API changes / unavailable | Additive feature only -- pipeline continues without it. Cache results on disk. |

---

## Estimated Timeline

| Day | Tasks | Deliverable |
|-----|-------|-------------|
| 1 | D3 (model wrappers), D2 (regime weights) | ModelWrapper protocol + regime weight utility |
| 2 | D1 (forward pass) | `run_forward_pass` with predict-compare-update loop |
| 3 | D4 (convergence burn-out), D5 start | `run_burnout` + begin iterative chaining |
| 4 | D5 complete, E1 (Gemini prompt) | `iterative_multi_step_predict` + 13-section prompt |
| 5 | E2 (validation), E3 (WDS), E4 (notebook) | Report validation + WDS client + notebook polish |

**Total: ~5 working days** for both phases.
