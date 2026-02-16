# Operator 1 -- Gap Analysis & Integration Plan

> Comparing the two specification PDFs (`The_Apps_core_idea.pdf` and
> `The_Apps_coding_instructions.pdf`) against the eight feature branches
> that exist in the repository.

---

## 1. Branch Inventory (what exists today)

| Branch | Phase | Key Files | Lines of Code |
|--------|-------|-----------|---------------|
| `feature/phase1-foundation` | Project scaffold, configs, HTTP utils, API clients | `operator1/clients/*.py`, `config/*.yml`, `http_utils.py` | ~1,200 |
| `feature/phase2-data-ingestion` | Identifier verification, macro mapping, entity discovery, data extraction | `operator1/steps/*.py` | ~1,100 |
| `feature/phase3-cache-features` | Daily cache builder, derived variables, linked aggregates, macro alignment | `operator1/features/*.py`, `cache_builder.py` | ~1,470 |
| `feature/phase4-analysis-logic` | Survival mode detection, hierarchy weights, vanity analysis, data quality | `operator1/analysis/*.py`, `quality/data_quality.py` | ~1,300 |
| `feature/phase5-estimation` | Two-pass estimation engine (identity fill + rolling imputation) | `operator1/estimation/estimator.py` | 675 |
| `feature/phase6-regime-detection` | HMM, GMM, PELT, BCP, Granger causality | `operator1/models/regime_detector.py`, `causality.py` | ~1,080 |
| `feature/phase6-prediction-aggregation` | Kalman, GARCH, VAR, LSTM, tree ensemble, baseline, burn-out, Monte Carlo, ensemble aggregation | `operator1/models/forecasting.py`, `monte_carlo.py`, `prediction_aggregator.py` | ~3,090 |
| `feature/phase7-report-assembly` | Profile builder, report generator (markdown + charts + PDF), Kaggle notebook | `operator1/report/*.py`, `notebooks/operator1.ipynb` | ~2,040 |

**Total implementation: ~12,000 lines across 25+ Python modules.**

---

## 2. Tier Hierarchy Weight Mismatch (Critical)

The PDF spec defines exact weights per survival regime. The current
implementation in `config/survival_hierarchy.yml` uses different values.

| Regime | PDF Spec (Tier 1/2/3/4/5) | Current Code | Delta |
|--------|---------------------------|--------------|-------|
| Normal | 20 / 20 / 20 / 20 / 20 | 15 / 15 / 20 / 25 / 25 | Tier 1-2 under-weighted, Tier 4-5 over-weighted |
| Company Survival | 50 / 30 / 15 / 4 / 1 | 35 / 25 / 20 / 15 / 5 | All tiers diverge significantly |
| Modified Survival (country crisis, company healthy) | 40 / 35 / 20 / 4 / 1 | 25 / 25 / 20 / 20 / 10 | Major divergence |
| Extreme Survival (both in crisis) | 60 / 30 / 10 / 0 / 0 | 40 / 30 / 20 / 8 / 2 | Tier 1 under-weighted, Tier 4-5 should be 0 |
| Vanity adjustment (>10%) | +5% to Tier 1, -2% Tier 4, -3% Tier 5 | +5% Tier 3, -5% Tier 5 | Adjusts wrong tier |

### Tier Variable Mapping Mismatch

The PDF defines specific tier assignments that differ from the implementation:

| Variable | PDF Tier | Code Tier | Impact |
|----------|----------|-----------|--------|
| `current_ratio` | Tier 2 (Solvency) | Tier 1 (Liquidity) | Moderate |
| `free_cash_flow_ttm`, `fcf_yield` | Tier 1 (Liquidity & Cash) | Tier 3 (Cash Flow) | High -- FCF should be highest priority |
| `operating_cash_flow` | Tier 1 (Liquidity & Cash) | Tier 3 (Cash Flow) | High |
| `volatility_21d` | Tier 3 (Market Stability) | Tier 4 (Profitability) | High -- volatility is market, not profitability |
| `drawdown_252d` | Tier 3 (Market Stability) | Tier 5 (Valuation) | High -- drawdown is market, not valuation |
| `volume` | Tier 3 (Market Stability) | Not mapped | Missing |
| `interest_coverage` | Tier 2 (Solvency) | Tier 2 | Correct |
| `revenue_asof` | Tier 5 (Growth) | Tier 5 | Correct |

**Where to fix:** `config/survival_hierarchy.yml` -- update both the
`regimes` weight presets and the `tiers` variable lists to match the PDF
exactly.

---

## 3. Missing Config Files

The coding instructions PDF (Section 3.2, 11.6) specifies config files that
do not exist in any branch:

| Missing Config | PDF Reference | Purpose | Where to Add |
|----------------|---------------|---------|--------------|
| `config/canonical_fields.yml` | Coding Instructions Section 3.2 | Maps every variable to: category (decision/linked), source endpoint, extraction mode (direct/derived), formula if derived | `config/` directory |
| `config/country_survival_rules.yml` | Coding Instructions Section 11.6 | Defines how to compute `country_survival_mode_flag` from macro thresholds | `config/` directory (currently hardcoded in `survival_mode.py`) |

---

## 4. Missing Ethical Filters (Critical -- core product feature)

The four ethical filters are a **defining feature** of Operator 1 per the PDF
(Core Idea Section G, Coding Instructions Section 11). They are referenced
throughout the report template but **not implemented** in the profile builder
or as standalone computation modules.

### 4.1 Purchasing Power Filter
- **Spec:** Compare nominal return vs real return (inflation-adjusted).
  Verdict: PASS / FAIL / WARNING.
- **Status:** `real_return_1d` is computed in `macro_alignment.py`, but no
  filter verdict logic exists. The profile builder does not produce a
  `filters.purchasing_power` section.
- **Where:** Create `operator1/analysis/ethical_filters.py` with
  `compute_purchasing_power_filter()`. Wire into `profile_builder.py`.

### 4.2 Solvency Filter (Debt-to-Equity / Riba)
- **Spec:** Flag companies with D/E > 3.0 as "Fragile". Thresholds: < 1.0
  Conservative, < 2.0 Stable, < 3.0 Warning, >= 3.0 Fail.
- **Status:** `debt_to_equity` is computed, survival mode uses it, but there
  is no standalone solvency filter verdict.
- **Where:** Same `ethical_filters.py`. Use `debt_to_equity_abs` for
  threshold checks.

### 4.3 Gharar Filter (Volatility / Speculation)
- **Spec:** Score 0-10 stability score based on `volatility_21d`. Verdicts:
  LOW / MODERATE / HIGH / EXTREME.
- **Status:** Not implemented.
- **Where:** Same `ethical_filters.py`. Map volatility ranges to stability
  scores and verdicts.

### 4.4 Cash is King Filter (Free Cash Flow Yield)
- **Spec:** Verdicts based on `fcf_yield`: > 5% Strong, > 2% Healthy, > 0
  Weak, <= 0 Burning Cash.
- **Status:** `fcf_yield` is computed but no filter verdict.
- **Where:** Same `ethical_filters.py`.

### Integration Point
The profile builder (`operator1/report/profile_builder.py`) needs a new
`_build_ethical_filters_section()` that calls the filter functions and
produces the `filters` key in the profile JSON. The report generator
(`report_generator.py`) already has placeholders expecting this data in
Section 10 of the Gemini prompt.

---

## 5. Missing Mathematical Modules (8 of 20+)

The PDF (Core Idea Section E.2) specifies 20+ mathematical modules. The
current implementation covers roughly half. Here is what is missing:

### 5.1 Implemented (in `operator1/models/`)

| Module | PDF Category | Status |
|--------|-------------|--------|
| HMM (Hidden Markov Model) | Cat 1: Regime Detection | Done (`regime_detector.py`) |
| GMM (Gaussian Mixture Model) | Cat 1: Regime Detection | Done (`regime_detector.py`) |
| BCP (Bayesian Change Point) | Cat 1: Structural Breaks | Done (`regime_detector.py`) |
| PELT (Pruned Exact Linear Time) | Cat 1: Structural Breaks | Done (`regime_detector.py`) |
| Adaptive Kalman Filter | Cat 2: Forecasting Core | Done (`forecasting.py`) |
| GARCH / EGARCH | Cat 2: Volatility Modeling | Done (`forecasting.py`) |
| VAR (Vector Autoregression) | Cat 2: Multi-Variable | Done (`forecasting.py`) |
| LSTM | Cat 2: Deep Learning | Done (`forecasting.py`) |
| Random Forest / XGBoost | Cat 4: Ensemble | Done (`forecasting.py`) |
| Granger Causality | Cat 3: Causality | Done (`causality.py`) |
| Regime-Aware Monte Carlo | Cat 5: Monte Carlo | Done (`monte_carlo.py`) |
| Importance Sampling MC | Cat 5: Monte Carlo | Done (`monte_carlo.py`) |
| Baseline (last-value / EMA) | Fallback | Done (`forecasting.py`) |

### 5.2 NOT Implemented

| Missing Module | PDF Category | Complexity | Priority | Where to Add |
|----------------|-------------|------------|----------|--------------|
| **Transformer Architecture** | Cat 2: Forecasting (attention-based) | High | Medium | `operator1/models/forecasting.py` -- add `fit_transformer()` alongside LSTM |
| **Particle Filter (Sequential MC)** | Cat 2: Non-linear state estimation | Medium | Medium | `operator1/models/forecasting.py` -- add `fit_particle_filter()` |
| **Copula Models** | Cat 3: Dependency structures | High | Low | New file `operator1/models/copula.py` |
| **Transfer Entropy / Information Flow** | Cat 3: Causality | Medium | Low | `operator1/models/causality.py` -- add `compute_transfer_entropy()` |
| **Genetic Algorithm (meta-optimization)** | Cat 4: Ensemble optimization | Medium | Medium | `operator1/models/prediction_aggregator.py` -- add GA-based weight optimization |
| **Sobol Sensitivity Analysis** | Cat 4: Variance decomposition | Medium | Low | New file `operator1/models/sensitivity.py` |
| **Candlestick Pattern Detector** | Cat 6: Chart patterns | Medium | High | New file `operator1/models/pattern_detector.py` -- use TA-Lib or rule-based |
| **Wave/Cycle Decomposition (Fourier/Wavelet)** | Cat 6: Frequency analysis | Medium | Low | New file `operator1/models/cycle_decomposition.py` |

### Integration Strategy for Missing Modules

Each missing module should follow the existing pattern:
1. Implement as a function in the appropriate file.
2. Wrap in `try/except` (per the spec's reliability rules).
3. Return a `ForecastResult` or equivalent dataclass.
4. Register with the ensemble in `run_forecasting()`.
5. Add fallback behavior (degrade gracefully on failure).
6. Add the dependency to `requirements.txt` (currently most are commented out).

---

## 6. Missing World Bank Documents & Reports API (WDS)

The PDF (Core Idea Section D, "Country enrichment") specifies using the World
Bank Documents & Reports search API at
`https://search.worldbank.org/api/v3/wds` to discover country-specific
public documents for qualitative context.

- **Status:** Not implemented. The `world_bank.py` client only handles
  indicator data.
- **Priority:** Low (additive source, explicitly labeled as "non-invasive"
  in the PDF).
- **Where:** Add a `fetch_wds_documents()` method to
  `operator1/clients/world_bank.py`. Store results as metadata in
  `cache/metadata.json`.

---

## 7. Missing Macro Stale Flags

The PDF (Core Idea Section D.2) requires that for each macro variable, the
system stores:
- `macro_asof_date_<var>` -- when the value was last published
- `is_stale_<var>` -- True if the value is older than 365 days

**Status:** Not implemented in `operator1/features/macro_alignment.py`.
The module does as-of alignment but does not compute staleness.

**Where:** Add staleness computation in `macro_alignment.py` after the
forward-fill step. For each variable, compare the original publication date
against the current row date and flag if delta > 365 days.

---

## 8. Missing Online/Incremental Learning

The PDF (Core Idea Sections L and M) describes a continuous learning loop:
- **Regime-weighted historical windows** where training weight is
  `w(t) ~ exp(-dt/half_life) * similarity(regime(t), regime(t_current))`
- **Online evaluation**: predict t+1, compare to cache, update model
  parameters incrementally.
- **Convergence detection**: stop when rolling error plateaus.

**Status:** The burn-out implementation in `forecasting.py` (`_burnout_refit`)
retrains the LSTM on the last ~126 days but does NOT implement:
- Online/incremental model parameter updates.
- Regime-weighted sample weighting.
- The predict-compare-update loop during the forward pass.
- Convergence detection.

**Where:** `operator1/models/forecasting.py` -- enhance `run_forecasting()` to
add a proper forward pass with incremental learning. The current
implementation runs a single burn-out refit rather than iterative day-by-day
refinement.

**Priority:** High -- this is the core temporal analysis engine behavior
described in Section E.4 of the PDF.

---

## 9. Missing or Incomplete Multi-Horizon Iterative Prediction

The PDF (Core Idea Section E.4, Phase 4) specifies that multi-step
predictions should be **iterative**:
- Predict Day +1
- Use Day +1 prediction as input to predict Day +2
- Repeat for 5d, 21d, 252d

**Status:** The prediction aggregator (`prediction_aggregator.py`) computes
multi-horizon forecasts by scaling single-step forecasts (sqrt-of-time
scaling), which is a shortcut rather than the iterative approach the PDF
describes.

**Where:** `operator1/models/prediction_aggregator.py` -- add an
`iterative_multi_step_predict()` function that chains single-step model
predictions forward.

**Priority:** Medium -- the current sqrt-scaling approach is a reasonable
approximation, but the PDF is explicit about iterative chaining.

---

## 10. Gemini Report Generation -- Prompt Template Gap

The PDF (Core Idea Section G, Coding Instructions Section 12) provides an
exact 13-section prompt template for Gemini. The current `report_generator.py`
generates the report via local markdown assembly (no Gemini API call).

**Status:** The report generator has `_build_*` functions that produce
markdown sections locally. There is **no Gemini API call** for report
generation. Charts are generated via matplotlib. PDF conversion via pandoc is
implemented.

**What's missing:**
1. A `generate_with_gemini()` function that sends the complete company
   profile JSON to Gemini with the exact prompt template from Section G.2.
2. Post-processing: validate that all 13 sections are present in Gemini's
   response.
3. Chart insertion into Gemini's markdown output.

**Where:** `operator1/report/report_generator.py` -- add a Gemini-based
generation path alongside the existing local generation.

**Priority:** Medium -- the local report generator is a functional fallback.
Adding Gemini would produce the "Bloomberg-style" prose quality the PDF
envisions.

---

## 11. Technical Alpha Protection (OHLC Masking)

The PDF (Core Idea Section H) specifies that next-day predictions must mask
Open/High/Close and only show Low. Week, month, and year predictions show
full OHLC.

**Status:** The `prediction_aggregator.py` has a `TechnicalAlphaMask` class
and `apply_technical_alpha_mask()` function. However, the profile builder
does not appear to apply it when building the predictions section.

**Where:** Verify that `_build_predictions_section()` in
`profile_builder.py` invokes `apply_technical_alpha_mask()` before emitting
next-day OHLC predictions.

**Priority:** High -- this is an ethical/IP protection feature that the PDF
treats as non-negotiable.

---

## 12. Request Log Persistence

The PDF (Core Idea Section D.4) requires storing request metadata in
`cache/request_log.jsonl`.

**Status:** `http_utils.py` maintains an in-memory `_request_log` list and
exposes `get_request_log()`, but there is no evidence the log is ever flushed
to `cache/request_log.jsonl`.

**Where:** Add a `flush_request_log()` function in `http_utils.py` that
writes to `cache/request_log.jsonl`. Call it at the end of each notebook
phase.

**Priority:** Low -- diagnostic/debugging feature.

---

## 13. Dependencies Not Activated

The `requirements.txt` has many dependencies commented out:

```
# arch>=6.0          # GARCH (already used in forecasting.py)
# xgboost>=1.7       # tree ensemble (already used)
# torch>=2.0         # LSTM (already used)
# pymc>=5.0          # BCP (already used in regime_detector.py)
# SALib>=1.4          # Sobol (not yet implemented)
# TA-Lib>=0.4         # Candlestick patterns (not yet implemented)
# deap>=1.4           # Genetic algorithm (not yet implemented)
# matplotlib>=3.6     # Charts (already used in report_generator.py)
# plotly>=5.13        # Charts (not used)
# pytest>=7.0         # Testing
```

**Action:** Uncomment `arch`, `xgboost`, `torch`, `pymc`, `matplotlib`,
`pytest` since the code already uses them. Add `pywt` (wavelets) and `scipy`
(already a transitive dep) when implementing missing modules.

---

## 14. Summary: Priority-Ordered Integration Roadmap

### Phase A -- Fix Spec Conformance (no new features, just corrections)

| # | Task | Files to Change | Effort |
|---|------|-----------------|--------|
| A1 | Fix tier hierarchy weights to match PDF exactly | `config/survival_hierarchy.yml` | Small |
| A2 | Fix tier variable mappings to match PDF exactly | `config/survival_hierarchy.yml` | Small |
| A3 | Fix vanity adjustment (currently adjusts Tier 3 instead of Tier 1) | `config/survival_hierarchy.yml` | Small |
| A4 | Uncomment active dependencies in `requirements.txt` | `requirements.txt` | Small |

### Phase B -- Implement Missing Core Features

| # | Task | Files to Create/Change | Effort |
|---|------|------------------------|--------|
| B1 | Implement 4 ethical filters (Purchasing Power, Solvency, Gharar, Cash is King) | New: `operator1/analysis/ethical_filters.py` | Medium |
| B2 | Wire ethical filters into profile builder | `operator1/report/profile_builder.py` | Small |
| B3 | Wire ethical filters into report generator (Section 10 of report) | `operator1/report/report_generator.py` | Small |
| B4 | Create `config/canonical_fields.yml` (variable registry) | New: `config/canonical_fields.yml` | Medium |
| B5 | Create `config/country_survival_rules.yml` (extract from hardcoded logic) | New: `config/country_survival_rules.yml` | Small |
| B6 | Add macro stale flags (`is_stale_<var>`, `macro_asof_date_<var>`) | `operator1/features/macro_alignment.py` | Small |
| B7 | Apply Technical Alpha mask in profile builder | `operator1/report/profile_builder.py` | Small |
| B8 | Persist request log to `cache/request_log.jsonl` | `operator1/http_utils.py` | Small |

### Phase C -- Implement Missing Math Modules

| # | Task | Files to Create/Change | Effort |
|---|------|------------------------|--------|
| C1 | Candlestick Pattern Detector (rule-based + optional TA-Lib) | New: `operator1/models/pattern_detector.py` | Medium |
| C2 | Transformer Architecture (attention-based forecasting) | `operator1/models/forecasting.py` | High |
| C3 | Particle Filter (Sequential Monte Carlo) | `operator1/models/forecasting.py` | Medium |
| C4 | Genetic Algorithm for ensemble weight optimization | `operator1/models/prediction_aggregator.py` | Medium |
| C5 | Transfer Entropy / Information Flow | `operator1/models/causality.py` | Medium |
| C6 | Sobol Global Sensitivity Analysis | New: `operator1/models/sensitivity.py` | Medium |
| C7 | Copula Models for tail dependency | New: `operator1/models/copula.py` | High |
| C8 | Wave/Cycle Decomposition (Fourier + Wavelet) | New: `operator1/models/cycle_decomposition.py` | Medium |

### Phase D -- Enhance Temporal Engine

| # | Task | Files to Create/Change | Effort |
|---|------|------------------------|--------|
| D1 | Implement day-by-day forward pass with predict-compare-update loop | `operator1/models/forecasting.py` | High |
| D2 | Add regime-weighted historical windows for training | `operator1/models/forecasting.py` | Medium |
| D3 | Add online/incremental model parameter updates | `operator1/models/forecasting.py` | High |
| D4 | Add convergence detection for burn-out | `operator1/models/forecasting.py` | Small |
| D5 | Implement iterative multi-step prediction (chain day predictions) | `operator1/models/prediction_aggregator.py` | Medium |

### Phase E -- Gemini Integration & Polish

| # | Task | Files to Create/Change | Effort |
|---|------|------------------------|--------|
| E1 | Add Gemini-based report generation path (13-section prompt) | `operator1/report/report_generator.py` | Medium |
| E2 | Add Gemini response validation (check all sections present) | `operator1/report/report_generator.py` | Small |
| E3 | Add World Bank WDS Documents API integration | `operator1/clients/world_bank.py` | Small |
| E4 | Refine notebook cell arrangement per Section 13 of coding instructions | `notebooks/operator1.ipynb` | Medium |

---

## 15. Architecture Diagram: Where Each Gap Lives

```
User Input (ISIN + FMP symbol)
       |
       v
  [Step 0: Verify Identifiers]  .................. OK
       |
       v
  [Step A: Resolve Company (Eulerpool)]  ......... OK
       |
       v
  [Step B: Relationship Graph (Gemini)]  ......... OK
       |
       v
  [Step C: Download Raw Data]  ................... OK
  [Step C.1: FMP OHLCV]  ........................ OK
       |
       v
  [Step D: Build 2-Year Cache]  .................. OK
       |
       v
  [Step E: Derived Features]  .................... OK
  [Macro Alignment]  ............................. OK (missing: stale flags) --> B6
       |
       v
  [Step F: Survival Mode Analysis]  .............. OK (weights wrong) --> A1-A3
  [Vanity Analysis]  ............................. OK
  [Ethical Filters]  ............................. MISSING --> B1-B3
       |
       v
  [Estimation / Sudoku Inference]  ............... OK
       |
       v
  [Regime Detection (HMM/GMM/PELT/BCP)]  ........ OK
  [Causality (Granger)]  ......................... OK (missing: Transfer Entropy) --> C5
       |
       v
  [Forecasting Models]
    Kalman ....................................... OK
    GARCH ........................................ OK
    VAR .......................................... OK
    LSTM ......................................... OK
    Tree Ensemble ................................ OK
    Transformer .................................. MISSING --> C2
    Particle Filter .............................. MISSING --> C3
       |
       v
  [Ensemble + Optimization]
    Weighted average .............................. OK
    Genetic Algorithm ............................. MISSING --> C4
    Sobol Sensitivity ............................. MISSING --> C6
       |
       v
  [Forward Pass + Burn-Out]
    Basic burn-out refit .......................... OK
    Day-by-day predict-compare-update ............ MISSING --> D1
    Regime-weighted windows ...................... MISSING --> D2
    Online/incremental learning .................. MISSING --> D3
    Convergence detection ........................ MISSING --> D4
       |
       v
  [Monte Carlo]
    Regime-aware MC .............................. OK
    Importance Sampling MC ....................... OK
    Copula tail dependencies ..................... MISSING --> C7
       |
       v
  [Pattern Detection]
    Candlestick patterns ......................... MISSING --> C1
    Fourier/Wavelet cycles ....................... MISSING --> C8
       |
       v
  [Multi-Horizon Prediction]
    sqrt-scaling approximation ................... OK
    Iterative chaining ........................... MISSING --> D5
    Technical Alpha mask ......................... PARTIAL --> B7
       |
       v
  [Profile Builder]
    Identity, survival, vanity, linked, regime ... OK
    Ethical filters section ....................... MISSING --> B2
       |
       v
  [Report Generator]
    Local markdown assembly ...................... OK
    Chart generation (matplotlib) ................ OK
    PDF conversion (pandoc) ...................... OK
    Gemini API generation ........................ MISSING --> E1
    WDS country documents ........................ MISSING --> E3
```

---

## 16. Estimated Total Effort

| Phase | Task Count | Estimated Effort |
|-------|-----------|-----------------|
| A (Spec Conformance) | 4 tasks | 1-2 hours |
| B (Missing Core Features) | 8 tasks | 1-2 days |
| C (Missing Math Modules) | 8 tasks | 3-5 days |
| D (Temporal Engine) | 5 tasks | 3-5 days |
| E (Gemini & Polish) | 4 tasks | 1-2 days |
| **Total** | **29 tasks** | **~2 weeks** |

Phase A and B should be tackled first since they fix spec conformance and
add the ethical filters that define the product's identity. Phases C and D
can be done in parallel. Phase E is polish.
