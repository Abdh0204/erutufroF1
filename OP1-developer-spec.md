## Operator 1 Kaggle Build Spec

Developer-facing instructions to implement Operator 1 as a Kaggle-ready Python project that builds 2-year daily caches for a target company and its linked entities, integrates macro data, and prepares survival-mode analysis plus downstream temporal modeling.

### 1) Kaggle execution requirements
- Notebook setting: Internet **ON**.
- Secrets: store in Kaggle `EULERPOOL_API_KEY`, `FMP_API_KEY`, `GEMINI_API_KEY`; optional `WORLD_BANK_API_KEY` if a gateway is used.
- Secrets access: read via Kaggle secrets client; fail fast with a clear message if any key is missing; never hardcode keys.
- Packages: `requests`, `pandas`, `numpy`, `datetime` (stdlib), optional `pyarrow` for Parquet.

### 2) Inputs (Step 0)
- Required user inputs: `target_isin` (Eulerpool ISIN), `fmp_symbol` (FMP symbol for OHLCV). Do **not** ask for company name or country.
- Extract `country` from Eulerpool profile after verifying ISIN.

### 3) Identifier verification (Step 0.1)
- Eulerpool: `GET /api/1/equity/profile/{target_isin}` — fail fast on error (invalid ISIN/API key).
- FMP: `GET /quote?symbol={fmp_symbol}&apikey=...` — fail fast on error (invalid symbol/API key). If missing, downstream modules requiring OHLCV must halt with a clear message.

### 4) World Bank indicator mapping (Step 0.2)
- Use `config/world_bank_indicator_map.yml` mapping `canonical_variable -> indicator_code`; avoid hardcoded country-specific codes in code.
- Optionally use Gemini to propose mappings; treat as suggestions to update config (human-reviewed), not runtime dependency.
- Convert Eulerpool country ISO-2 → World Bank ISO-3 via World Bank countries endpoint (cache result).

### 5) Relationship discovery (Step A/B)
- Target resolution: search Eulerpool, present top matches (name, ticker, ISIN, exchange, country); user confirms. Store `target_isin`, `target_ticker`, `target_country`, `target_sector`, `target_industry`.
- Linked entities discovery: use Gemini to propose competitors, suppliers, customers, financial institutions, logistics, regulators as search terms with country/sector hints. Auto-resolve to Eulerpool via search + scoring (ticker exact > name similarity > country > sector). Accept match only if score ≥ 70; else drop. If no competitors found, fallback to sector peers sorted by market cap (top 5, excluding target).
- Keep per-group/global search budgets; checkpoint progress (`cache/progress.json`) if batching.

### 6) Data extraction (Step C/C.1)
- Eulerpool per entity (target + linked):
  - Profile: `GET /api/1/equity/profile/{isin}`
  - Quotes: `GET /api/1/equity/quotes/{identifier}` (daily series)
  - Income statement: `GET /api/1/equity/incomestatement/{isin}`
  - Balance sheet: `GET /api/1/equity/balancesheet/{isin}`
  - Cash flow: `GET /api/1/equity/cashflowstatement/{isin}`
- Eulerpool target-only extras:
  - Peers: `GET /api/1/equity/peers/{isin}`
  - Supply chain: `GET /api/1/equity/supply-chain/{isin}`
  - Executives: `GET /api/1/equity/executives/{isin}`
- FMP (OHLCV authoritative):
  - Daily OHLCV 2y: `GET /historical-price-eod/full?symbol={SYMBOL}&apikey=...`
  - Optional: dividend-adjusted, non-split-adjusted variants.
  - Quote verification: `GET /quote?symbol={SYMBOL}&apikey=...`
  - Intraday (optional for UI zoom): multiple `historical-chart/{interval}` endpoints.
- API key rule: if URL already has `?`, append `&apikey=...`; otherwise `?apikey=...`.

### 7) Cache building (Step D)
- Build 2-year daily cache for target and each linked entity.
- Statements are periodic; align to daily using **as-of** logic: for each day `t`, attach latest statement with `report_date <= t`. Reject any `report_date > t` (look-ahead failure).
- Persist artifacts (Kaggle output):
  - `cache/target_company_daily.parquet`
  - `cache/linked_entities_daily.parquet`
  - `cache/linked_aggregates_daily.parquet`
  - `cache/full_feature_table.parquet`
  - `cache/metadata.json` (inputs, identifiers, relationships, versions)

### 8) Direct fields to store
- Profile: `isin`, `ticker`, `exchange`, `currency`, `country`, `sector`, `industry`, `sub_industry` (if available).
- Quotes (daily): `open`, `high`, `low`, `close`, `volume`, optional `adjusted_close`, `vwap`, `market_cap`, `shares_outstanding`.
- Statements (periodic, decision variables only): revenue, gross profit, EBIT, EBITDA, net income, interest expense, taxes, total assets, total liabilities, total equity, current assets, current liabilities, cash & equivalents, short-term debt, long-term debt, receivables, operating cash flow, capex, investing CF, financing CF, dividends paid.

### 9) Derived decision variables (daily per entity)
- Returns/risk: `return_1d`, `log_return_1d`, `volatility_21d`, `drawdown_252d`.
- Solvency/leverage: `total_debt_asof`, `debt_to_equity_signed`, `debt_to_equity_abs`, `net_debt`, `net_debt_to_ebitda`.
- Liquidity/survival: `current_ratio`, `quick_ratio` (if components), `cash_ratio`.
- Cash reality: `free_cash_flow`, `free_cash_flow_ttm_asof`, `fcf_yield`.
- Profitability: `gross_margin`, `operating_margin`, `net_margin`, `roe`.
- Valuation (optional): `pe_ratio_calc`, `earnings_yield_calc`, `ps_ratio_calc`, `enterprise_value`, `ev_to_ebitda`.

### 10) Derived linked variables (aggregates)
- For each day, compute aggregates for competitors, supply chain (suppliers + customers), financial institutions; also sector and industry peers.
- Examples: `competitors_avg_return_21d`, `competitors_median_vol_21d`, `supply_chain_avg_drawdown_252d`, `rel_strength_vs_sector`, `valuation_premium_vs_industry`.

### 11) Macro integration (World Bank)
- Canonical variables: `inflation_rate_yoy`, `cpi_index`, `unemployment_rate`, `gdp_growth`, `gdp_current_usd`, `official_exchange_rate_lcu_per_usd`, `current_account_balance_pct_gdp`, `reserves_months_of_imports`; optional proxies: `real_interest_rate`, `lending_interest_rate`, `deposit_interest_rate`.
- Fetch with retries + backoff (respect `Retry-After`), `timeout_s`, and on-disk caching to avoid repeated calls.
- Align infrequent series to daily via as-of; add `is_missing_<var>` flags.
- Compute `inflation_rate_daily_equivalent` and `real_return_1d = nominal_return_1d - inflation_rate_daily_equivalent`.

### 12) Survival mode logic (Step F)
- Company survival trigger (any true): `current_ratio < 1.0` OR `debt_to_equity_abs > 3.0` OR `fcf_yield < 0` OR `drawdown_252d < -0.40`.
- Country survival (config-driven thresholds): credit spread > 5%, unemployment +3% in 6 months, yield curve slope < -0.5%, FX volatility > 20%. Use daily as-of macro data.
- Country protection flag: strategic sector (defense, energy, banking, utilities, telecom) OR market cap > 0.1% of GDP OR emergency policy-rate cut >2% in 3 months (uses `policy_rate` proxy if available). Make rules data-driven via `config/country_protection_rules.yml`.

### 13) Vanity percentage
- Components: executive compensation excess (>5% of net income), SG&A bloat (20% above industry median SG&A ratio), buybacks while FCF TTM < 0, marketing excess during survival mode (>10% revenue).
- Vanity % = total vanity spend / revenue; clip 0–100; store per day.

### 14) Survival hierarchy weights
- Config `config/survival_hierarchy.yml` defines tiers and weight presets: normal, company_survival, modified_survival (country crisis), extreme_survival (both), vanity_adjusted deltas.
- Select weights daily based on `company_survival_mode_flag`, `country_survival_mode_flag`, `country_protected_flag`, `vanity_percentage`. Store `hierarchy_tier1_weight` … `hierarchy_tier5_weight` and `survival_regime` label.

### 15) Data quality rules
- No look-ahead: fail pipeline if any statement `report_date > t` when applied to day `t`.
- Ratio safety: shared helper — if denominator null/0/|denom|<ε → return null, set `is_missing_<var>=1` and `invalid_math_<var>=1`. Handle negative equity by storing both signed and absolute D/E; use absolute for triggers.
- Missing data: keep `null` plus `is_missing_<feature>` for every derived feature.
- Unit normalization: capture units from Eulerpool if provided or normalize; persist metadata.

### 16) Estimation (post-cache Sudoku inference)
- Inputs: `full_feature_table` only; no extra API calls.
- Outputs per variable: `x_observed`, `x_estimated`, `x_final`, `x_source` ∈ {observed, estimated}, `x_confidence` ∈ [0,1].
- Two-pass linear-time: (1) deterministic identity fill; (2) regime-weighted rolling imputer (e.g., BayesianRidge per tier), trained online on observed data up to day `t`. Do not overwrite observed values; penalize estimated dimensions in loss; log coverage percentages per variable/tier.

### 17) Temporal analysis (Phase 2-3 overview)
- Load libraries: statsmodels (VAR, SARIMAX, Kalman), `arch` (GARCH), sklearn (RF/GBM), XGBoost, PyTorch (LSTM), `ruptures`, `hmmlearn`, `pymc`, `SALib`, `talib`, wavelets, FFT, GA optimizer (`deap`), etc.
- Regime detection: HMM (returns+vol), GMM (returns), structural breaks via PELT and optional Bayesian change point; label regimes (bull/bear/high_vol/low_vol) and mark breaks.
- Model robustness: wrap fits in try/except; set `model_failed_<name>` flags; fall back to simpler models (AR(1) per var if VAR fails; tree/linear if LSTM fails; baseline last-value/EMA if all fail). Ensemble reweights over available models.
- Kalman filter for tier 1–2 liquidity/solvency; GARCH for volatility; VAR for multivariate; LSTM for nonlinear; RF/GBM/XGB for tabular.
- Forward pass: day-by-day training; track errors per tier. Burn-out phase: intensive retraining on last ~6 months with early stop if no improvement ≥3 iterations.
- Monte Carlo: regime-aware simulations (n≈10k) and importance sampling for tail survival probability.
- Predictions: next day/week/month/year with uncertainty bands; apply Technical Alpha protection by masking next-day OHLC except Low.

### 18) Report generation (Phase 4 overview)
- Build `company_profile.json` from caches, survival flags, linked aggregates, temporal results, vanity, regimes, predictions, model metrics, filters.
- Gemini report generator produces Bloomberg-style markdown; include required short **LIMITATIONS** section (data window, OHLCV source caveats, macro frequency, missingness summary, failed modules and mitigations) for non-technical readers.
- Optional charts: price history with regime shading, forecast candlesticks, hierarchy weights, survival timeline, Monte Carlo distribution. Optional PDF via pandoc.

### 19) Notebook cell layout (recommended)
1) Intro/guarantees
2) Inputs (`target_isin`, `fmp_symbol`)
3) Load secrets
4) Global config (dates, timeouts, retries, budgets, `FORCE_REBUILD`)
5) HTTP utilities (retries/backoff/cache/logging)
6) Eulerpool client
7) FMP client
8) Identifier verification (fail fast) + extract country
9) World Bank macro module (load mapping, fetch, align)
10) Linked entities discovery (Gemini, optional flag)
11) Linked resolution with budgets + checkpointing
12) Target OHLCV build (FMP authoritative)
13) Target fundamentals as-of table
14) Feature engineering + safety flags
15) Linked caches in batches (optional)
16) Linked aggregates join
17) Survival mode + hierarchy weights + vanity
18) Save cache artifacts
19) Modeling/prediction cells (optional, one per model with try/except)
20) Report generation (Gemini) with LIMITATIONS section and optional PDF

### 20) Operational guardrails
- Idempotent cells; prefer “load if cached unless `FORCE_REBUILD=True`”.
- Disk cache HTTP responses to stay within Kaggle budgets; log requests for `metadata.json`.
- Respect secrets hygiene; never print keys.
- Short-circuit on critical verification failures (ISIN/FMP symbol/API key).

