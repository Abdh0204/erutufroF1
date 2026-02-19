"""Temporal modeling modules (regime detection, forecasting, Monte Carlo,
prediction aggregation, financial health scoring).

T6.1: regime_detector        -- HMM, GMM, PELT, BCP regime/break detection
T6.1: causality              -- Granger causality testing and variable pruning
T6.2: forecasting            -- Kalman, GARCH, VAR, LSTM, tree ensemble, baseline
T6.3: monte_carlo            -- regime-aware MC simulations, importance sampling,
                                survival probability estimation
T6.4: prediction_aggregator  -- ensemble weighting, multi-horizon predictions,
                                uncertainty bands, Technical Alpha masking
T5d:  financial_health       -- daily composite health scores injected into cache
                                so temporal models learn from them automatically
"""
