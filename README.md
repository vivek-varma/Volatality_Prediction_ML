# Volatality_Prediction_ML
Regime Prediction for ES Futures (ML + DL)

A practical, end-to-end project to predict volatility regimes (LOW / MID / HIGH) for E-mini S&P 500 (ES) using multiple modeling approaches. The goal is to produce a robust, explainable regime signal that can gate downstream strategies (carry, trend, mean-reversion, options overlays).

📌 Highlights

Target: next-day realized variance RV_t+1, mapped to regimes (LOW/MID/HIGH).

Data:

ES 1-minute OHLCV (Databento GLBX.MDP3, ES.FUT) → daily aggregates & realized vol.

Vol surface proxies: VIX, VIX3M, VIX6M, VVIX (official public CSVs).

Derived term-structure features (slopes, butterflies), realized-moment features (skew/kurt), calendar & session splits.

Leakage-safe pipeline: train/val/test split first → forward-fill only within split → train-only scaling.

Models: 5 baselines trained & compared. Best performer: LightGBM with train-fixed labels.

🧱 Feature Set (daily)

Realized stats (from ES 1m): RV (σ²), rolling RV means/vols, intraday vs. overnight splits, intraday r-skew/r-kurt.

Vol proxies: VIX, VIX3M, VIX6M, VVIX; term structure slopes (VIX6M−VIX, VIX3M−VIX), curvature.

Price/Trend: ES daily returns, rolling momentum, ATR-style ranges.

Calendar: weekday, month, pre/post holidays (optional).

Sanity: no future info; no backfill; all transforms fitted on train only.

🎯 Labels

Primary: train-only quantile cuts on RV_t+1.

Example: LOW = bottom 25%, MID = 25–75%, HIGH = top 25% (tunable).

Why: fixed cuts prevent label drift through time and improved HIGH recall.

🧪 Models

XGBoost (multiclass)

LightGBM (multiclass) ← best

TCN (Temporal Conv Net) — sequence model over 30–60 day windows

ElasticNet (regression on RV_t+1) → mapped to regimes by fixed cuts

LightGBM (re-labeled) — same as #2 but with train-fixed thresholds and optional HIGH recall booster (if P(HIGH) ≥ τ, force class=HIGH)

Notes:

Class imbalance handled via class weights.

Early stopping on validation; test held out (20%).

TCN trained with AMP, cosine LR, gradient clipping.

🏆 Results (summary)

Best single model: LightGBM (re-labeled) with train-fixed cuts and careful preprocessing (no backfill).

Material lift in HIGH recall vs. original labels; balanced accuracy stable.

Deep model (TCN) underperforms with limited daily data (expected); shines with richer intraday inputs.

BEST MODEL STATS:
y_val_pred,  y_val_proba  = predict_with_high_boost(lgbm, X_val,  use_boost=True)
y_test_pred, y_test_proba = predict_with_high_boost(lgbm, X_test, use_boost=True)

report_split("VALIDATION (boosted)", y_val, y_val_pred, y_val_proba)
report_split("TEST (boosted)",       y_test, y_test_pred, y_test_proba)

# Also show pure argmax (no boost) for comparison
y_val_pred_arg,  y_val_proba_arg  = predict_with_high_boost(lgbm, X_val,  use_boost=False)
y_test_pred_arg, y_test_proba_arg = predict_with_high_boost(lgbm, X_test, use_boost=False)
report_split("VALIDATION (argmax)", y_val, y_val_pred_arg, y_val_proba_arg)
report_split("TEST (argmax)",       y_test, y_test_pred_arg, y_test_proba_arg)
