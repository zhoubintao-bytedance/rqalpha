# -*- coding: utf-8 -*-
"""
TX1 Robustness & Stability Module

Provides cross-fold stability scoring, overfit detection, and
regime-conditioned evaluation. Reuses stability scoring methodology
from skyeye/evaluation/rolling_score/engine.py with adaptations
for walk-forward fold analysis.

Three core functions:
  1. compute_stability_score - CV + worst-fold + consecutive-bad-folds
  2. detect_overfit_flags    - validation-vs-test decay detection
  3. compute_regime_scores   - per-regime metric breakdown
"""

import numpy as np


def compute_stability_score(fold_metrics_list, metric_key="rank_ic_mean"):
    """Compute cross-fold stability score for a given metric.

    Three-dimensional score (mirrors rolling_score/engine.py stability):
      - CV (coefficient of variation): 50% weight
      - Worst fold value: 30% weight
      - Max consecutive below-median folds: 20% weight

    Args:
        fold_metrics_list: List of dicts, each containing prediction/portfolio metrics
        metric_key: Which metric to evaluate stability on

    Returns:
        Dict with stability_score (0-100), cv, worst_value, max_consecutive_low,
        and per-component scores
    """
    if not fold_metrics_list:
        return _empty_stability()

    values = _extract_metric_values(fold_metrics_list, metric_key)
    if len(values) == 0:
        return _empty_stability()

    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    worst_val = float(np.min(arr))
    median_val = float(np.median(arr))

    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float("inf")
    cv_score = _score_cv(cv)

    worst_score = _score_worst(worst_val, median_val)

    consecutive_low = _max_consecutive_below(arr, median_val)
    consec_score = _score_consecutive(consecutive_low, len(arr))

    stability = 0.5 * cv_score + 0.3 * worst_score + 0.2 * consec_score

    return {
        "stability_score": float(np.clip(stability, 0, 100)),
        "cv": cv,
        "worst_value": worst_val,
        "max_consecutive_low": consecutive_low,
        "cv_score": cv_score,
        "worst_score": worst_score,
        "consecutive_score": consec_score,
        "n_folds": len(arr),
        "mean": mean_val,
        "std": std_val,
        "metric_key": metric_key,
    }


def detect_overfit_flags(fold_results):
    """Detect potential overfitting by comparing validation vs test metrics.

    Flags:
      - val_test_ic_decay: Mean Rank IC drops significantly from val to test
      - val_test_spread_decay: Top bucket spread decays val -> test
      - val_dominant: Validation metrics consistently better than test

    Args:
        fold_results: List of fold result dicts with validation_metrics and prediction_metrics

    Returns:
        Dict with boolean flags and numeric decay ratios
    """
    if not fold_results:
        return _empty_overfit()

    ic_decays = []
    spread_decays = []
    val_wins = 0

    for fold in fold_results:
        val_m = fold.get("validation_metrics", {})
        test_m = fold.get("prediction_metrics", {})
        if not val_m or not test_m:
            continue

        val_ic = val_m.get("rank_ic_mean", 0.0)
        test_ic = test_m.get("rank_ic_mean", 0.0)
        ic_decays.append(val_ic - test_ic)

        val_spread = val_m.get("top_bucket_spread_mean", 0.0)
        test_spread = test_m.get("top_bucket_spread_mean", 0.0)
        spread_decays.append(val_spread - test_spread)

        val_score = val_ic + val_spread
        test_score = test_ic + test_spread
        if val_score > test_score:
            val_wins += 1

    n = len(ic_decays)
    if n == 0:
        return _empty_overfit()

    mean_ic_decay = float(np.mean(ic_decays))
    mean_spread_decay = float(np.mean(spread_decays))
    val_dominant_ratio = val_wins / n

    return {
        "val_test_ic_decay": mean_ic_decay,
        "val_test_spread_decay": mean_spread_decay,
        "val_dominant_ratio": val_dominant_ratio,
        "flag_ic_decay": mean_ic_decay > 0.02,
        "flag_spread_decay": mean_spread_decay > 0.005,
        "flag_val_dominant": val_dominant_ratio > 0.8,
        "n_folds_compared": n,
    }


def compute_regime_scores(fold_results, metric_key="rank_ic_mean"):
    """Score performance across different market regimes within each fold.

    Regimes are determined by portfolio return sign:
      - up: portfolio_return > 0 periods
      - down: portfolio_return <= 0 periods

    This is a simplified regime analysis; full regime detection
    (bull/bear/sideways) requires market-level data not available
    in the research proxy output.

    Args:
        fold_results: List of fold result dicts with portfolio_returns_df
        metric_key: Metric to report per-regime

    Returns:
        Dict with per-regime statistics
    """
    up_returns = []
    down_returns = []

    for fold in fold_results:
        port_df = fold.get("portfolio_returns_df")
        if port_df is None or len(port_df) == 0:
            continue
        returns = port_df["portfolio_return"].values
        up_mask = returns > 0
        down_mask = ~up_mask
        if up_mask.any():
            up_returns.extend(returns[up_mask].tolist())
        if down_mask.any():
            down_returns.extend(returns[down_mask].tolist())

    up_arr = np.array(up_returns) if up_returns else np.array([])
    down_arr = np.array(down_returns) if down_returns else np.array([])

    # Also compute fold-level metric consistency
    fold_values = _extract_metric_values(fold_results, metric_key)
    positive_folds = sum(1 for v in fold_values if v > 0)
    n_folds = len(fold_values)

    return {
        "up_regime": {
            "n_periods": len(up_arr),
            "mean_return": float(np.mean(up_arr)) if len(up_arr) > 0 else 0.0,
            "total_return": float(np.sum(up_arr)) if len(up_arr) > 0 else 0.0,
        },
        "down_regime": {
            "n_periods": len(down_arr),
            "mean_return": float(np.mean(down_arr)) if len(down_arr) > 0 else 0.0,
            "total_return": float(np.sum(down_arr)) if len(down_arr) > 0 else 0.0,
        },
        "metric_consistency": {
            "metric_key": metric_key,
            "positive_folds": positive_folds,
            "total_folds": n_folds,
            "positive_ratio": positive_folds / n_folds if n_folds > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_metric_values(fold_results, metric_key):
    """Extract a metric from fold results, searching prediction_metrics then portfolio_metrics."""
    values = []
    for fold in fold_results:
        pred_m = fold.get("prediction_metrics", {})
        port_m = fold.get("portfolio_metrics", {})
        if metric_key in pred_m:
            values.append(float(pred_m[metric_key]))
        elif metric_key in port_m:
            values.append(float(port_m[metric_key]))
    return values


def _score_cv(cv):
    """Score coefficient of variation (lower CV = more stable)."""
    if cv >= 0.5:
        return 0.0
    if cv <= 0.1:
        return 100.0
    return (0.5 - cv) / 0.4 * 100.0


def _score_worst(worst_val, median_val):
    """Score worst fold relative to median."""
    if median_val == 0:
        return 50.0
    ratio = worst_val / median_val if median_val != 0 else 0.0
    # ratio close to 1.0 means worst ≈ median → good
    # ratio <= 0 means worst is opposite sign → bad
    if ratio >= 0.8:
        return 100.0
    if ratio <= 0.0:
        return 0.0
    return ratio / 0.8 * 100.0


def _max_consecutive_below(arr, threshold):
    """Count max consecutive values below threshold."""
    max_run = 0
    current_run = 0
    for v in arr:
        if v < threshold:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _score_consecutive(max_consec, n_folds):
    """Score based on max consecutive below-median folds."""
    if n_folds <= 1:
        return 100.0
    ratio = max_consec / n_folds
    if ratio >= 0.5:
        return 0.0
    if ratio <= 0.1:
        return 100.0
    return (0.5 - ratio) / 0.4 * 100.0


def _empty_stability():
    return {
        "stability_score": 0.0,
        "cv": float("inf"),
        "worst_value": 0.0,
        "max_consecutive_low": 0,
        "cv_score": 0.0,
        "worst_score": 0.0,
        "consecutive_score": 0.0,
        "n_folds": 0,
        "mean": 0.0,
        "std": 0.0,
        "metric_key": "",
    }


def _empty_overfit():
    return {
        "val_test_ic_decay": 0.0,
        "val_test_spread_decay": 0.0,
        "val_dominant_ratio": 0.0,
        "flag_ic_decay": False,
        "flag_spread_decay": False,
        "flag_val_dominant": False,
        "n_folds_compared": 0,
    }
