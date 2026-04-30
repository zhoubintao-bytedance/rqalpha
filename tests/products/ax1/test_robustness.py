# -*- coding: utf-8 -*-

import pytest

from skyeye.products.ax1.robustness import (
    aggregate_fold_metrics,
    bootstrap_metric_ci,
    build_robustness_summary,
    compute_positive_ratio,
    compute_sample_decay,
    compute_stability_score,
    detect_overfit_flags,
)


def _fold(pred_ic: float, pred_spread: float, val_ic: float | None = None, val_spread: float | None = None):
    return {
        "prediction_metrics": {
            "rank_ic_mean": pred_ic,
            "top_bucket_spread_mean": pred_spread,
            "top_k_hit_rate": 0.60,
        },
        "validation_metrics": {
            "rank_ic_mean": pred_ic if val_ic is None else val_ic,
            "top_bucket_spread_mean": pred_spread if val_spread is None else val_spread,
            "top_k_hit_rate": 0.55,
        },
    }


def test_aggregate_fold_metrics_outputs_prediction_and_validation_means():
    result = aggregate_fold_metrics(
        [
            _fold(0.02, 0.010, val_ic=0.04, val_spread=0.020),
            _fold(0.06, 0.030, val_ic=0.08, val_spread=0.040),
        ]
    )

    assert result["n_folds"] == 2
    assert result["prediction"]["rank_ic_mean_mean"] == pytest.approx(0.04)
    assert result["prediction"]["top_bucket_spread_mean_mean"] == pytest.approx(0.02)
    assert result["validation"]["rank_ic_mean_mean"] == pytest.approx(0.06)
    assert result["validation"]["top_k_hit_rate_mean"] == pytest.approx(0.55)


def test_stability_and_positive_ratio_use_prediction_metrics():
    folds = [_fold(0.04, 0.01), _fold(-0.02, 0.02), _fold(0.06, 0.03)]

    stability = compute_stability_score(folds)
    positive_ratio = compute_positive_ratio(folds)

    assert stability["metric_key"] == "top_bucket_spread_mean"
    assert stability["n_folds"] == 3
    assert stability["worst_value"] == pytest.approx(0.01)
    assert 0.0 <= stability["stability_score"] <= 100.0
    assert positive_ratio["positive_ratio"] == pytest.approx(1.0)
    assert positive_ratio["n_folds"] == 3


def test_detect_overfit_flags_compares_validation_to_prediction_metrics():
    folds = [
        _fold(0.01, 0.002, val_ic=0.08, val_spread=0.020),
        _fold(0.02, 0.003, val_ic=0.09, val_spread=0.021),
        _fold(0.01, 0.002, val_ic=0.10, val_spread=0.022),
    ]

    result = detect_overfit_flags(folds)

    # mean_oos_ic ≈ 0.0133, threshold = max(0.01, 0.3*0.0133) ≈ 0.01
    # ic_decay ≈ 0.077 > 0.01 → flag True
    assert result["val_test_ic_decay"] > 0.01
    # mean_oos_spread ≈ 0.0023, threshold = max(0.002, 0.25*0.0023) ≈ 0.002
    # spread_decay ≈ 0.018 > 0.002 → flag True
    assert result["val_test_spread_decay"] > 0.002
    assert result["val_dominant_ratio"] == pytest.approx(1.0)
    assert result["flag_ic_decay"] is True
    assert result["flag_spread_decay"] is True
    assert result["flag_val_dominant"] is True


def test_bootstrap_metric_ci_is_deterministic_and_flags_zero_crossing():
    folds = [_fold(0.03, value) for value in [0.010, 0.012, -0.040, 0.015, 0.008, 0.011]]

    first = bootstrap_metric_ci(folds, n_bootstrap=200, seed=7)
    second = bootstrap_metric_ci(folds, n_bootstrap=200, seed=7)

    assert first == second
    assert first["metric_key"] == "top_bucket_spread_mean"
    assert first["n_observations"] == 6
    assert first["confidence"] == pytest.approx(0.80)
    assert first["test_side"] == "one_sided_positive"
    assert first["alpha"] == pytest.approx(0.20)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert first["ci_crosses_zero"] is True


def test_robustness_summary_includes_bootstrap_ci_and_sample_decay():
    folds = [
        _fold(0.04, 0.030),
        _fold(0.03, 0.025),
        _fold(0.035, 0.028),
        _fold(0.032, 0.026),
        _fold(0.02, 0.006),
        _fold(0.01, 0.004),
        _fold(0.012, 0.005),
        _fold(0.011, 0.004),
    ]

    decay = compute_sample_decay(folds)
    summary = build_robustness_summary(
        folds,
        parameter_validation_summary={"warning_count": 1, "warnings": ["high_candidate_dispersion"]},
        seed=11,
        n_bootstrap=200,
    )

    assert decay["late_minus_early"] < 0.0
    assert summary["bootstrap_ci"]["n_bootstrap"] == 200
    assert summary["bootstrap_ci"]["confidence"] == pytest.approx(0.80)
    assert summary["bootstrap_ci"]["test_side"] == "one_sided_positive"
    assert summary["sample_decay"]["flag_late_decay"] is True
    assert summary["parameter_sensitivity"]["warning_count"] == 1
    assert summary["warning_count"] >= 1
