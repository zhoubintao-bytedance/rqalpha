# -*- coding: utf-8 -*-

import pytest


def _gate_ready_result(**overrides):
    result = {
        "training_summary": {
            "fold_results": [{"fold_id": idx} for idx in range(6)],
            "aggregate_metrics": {
                "n_folds": 6,
                "prediction": {
                    "rank_ic_mean_mean": 0.035,
                    "top_bucket_spread_mean_mean": 0.004,
                },
            },
            "stability": {"stability_score": 55.0, "cv": 0.50},
            "positive_ratio": {"positive_ratio": 0.67},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
        },
        "calibration_bundle": {
            "bucket_stats": [{"bucket_id": f"b{idx:02d}", "sample_count": 12} for idx in range(10)],
            "summary": {"oos_rows": 120},
        },
        "evaluation": {
            "portfolio": {
                "net_mean_return": 0.001,
                "max_drawdown": 0.04,
                "opportunity_benchmark_available": True,
                "excess_net_mean_return": 0.001,
                "max_excess_drawdown": 0.03,
                "alpha_hit_rate": 0.60,
                "max_rolling_underperformance": 0.04,
                "active_day_ratio": 0.80,
                "active_gross_mean": 0.70,
                "cash_sitting_ratio": 0.20,
                "mean_turnover": 0.10,
                "annual_turnover": 25.2,
                "cost_drag_annual": 0.01,
                "manual_operation_burden": {"total_orders": 12, "max_orders_single_date": 4},
                "allocation": {
                    "allocation_drift_mean": 0.01,
                    "cash_buffer_deviation_mean": 0.01,
                },
            }
        },
    }
    result.update(overrides)
    return result


def test_evaluate_promotion_gate_passes_canary_with_default_thresholds():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    summary = evaluate_promotion_gate(_gate_ready_result())

    assert summary["gate_level"] == "canary_live"
    assert summary["passed"] is True
    assert summary["metrics"]["n_folds"] == 6
    assert summary["metrics"]["rank_ic_mean"] == pytest.approx(0.035)
    assert summary["checks"]["min_rank_ic_mean"]["hard_gate"] is False
    assert summary["checks"]["min_excess_net_mean_return"]["passed"] is True
    assert "min_rank_ic_mean" not in summary["failed_checks"]
    assert summary["tradability_gate"]["passed"] is True
    assert summary["research_support_gate"]["passed"] is True
    assert "min_excess_net_mean_return" in summary["tradability_gate"]["checks"]
    assert "min_rank_ic_mean" in summary["research_support_gate"]["checks"]


def test_evaluate_promotion_gate_splits_tradability_failures_from_research_support():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        evaluation={
            "portfolio": {
                **_gate_ready_result()["evaluation"]["portfolio"],
                "excess_net_mean_return": -0.001,
            }
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is False
    assert summary["tradability_gate"]["passed"] is False
    assert summary["research_support_gate"]["passed"] is True
    assert summary["tradability_gate"]["failed_checks"] == ["min_excess_net_mean_return"]
    assert summary["failed_checks"] == ["min_excess_net_mean_return"]


def test_evaluate_promotion_gate_keeps_research_soft_failures_out_of_tradability_gate():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        training_summary={
            **_gate_ready_result()["training_summary"],
            "aggregate_metrics": {
                "n_folds": 6,
                "prediction": {
                    "rank_ic_mean_mean": -0.01,
                    "top_bucket_spread_mean_mean": -0.001,
                },
            },
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["tradability_gate"]["passed"] is True
    assert summary["research_support_gate"]["passed"] is True
    assert "min_rank_ic_mean" in summary["research_support_gate"]["soft_failed_checks"]
    assert "min_top_bucket_spread_mean" in summary["research_support_gate"]["soft_failed_checks"]
    assert summary["tradability_gate"]["soft_failed_checks"] == []


def test_default_promotion_thresholds_reflect_trader_review_contract():
    from skyeye.products.ax1.promotion import DEFAULT_PROMOTION_THRESHOLDS, evaluate_promotion_gate
    from skyeye.products.ax1.robustness import bootstrap_metric_ci

    thresholds = DEFAULT_PROMOTION_THRESHOLDS
    assert thresholds["canary_live"]["min_oos_rows"] == 100
    assert thresholds["default_live"]["min_oos_rows"] == 300
    assert thresholds["canary_live"]["max_cv"] == pytest.approx(0.50)
    assert thresholds["default_live"]["max_cv"] == pytest.approx(0.40)
    bootstrap = bootstrap_metric_ci(
        [{"prediction_metrics": {"top_bucket_spread_mean": value}} for value in [0.010, 0.012, -0.020, 0.015]],
        n_bootstrap=100,
        seed=7,
    )
    assert bootstrap["confidence"] == pytest.approx(0.80)
    assert bootstrap["test_side"] == "one_sided_positive"
    assert bootstrap["alpha"] == pytest.approx(0.20)

    low_sample = _gate_ready_result(calibration_bundle={"summary": {"oos_rows": 99}})
    low_sample_summary = evaluate_promotion_gate(low_sample)
    assert low_sample_summary["passed"] is False
    assert "min_oos_rows" in low_sample_summary["failed_checks"]

    high_cv = _gate_ready_result(
        training_summary={
            **_gate_ready_result()["training_summary"],
            "stability": {"stability_score": 55.0, "cv": 0.51},
        }
    )
    high_cv_summary = evaluate_promotion_gate(high_cv)
    assert high_cv_summary["passed"] is False
    assert "max_cv" in high_cv_summary["failed_checks"]

    default_live_ready = _gate_ready_result(
        calibration_bundle={"summary": {"oos_rows": 300}},
        training_summary={
            **_gate_ready_result()["training_summary"],
            "stability": {"stability_score": 55.0, "cv": 0.40},
        },
    )
    default_summary = evaluate_promotion_gate(default_live_ready, gate_level="default_live")
    assert default_summary["checks"]["min_oos_rows"]["threshold"] == pytest.approx(300)
    assert default_summary["checks"]["max_cv"]["threshold"] == pytest.approx(0.40)
    assert default_summary["checks"]["min_oos_rows"]["passed"] is True
    assert default_summary["checks"]["max_cv"]["passed"] is True


def test_evaluate_promotion_gate_uses_injected_thresholds_and_reports_failures():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    summary = evaluate_promotion_gate(
        _gate_ready_result(calibration_bundle={"summary": {"oos_rows": 300}}),
        gate_level="default_live",
        thresholds={
            "default_live": {
                "min_folds": 6,
                "min_rank_ic_mean": 0.05,
                "min_excess_net_mean_return": 0.002,
                "max_cv": 0.80,
                "min_stability_score": 40,
            }
        },
    )

    assert summary["passed"] is False
    assert "min_excess_net_mean_return" in summary["failed_checks"]
    assert "min_rank_ic_mean" not in summary["failed_checks"]
    assert summary["checks"]["min_rank_ic_mean"]["actual"] == pytest.approx(0.035)
    assert summary["checks"]["min_rank_ic_mean"]["threshold"] == pytest.approx(0.05)
    assert summary["checks"]["min_rank_ic_mean"]["hard_gate"] is False


def test_evaluate_promotion_gate_treats_feature_diagnostics_warnings_as_soft_check():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        training_summary={
            **_gate_ready_result()["training_summary"],
            "feature_review_summary": {
                "warning_count": 3,
                "warnings": [
                    {"feature": "factor_a", "decision": "review", "reasons": ["low_coverage"]},
                    {"feature": "factor_b", "decision": "drop_candidate", "reasons": ["negative_rank_ic"]},
                    {"feature": "factor_c", "decision": "drop_candidate", "reasons": ["low_cross_sectional_variance"]},
                ],
            },
        },
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["metrics"]["feature_diagnostics_warning_count"] == 3
    assert summary["checks"]["max_feature_diagnostics_warning_count"]["passed"] is False
    assert summary["checks"]["max_feature_diagnostics_warning_count"]["hard_gate"] is False
    assert "max_feature_diagnostics_warning_count" not in summary["failed_checks"]


def test_evaluate_promotion_gate_treats_parameter_validation_fragility_as_soft_check():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        parameter_validation_summary={
            "schema_version": 1,
            "fragile": True,
            "warning_count": 2,
            "warnings": ["current_params_not_top_candidate", "high_candidate_dispersion"],
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["metrics"]["parameter_validation_warning_count"] == 2
    assert summary["checks"]["max_parameter_validation_warning_count"]["passed"] is False
    assert summary["checks"]["max_parameter_validation_warning_count"]["hard_gate"] is False
    assert "max_parameter_validation_warning_count" not in summary["failed_checks"]


def test_evaluate_promotion_gate_does_not_beta_block_on_absolute_drawdown_when_excess_skill_passes():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        evaluation={
            "portfolio": {
                **_gate_ready_result()["evaluation"]["portfolio"],
                "max_drawdown": 0.30,
                "opportunity_benchmark_available": True,
                "excess_net_mean_return": 0.002,
                "max_excess_drawdown": 0.04,
                "alpha_hit_rate": 0.62,
                "max_rolling_underperformance": 0.04,
            }
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert "max_drawdown" not in summary["checks"]
    assert summary["checks"]["catastrophic_max_drawdown"]["passed"] is True


def test_evaluate_promotion_gate_fails_when_opportunity_benchmark_unavailable():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        evaluation={
            "portfolio": {
                **_gate_ready_result()["evaluation"]["portfolio"],
                "opportunity_benchmark_available": False,
            }
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is False
    assert "opportunity_benchmark_available" in summary["failed_checks"]


def test_evaluate_promotion_gate_treats_bootstrap_ci_crossing_zero_as_soft_check():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        training_summary={
            **_gate_ready_result()["training_summary"],
            "robustness": {
                "bootstrap_ci": {
                    "metric_key": "top_bucket_spread_mean",
                    "ci_low": -0.0001,
                    "ci_high": 0.004,
                    "confidence": 0.80,
                    "test_side": "one_sided_positive",
                    "ci_crosses_zero": True,
                },
                "sample_decay": {"flag_late_decay": False},
                "warning_count": 1,
            },
        },
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["metrics"]["bootstrap_ci_crosses_zero"] is True
    assert summary["checks"]["bootstrap_ci_crosses_zero"]["passed"] is False
    assert summary["checks"]["bootstrap_ci_crosses_zero"]["hard_gate"] is False
    assert "bootstrap_ci_crosses_zero" not in summary["failed_checks"]


def test_evaluate_promotion_gate_treats_rank_ic_fdr_significance_as_soft_check():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        evaluation={
            "signal": {
                "rank_ic_significance": {
                    "method": "newey_west_mean_t_test",
                    "fdr_method": "benjamini_hochberg",
                    "n_observations": 40,
                    "newey_west_lags": 3,
                    "mean": 0.01,
                    "t_stat": 1.5,
                    "p_value": 0.08,
                    "fdr_adjusted_p_value": 0.08,
                    "significant_at_5pct": False,
                }
            },
            "portfolio": _gate_ready_result()["evaluation"]["portfolio"],
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["metrics"]["rank_ic_fdr_significant"] is False
    assert summary["checks"]["rank_ic_fdr_significant"]["passed"] is False
    assert summary["checks"]["rank_ic_fdr_significant"]["hard_gate"] is False


def test_promote_package_require_gate_fails_with_clear_error(tmp_path):
    from skyeye.products.ax1.persistence import save_experiment
    from skyeye.products.ax1.promote_package import promote_package

    experiment_dir = save_experiment(
        {
            "experiment_name": "ax1_gate_fail",
            "gate_summary": {
                "gate_level": "canary_live",
                "passed": False,
                "failed_checks": ["min_rank_ic_mean", "min_folds"],
            },
        },
        tmp_path / "experiments" / "ax1_gate_fail",
        experiment_name="ax1_gate_fail",
    )

    with pytest.raises(ValueError, match="promotion gate failed.*failed_checks.*min_rank_ic_mean.*min_folds"):
        promote_package(
            experiment_dir=experiment_dir,
            packages_root=tmp_path / "packages",
            require_gate=True,
            gate_level="canary_live",
        )


def test_promote_package_requires_gate_by_default(tmp_path):
    from skyeye.products.ax1.persistence import save_experiment
    from skyeye.products.ax1.promote_package import promote_package

    experiment_dir = save_experiment(
        {
            "experiment_name": "ax1_gate_fail_default",
            "gate_summary": {
                "gate_level": "canary_live",
                "passed": False,
                "failed_checks": ["min_excess_net_mean_return"],
            },
        },
        tmp_path / "experiments" / "ax1_gate_fail_default",
        experiment_name="ax1_gate_fail_default",
    )

    with pytest.raises(ValueError, match="promotion gate failed.*min_excess_net_mean_return"):
        promote_package(
            experiment_dir=experiment_dir,
            packages_root=tmp_path / "packages",
        )


def test_feature_set_promotion_requires_lineage_and_clean_data_audit():
    from skyeye.products.ax1.promotion import validate_feature_set_promotion

    with pytest.raises(ValueError, match="lineage_id"):
        validate_feature_set_promotion({"data_audit": {"passed": True}})

    with pytest.raises(ValueError, match="data audit"):
        validate_feature_set_promotion(
            {
                "feature_set_candidate": {"lineage_id": "lin_1"},
                "data_audit": {"passed": False},
            }
        )


def test_build_feature_set_promotion_patch_carries_lineage_and_feature_set_version():
    from skyeye.products.ax1.promotion import build_feature_set_promotion_patch

    patch = build_feature_set_promotion_patch(
        {
            "feature_set_candidate": {
                "candidate_id": "candidate_1",
                "lineage_id": "lineage_20260429",
                "feature_set_version": "ax1_unified_v2",
                "features": ["feature_momentum_5d", "feature_z_excess_mom_20d"],
            },
            "data_audit": {"passed": True},
            "gate_summary": {"passed": True},
        }
    )

    assert patch["model"]["feature_set_version"] == "ax1_unified_v2"
    assert patch["model"]["feature_set_lineage_id"] == "lineage_20260429"
    assert patch["model"]["champion_experiment_id"] == "candidate_1"
    assert patch["model"]["feature_columns"] == ["feature_momentum_5d", "feature_z_excess_mom_20d"]
