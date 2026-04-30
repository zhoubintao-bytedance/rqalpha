# -*- coding: utf-8 -*-
"""AX1 research package promotion gate helpers."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


DEFAULT_PROMOTION_THRESHOLDS: dict[str, dict[str, float]] = {
    "canary_live": {
        "min_folds": 6,
        "min_oos_rows": 100,
        "min_rank_ic_mean": 0.02,
        "min_top_bucket_spread_mean": 0.0,
        "min_positive_ratio": 0.50,
        "max_cv": 0.50,
        "min_stability_score": 30.0,
        "catastrophic_max_drawdown": 0.35,
        "min_excess_net_mean_return": 0.0,
        "max_excess_drawdown": 0.08,
        "min_alpha_hit_rate": 0.50,
        "max_rolling_underperformance": 0.10,
        "min_active_day_ratio": 0.0,
        "max_mean_turnover": 0.30,
        "max_feature_diagnostics_warning_count": 0,
        "max_parameter_validation_warning_count": 0,
        "min_bootstrap_ci_low": 0.0,
        "max_robustness_warning_count": 0,
        "min_effective_breadth": 10.0,
        "min_effective_breadth_ratio": 0.50,
    },
    "default_live": {
        "min_folds": 6,
        "min_oos_rows": 300,
        "min_rank_ic_mean": 0.02,
        "min_top_bucket_spread_mean": 0.0,
        "min_positive_ratio": 0.55,
        "max_cv": 0.40,
        "min_stability_score": 40.0,
        "catastrophic_max_drawdown": 0.35,
        "min_excess_net_mean_return": 0.0,
        "max_excess_drawdown": 0.06,
        "min_alpha_hit_rate": 0.52,
        "max_rolling_underperformance": 0.08,
        "min_active_day_ratio": 0.50,
        "max_mean_turnover": 0.25,
        "max_feature_diagnostics_warning_count": 0,
        "max_parameter_validation_warning_count": 0,
        "min_bootstrap_ci_low": 0.0,
        "max_robustness_warning_count": 0,
        "min_effective_breadth": 15.0,
        "min_effective_breadth_ratio": 0.50,
    },
}


TRADABILITY_CHECK_NAMES = {
    "opportunity_benchmark_available",
    "catastrophic_max_drawdown",
    "min_excess_net_mean_return",
    "max_excess_drawdown",
    "min_alpha_hit_rate",
    "max_rolling_underperformance",
    "min_active_day_ratio",
    "max_mean_turnover",
}


def evaluate_promotion_gate(
    experiment_result: dict[str, Any],
    *,
    gate_level: str = "canary_live",
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate whether an AX1 research result passes a promotion gate."""
    if gate_level not in DEFAULT_PROMOTION_THRESHOLDS:
        raise ValueError(f"unsupported gate_level: {gate_level}")

    merged_thresholds = _merge_thresholds(thresholds)
    gate_thresholds = merged_thresholds[gate_level]
    metrics = _extract_gate_metrics(experiment_result)
    checks = {
        "min_folds": _build_check(metrics["n_folds"], ">=", gate_thresholds["min_folds"]),
        "min_oos_rows": _build_check(metrics["oos_rows"], ">=", gate_thresholds["min_oos_rows"]),
        "min_rank_ic_mean": _build_check(
            metrics["rank_ic_mean"],
            ">=",
            gate_thresholds["min_rank_ic_mean"],
            hard_gate=False,
        ),
        "min_top_bucket_spread_mean": _build_check(
            metrics["top_bucket_spread_mean"],
            ">=",
            gate_thresholds["min_top_bucket_spread_mean"],
            hard_gate=False,
        ),
        "min_positive_ratio": _build_check(
            metrics["positive_ratio"],
            ">=",
            gate_thresholds["min_positive_ratio"],
        ),
        "max_cv": _build_check(metrics["cv"], "<=", gate_thresholds["max_cv"]),
        "min_stability_score": _build_check(
            metrics["stability_score"],
            ">=",
            gate_thresholds["min_stability_score"],
        ),
        "opportunity_benchmark_available": _build_flag_check(
            metrics["opportunity_benchmark_available"],
            expected=True,
        ),
        "catastrophic_max_drawdown": _build_check(
            metrics["max_drawdown"],
            "<=",
            gate_thresholds["catastrophic_max_drawdown"],
        ),
        "min_excess_net_mean_return": _build_check(
            metrics["excess_net_mean_return"],
            ">",
            gate_thresholds["min_excess_net_mean_return"],
        ),
        "max_excess_drawdown": _build_check(
            metrics["max_excess_drawdown"],
            "<=",
            gate_thresholds["max_excess_drawdown"],
        ),
        "min_alpha_hit_rate": _build_check(
            metrics["alpha_hit_rate"],
            ">=",
            gate_thresholds["min_alpha_hit_rate"],
        ),
        "max_rolling_underperformance": _build_check(
            metrics["max_rolling_underperformance"],
            "<=",
            gate_thresholds["max_rolling_underperformance"],
        ),
        "min_active_day_ratio": _build_check(
            metrics["active_day_ratio"],
            ">=",
            gate_thresholds["min_active_day_ratio"],
            hard_gate=False,
        ),
        "max_mean_turnover": _build_check(
            metrics["mean_turnover"],
            "<=",
            gate_thresholds["max_mean_turnover"],
        ),
        "flag_ic_decay": _build_flag_check(metrics["flag_ic_decay"], expected=False, hard_gate=False),
        "flag_spread_decay": _build_flag_check(metrics["flag_spread_decay"], expected=False, hard_gate=False),
        "flag_val_dominant": _build_flag_check(metrics["flag_val_dominant"], expected=False, hard_gate=False),
        "max_feature_diagnostics_warning_count": _build_check(
            metrics["feature_diagnostics_warning_count"],
            "<=",
            gate_thresholds["max_feature_diagnostics_warning_count"],
            hard_gate=False,
        ),
        "max_parameter_validation_warning_count": _build_check(
            metrics["parameter_validation_warning_count"],
            "<=",
            gate_thresholds["max_parameter_validation_warning_count"],
            hard_gate=False,
        ),
        "min_bootstrap_ci_low": _build_check(
            metrics["bootstrap_ci_low"],
            ">=",
            gate_thresholds["min_bootstrap_ci_low"],
            hard_gate=False,
        ),
        "rank_ic_fdr_significant": _build_flag_check(
            metrics["rank_ic_fdr_significant"],
            expected=True,
            hard_gate=False,
        ),
        "bootstrap_ci_crosses_zero": _build_flag_check(
            metrics["bootstrap_ci_crosses_zero"],
            expected=False,
            hard_gate=False,
        ),
        "sample_decay_flag_late_decay": _build_flag_check(
            metrics["sample_decay_flag_late_decay"],
            expected=False,
            hard_gate=False,
        ),
        "max_robustness_warning_count": _build_check(
            metrics["robustness_warning_count"],
            "<=",
            gate_thresholds["max_robustness_warning_count"],
            hard_gate=False,
        ),
        "min_effective_breadth": _build_check(
            metrics["effective_breadth"],
            ">=",
            gate_thresholds["min_effective_breadth"],
            hard_gate=False,
        ),
        "min_effective_breadth_ratio": _build_check(
            metrics["effective_breadth_ratio"],
            ">=",
            gate_thresholds["min_effective_breadth_ratio"],
            hard_gate=False,
        ),
    }
    tradability_gate = _build_sub_gate("tradability", checks, TRADABILITY_CHECK_NAMES)
    research_check_names = set(checks) - TRADABILITY_CHECK_NAMES
    research_support_gate = _build_sub_gate("research_support", checks, research_check_names)
    failed_checks = [
        *tradability_gate["failed_checks"],
        *research_support_gate["failed_checks"],
    ]
    return {
        "gate_level": gate_level,
        "passed": bool(tradability_gate["passed"] and research_support_gate["passed"]),
        "failed_checks": failed_checks,
        "tradability_gate": tradability_gate,
        "research_support_gate": research_support_gate,
        "checks": checks,
        "metrics": metrics,
        "thresholds": deepcopy(gate_thresholds),
    }


def validate_feature_set_promotion(experiment_result: dict[str, Any]) -> dict[str, Any]:
    """Validate that a feature-set champion can be promoted into a profile patch."""

    candidate = experiment_result.get("feature_set_candidate") or {}
    lineage_id = candidate.get("lineage_id")
    if not lineage_id:
        raise ValueError("feature set promotion requires lineage_id")
    data_audit = experiment_result.get("data_audit") or {}
    if not bool(data_audit.get("passed", False)):
        raise ValueError("feature set promotion requires a passing data audit")
    for quality_key in ("raw_data_quality", "feature_data_quality"):
        quality_report = experiment_result.get(quality_key)
        if quality_report is not None and not bool((quality_report or {}).get("passed", False)):
            raise ValueError(f"feature set promotion requires a passing {quality_key}")
    gate_summary = experiment_result.get("gate_summary")
    if gate_summary is not None and not bool(gate_summary.get("passed", False)):
        raise ValueError("feature set promotion requires a passing promotion gate")
    features = candidate.get("features") or candidate.get("feature_columns") or []
    if not features:
        raise ValueError("feature set promotion requires non-empty features")
    return dict(candidate)


def build_feature_set_promotion_patch(experiment_result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-friendly profile patch for an explicitly approved champion."""

    candidate = validate_feature_set_promotion(experiment_result)
    feature_set_version = str(candidate.get("feature_set_version") or candidate.get("lineage_id"))
    features = [str(feature) for feature in (candidate.get("features") or candidate.get("feature_columns") or [])]
    return {
        "model": {
            "feature_set_version": feature_set_version,
            "feature_set_lineage_id": str(candidate["lineage_id"]),
            "champion_experiment_id": str(candidate.get("candidate_id") or candidate.get("experiment_id") or ""),
            "feature_columns": features,
        }
    }


def _merge_thresholds(
    overrides: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    merged = deepcopy(DEFAULT_PROMOTION_THRESHOLDS)
    for level, values in (overrides or {}).items():
        if level not in merged:
            raise ValueError(f"unsupported gate_level in thresholds: {level}")
        merged[level].update({str(key): float(value) for key, value in values.items()})
    return merged


def _extract_gate_metrics(experiment_result: dict[str, Any]) -> dict[str, Any]:
    training_summary = experiment_result.get("training_summary") or {}
    aggregate_metrics = training_summary.get("aggregate_metrics") or experiment_result.get("aggregate_metrics") or {}
    prediction_metrics = aggregate_metrics.get("prediction") or {}
    stability = training_summary.get("stability") or {}
    positive_ratio = training_summary.get("positive_ratio") or {}
    overfit_flags = training_summary.get("overfit_flags") or {}
    portfolio_metrics = (experiment_result.get("evaluation") or {}).get("portfolio") or {}
    signal_metrics = (experiment_result.get("evaluation") or {}).get("signal") or {}
    calibration_bundle = experiment_result.get("calibration_bundle") or {}
    calibration_summary = calibration_bundle.get("summary") or {}
    feature_review_summary = (
        training_summary.get("feature_review_summary")
        or experiment_result.get("feature_review_summary")
        or {}
    )
    parameter_validation_summary = experiment_result.get("parameter_validation_summary") or {}
    robustness_summary = (
        training_summary.get("robustness")
        or experiment_result.get("robustness_summary")
        or {}
    )
    bootstrap_ci = robustness_summary.get("bootstrap_ci") or {}
    sample_decay = robustness_summary.get("sample_decay") or {}
    effective_breadth = (
        robustness_summary.get("effective_breadth")
        or experiment_result.get("effective_breadth_summary")
        or portfolio_metrics.get("effective_breadth")
        or {}
    )
    latest_breadth = effective_breadth.get("latest") or {}
    rank_ic_significance = signal_metrics.get("rank_ic_significance") or {}

    fold_results = training_summary.get("fold_results") or experiment_result.get("fold_results") or []
    n_folds = _first_number(
        aggregate_metrics.get("n_folds"),
        training_summary.get("n_folds"),
        len(fold_results),
    )
    return {
        "n_folds": int(n_folds),
        "oos_rows": int(_first_number(calibration_summary.get("oos_rows"), training_summary.get("aggregate_predictions_row_count"), 0)),
        "rank_ic_mean": _first_number(
            prediction_metrics.get("rank_ic_mean_mean"),
            prediction_metrics.get("rank_ic_mean"),
            0.0,
        ),
        "top_bucket_spread_mean": _first_number(
            prediction_metrics.get("top_bucket_spread_mean_mean"),
            prediction_metrics.get("top_bucket_spread_mean"),
            0.0,
        ),
        "positive_ratio": _first_number(positive_ratio.get("positive_ratio"), 0.0),
        "cv": _first_number(stability.get("cv"), math.inf),
        "stability_score": _first_number(stability.get("stability_score"), 0.0),
        "net_mean_return": _first_number(portfolio_metrics.get("net_mean_return"), 0.0),
        "max_drawdown": _first_number(portfolio_metrics.get("max_drawdown"), math.inf),
        "opportunity_benchmark_available": bool(portfolio_metrics.get("opportunity_benchmark_available", False)),
        "excess_net_mean_return": _first_number(portfolio_metrics.get("excess_net_mean_return"), -math.inf),
        "max_excess_drawdown": _first_number(portfolio_metrics.get("max_excess_drawdown"), math.inf),
        "alpha_hit_rate": _first_number(portfolio_metrics.get("alpha_hit_rate"), -math.inf),
        "max_rolling_underperformance": _first_number(portfolio_metrics.get("max_rolling_underperformance"), math.inf),
        "active_day_ratio": _first_number(portfolio_metrics.get("active_day_ratio"), 0.0),
        "mean_turnover": _first_number(portfolio_metrics.get("mean_turnover"), math.inf),
        "annual_turnover": _first_number(portfolio_metrics.get("annual_turnover"), 0.0),
        "flag_ic_decay": bool(overfit_flags.get("flag_ic_decay", False)),
        "flag_spread_decay": bool(overfit_flags.get("flag_spread_decay", False)),
        "flag_val_dominant": bool(overfit_flags.get("flag_val_dominant", False)),
        "feature_diagnostics_warning_count": int(
            _first_number(feature_review_summary.get("warning_count"), 0)
        ),
        "parameter_validation_warning_count": int(
            _first_number(parameter_validation_summary.get("warning_count"), 0)
        ),
        "bootstrap_ci_low": _first_number(bootstrap_ci.get("ci_low"), 0.0),
        "bootstrap_ci_crosses_zero": bool(bootstrap_ci.get("ci_crosses_zero", False)),
        "rank_ic_p_value": _first_number(rank_ic_significance.get("p_value"), 1.0),
        "rank_ic_fdr_adjusted_p_value": _first_number(rank_ic_significance.get("fdr_adjusted_p_value"), 1.0),
        "rank_ic_fdr_significant": bool(rank_ic_significance.get("significant_at_5pct", True)),
        "sample_decay_flag_late_decay": bool(sample_decay.get("flag_late_decay", False)),
        "robustness_warning_count": int(_first_number(robustness_summary.get("warning_count"), 0)),
        "effective_breadth": _first_number(
            effective_breadth.get("p5_effective_breadth"),
            effective_breadth.get("mean_effective_breadth"),
            latest_breadth.get("effective_breadth"),
            math.inf,
        ),
        "effective_breadth_ratio": _first_number(
            effective_breadth.get("p5_breadth_ratio"),
            effective_breadth.get("mean_breadth_ratio"),
            latest_breadth.get("breadth_ratio"),
            math.inf,
        ),
    }


def _build_check(actual: float, operator: str, threshold: float, *, hard_gate: bool = True) -> dict[str, Any]:
    actual_value = float(actual)
    threshold_value = float(threshold)
    if operator == ">=":
        passed = actual_value >= threshold_value
    elif operator == ">":
        passed = actual_value > threshold_value
    elif operator == "<=":
        passed = actual_value <= threshold_value
    else:
        raise ValueError(f"unsupported check operator: {operator}")
    return {
        "actual": actual_value,
        "operator": operator,
        "threshold": threshold_value,
        "passed": bool(passed),
        "hard_gate": bool(hard_gate),
    }


def _build_flag_check(actual: bool, *, expected: bool, hard_gate: bool = True) -> dict[str, Any]:
    passed = bool(actual) is bool(expected)
    return {
        "actual": bool(actual),
        "operator": "is",
        "threshold": bool(expected),
        "passed": bool(passed),
        "hard_gate": bool(hard_gate),
    }


def _build_sub_gate(name: str, checks: dict[str, dict[str, Any]], check_names: set[str]) -> dict[str, Any]:
    selected_names = [check_name for check_name in checks if check_name in check_names]
    selected_checks = {check_name: checks[check_name] for check_name in selected_names}
    failed_checks = [
        check_name
        for check_name, check in selected_checks.items()
        if check.get("hard_gate", True) and not bool(check.get("passed", False))
    ]
    soft_failed_checks = [
        check_name
        for check_name, check in selected_checks.items()
        if not check.get("hard_gate", True) and not bool(check.get("passed", False))
    ]
    return {
        "name": name,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "soft_failed_checks": soft_failed_checks,
        "checks": selected_checks,
    }


def _first_number(*values: Any) -> float:
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        return numeric
    return 0.0
