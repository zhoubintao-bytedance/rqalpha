"""AX1 feature-set candidate judge."""

from __future__ import annotations

from typing import Any


DEFAULT_GUARDRAILS = {
    "max_drawdown": 0.20,
    "mean_turnover": 0.30,
    "stability_score": 30.0,
    "max_feature_diagnostics_warning_count": 3,
    "min_net_mean_return_delta": 0.0005,
    "min_top_bucket_spread_delta": 0.0002,
    "max_turnover_delta": 0.0,
    "min_stability_delta": 0.0,
}


def judge_feature_set_candidate(
    candidate_summary: dict[str, Any],
    *,
    baseline_summary: dict[str, Any] | None = None,
    best_summary: dict[str, Any] | None = None,
    guardrails: dict[str, float] | None = None,
) -> dict[str, Any]:
    data_audit = candidate_summary.get("data_audit") or {}
    if not bool(data_audit.get("passed", False)):
        return {
            "status": "discard",
            "reason_code": "data_audit_failed",
            "failed_guards": ["data_audit"],
            "score_delta": {},
        }

    limits = dict(DEFAULT_GUARDRAILS)
    limits.update(guardrails or {})
    metrics = _metrics(candidate_summary)
    failed = []
    if metrics["max_drawdown"] > limits["max_drawdown"]:
        failed.append("max_drawdown")
    if metrics["mean_turnover"] > limits["mean_turnover"]:
        failed.append("mean_turnover")
    if metrics["stability_score"] < limits["stability_score"]:
        failed.append("stability_score")
    warning_count = int((candidate_summary.get("feature_review_summary") or {}).get("warning_count", 0) or 0)
    if warning_count > limits["max_feature_diagnostics_warning_count"]:
        failed.append("feature_diagnostics_warning_count")
    robustness = _robustness(candidate_summary)
    bootstrap = robustness.get("bootstrap_ci") or {}
    if bootstrap and (bool(bootstrap.get("ci_crosses_zero", False)) or _num(bootstrap.get("ci_low")) < 0.0):
        failed.append("bootstrap_ci")
    if failed:
        return {
            "status": "discard",
            "reason_code": "guardrail_failed",
            "failed_guards": failed,
            "score_delta": _score_delta(candidate_summary, baseline_summary or {}),
        }

    best_delta = _score_delta(candidate_summary, best_summary or baseline_summary or {})
    if _materially_improves(best_delta, limits):
        return {
            "status": "champion",
            "reason_code": "full_improved",
            "failed_guards": [],
            "score_delta": _score_delta(candidate_summary, baseline_summary or {}),
            "best_score_delta": best_delta,
        }
    return {
        "status": "keep",
        "reason_code": "insufficient_material_improvement",
        "failed_guards": [],
        "score_delta": _score_delta(candidate_summary, baseline_summary or {}),
        "best_score_delta": best_delta,
    }


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics") or {}
    return {
        "net_mean_return": _num(metrics.get("net_mean_return")),
        "max_drawdown": _num(metrics.get("max_drawdown")),
        "mean_turnover": _num(metrics.get("mean_turnover")),
        "top_bucket_spread_mean": _num(metrics.get("top_bucket_spread_mean")),
        "stability_score": _num(metrics.get("stability_score")),
    }


def _robustness(summary: dict[str, Any]) -> dict[str, Any]:
    return (
        summary.get("robustness_summary")
        or (summary.get("training_summary") or {}).get("robustness")
        or {}
    )


def _score_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    candidate_metrics = _metrics(candidate)
    baseline_metrics = _metrics(baseline)
    return {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in candidate_metrics
    }


def _materially_improves(delta: dict[str, float], limits: dict[str, float]) -> bool:
    return (
        delta.get("net_mean_return", 0.0) >= float(limits["min_net_mean_return_delta"])
        and delta.get("top_bucket_spread_mean", 0.0) >= float(limits["min_top_bucket_spread_delta"])
        and delta.get("max_drawdown", 0.0) <= 0.0
        and delta.get("mean_turnover", 0.0) <= float(limits["max_turnover_delta"])
        and delta.get("stability_score", 0.0) >= float(limits["min_stability_delta"])
    )


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
