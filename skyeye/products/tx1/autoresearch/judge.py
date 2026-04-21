"""TX1 autoresearch 的候选判定逻辑。"""

from __future__ import annotations

from typing import Any


DEFAULT_GUARDRAILS = {
    "max_drawdown": 0.12,
    "mean_turnover": 0.20,
    "stability_score": 50.0,
    "cv": 0.60,
    "positive_ratio": 0.70,
}


def judge_candidate(
    candidate_summary: dict[str, Any] | None,
    *,
    baseline_summary: dict[str, Any] | None = None,
    best_summary: dict[str, Any] | None = None,
    stage: str = "full",
    guardrails: dict[str, float] | None = None,
) -> dict[str, Any]:
    """按“稳健优先、收益最大化”规则判定候选命运。"""
    if not candidate_summary:
        return {
            "status": "crash",
            "reason_code": "missing_summary",
            "failed_guards": ["summary_missing"],
            "score_delta": {},
        }

    baseline_summary = dict(baseline_summary or {})
    best_summary = dict(best_summary or baseline_summary)
    limits = _resolve_guardrails(baseline_summary=baseline_summary, guardrails=guardrails)
    failed_guards = _collect_failed_guards(candidate_summary, limits)

    if failed_guards:
        return {
            "status": "discard",
            "reason_code": "guardrail_failed",
            "failed_guards": failed_guards,
            "score_delta": _build_score_delta(candidate_summary, baseline_summary),
            "best_score_delta": _build_score_delta(candidate_summary, best_summary),
        }

    score_delta = _build_score_delta(candidate_summary, baseline_summary)
    best_score_delta = _build_score_delta(candidate_summary, best_summary)
    stage_name = str(stage).strip().lower()
    if stage_name == "smoke":
        return {
            "status": "keep",
            "reason_code": "smoke_pass",
            "failed_guards": [],
            "score_delta": score_delta,
            "best_score_delta": best_score_delta,
        }

    if _materially_improves(score_delta):
        if _materially_improves(best_score_delta):
            return {
                "status": "champion",
                "reason_code": "full_improved",
                "failed_guards": [],
                "score_delta": score_delta,
                "best_score_delta": best_score_delta,
            }
        return {
            "status": "keep",
            "reason_code": "full_pass_not_best",
            "failed_guards": [],
            "score_delta": score_delta,
            "best_score_delta": best_score_delta,
        }

    return {
        "status": "discard",
        "reason_code": "no_material_improvement",
        "failed_guards": [],
        "score_delta": score_delta,
        "best_score_delta": best_score_delta,
    }


def _collect_failed_guards(summary: dict[str, Any], limits: dict[str, float]) -> list[str]:
    """提取候选在硬门槛上失败的字段列表。"""
    failed = []
    if _metric(summary, "portfolio", "max_drawdown") > limits["max_drawdown"]:
        failed.append("max_drawdown")
    if _metric(summary, "portfolio", "mean_turnover") > limits["mean_turnover"]:
        failed.append("mean_turnover")
    if _metric(summary, "robustness", "stability", "stability_score") < limits["stability_score"]:
        failed.append("stability_score")
    if _metric(summary, "robustness", "stability", "cv") > limits["cv"]:
        failed.append("cv")
    if _metric(summary, "robustness", "regime_scores", "metric_consistency", "positive_ratio") < limits["positive_ratio"]:
        failed.append("positive_ratio")
    if _metric(summary, "robustness", "overfit_flags", "flag_ic_decay", default=False):
        failed.append("flag_ic_decay")
    if _metric(summary, "robustness", "overfit_flags", "flag_spread_decay", default=False):
        failed.append("flag_spread_decay")
    if _metric(summary, "robustness", "overfit_flags", "flag_val_dominant", default=False):
        failed.append("flag_val_dominant")
    return failed


def _resolve_guardrails(
    *,
    baseline_summary: dict[str, Any],
    guardrails: dict[str, float] | None,
) -> dict[str, float]:
    """把默认 guardrail 和基线相对 guardrail 合并成真实可用的门槛。"""
    limits = dict(DEFAULT_GUARDRAILS)
    limits.update(guardrails or {})
    if not baseline_summary:
        return limits

    baseline_stability = float(_metric(baseline_summary, "robustness", "stability", "stability_score"))
    baseline_cv = float(_metric(baseline_summary, "robustness", "stability", "cv"))
    baseline_positive_ratio = float(
        _metric(baseline_summary, "robustness", "regime_scores", "metric_consistency", "positive_ratio")
    )

    # TX1 当前默认线本身就处在“低稳定分、高 CV”的真实分布里。
    # 如果还坚持绝对门槛，autoresearch 会把所有同量级候选都误判为 invalid。
    limits["stability_score"] = min(
        limits["stability_score"],
        max(10.0, baseline_stability * 0.85),
    )
    limits["cv"] = max(limits["cv"], baseline_cv + 0.15)
    limits["positive_ratio"] = max(
        min(limits["positive_ratio"], baseline_positive_ratio),
        baseline_positive_ratio - 0.10,
    )
    return limits


def _build_score_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """构造相对 baseline 的关键指标差值。"""
    return {
        "net_mean_return": _metric(candidate, "portfolio", "net_mean_return") - _metric(
            baseline, "portfolio", "net_mean_return"
        ),
        "top_bucket_spread_mean": _metric(candidate, "prediction", "top_bucket_spread_mean") - _metric(
            baseline, "prediction", "top_bucket_spread_mean"
        ),
        "rank_ic_mean": _metric(candidate, "prediction", "rank_ic_mean") - _metric(
            baseline, "prediction", "rank_ic_mean"
        ),
        "max_drawdown": _metric(candidate, "portfolio", "max_drawdown") - _metric(
            baseline, "portfolio", "max_drawdown"
        ),
        "stability_score": _metric(candidate, "robustness", "stability", "stability_score") - _metric(
            baseline, "robustness", "stability", "stability_score"
        ),
    }


def _materially_improves(score_delta: dict[str, float]) -> bool:
    """判断候选是否在收益和稳健性上形成材料性改进。"""
    return (
        score_delta["net_mean_return"] > 0.0
        and score_delta["max_drawdown"] <= 0.0
        and score_delta["stability_score"] >= 0.0
    )


def _metric(payload: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    """安全读取嵌套指标，缺失时返回默认值。"""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
