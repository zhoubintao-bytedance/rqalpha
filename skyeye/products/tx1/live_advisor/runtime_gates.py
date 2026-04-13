# -*- coding: utf-8 -*-
"""TX1 live advisor 运行期闸门。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate_snapshot_runtime_gates(
    snapshot: dict,
    *,
    required_features: list[str],
    freshness_tolerance_days: int = 1,
    freshness_policy: dict | None = None,
    label_end_date: str | None = None,
    evidence_end_date: str | None = None,
) -> dict:
    """在出分前校验快照是否满足运行要求。"""
    reasons = []
    warnings = []
    diagnostics = {}
    requested_trade_date = pd.Timestamp(
        snapshot.get("requested_trade_date", snapshot["trade_date"])
    ).normalize()
    trade_date = pd.Timestamp(snapshot["trade_date"]).normalize()
    latest_available_trade_date = pd.Timestamp(
        snapshot.get("latest_available_trade_date", snapshot["trade_date"])
    ).normalize()
    raw_data_end_date = pd.Timestamp(snapshot["raw_data_end_date"]).normalize()
    requested_gap = int(snapshot.get("requested_vs_available_trading_gap", 0) or 0)
    resolved_policy = _resolve_freshness_policy(
        freshness_policy=freshness_policy,
        freshness_tolerance_days=freshness_tolerance_days,
    )
    snapshot_tolerance_days = int(resolved_policy["snapshot_max_delay_days"])
    if requested_gap > snapshot_tolerance_days:
        reasons.append(
            "freshness_requested_trade_date_exceeded:requested={} latest_available={} gap={}".format(
                requested_trade_date.date(),
                latest_available_trade_date.date(),
                requested_gap,
            )
        )
        warnings.append(
            {
                "level": "critical",
                "code": "requested_trade_date_stale",
                "message": (
                    "WARNING: requested_trade_date={} but latest_available_trade_date={}; "
                    "advisor stopped, old snapshot will not be used as today's advice."
                ).format(
                    requested_trade_date.date(),
                    latest_available_trade_date.date(),
                ),
            }
        )
    if (trade_date - raw_data_end_date).days > snapshot_tolerance_days:
        reasons.append("freshness_exceeded:data_end={} trade_date={}".format(raw_data_end_date.date(), trade_date.date()))

    feature_frame = snapshot.get("snapshot_features", pd.DataFrame())
    missing_columns = [feature_name for feature_name in required_features if feature_name not in feature_frame.columns]
    if missing_columns:
        reasons.append("feature_completeness_missing_columns:{}".format(",".join(sorted(missing_columns))))
    elif feature_frame[required_features].isna().any().any():
        reasons.append("feature_completeness_missing_values")

    history_counts = snapshot.get("history_counts", [])
    historical_values = []
    for item in history_counts:
        if isinstance(item, dict):
            historical_values.append(int(item.get("eligible_count", 0)))
        else:
            historical_values.append(int(item))
    history_median = float(np.median(historical_values)) if historical_values else 0.0
    total_candidates = int(snapshot.get("feature_coverage_summary", {}).get("total_candidates", len(feature_frame)))
    base_threshold = max(300, int(math.floor(history_median * 0.8)) if history_median > 0 else 300)
    threshold = base_threshold
    eligible_count = int(len(snapshot.get("eligible_universe", [])))
    if eligible_count < threshold:
        reasons.append("universe_coverage_below_threshold:{}<{}".format(eligible_count, threshold))

    model_diag = _evaluate_reference_freshness(
        reference_name="model",
        reference_date=label_end_date,
        trade_date=trade_date,
        warning_days=int(resolved_policy["model_warning_days"]),
        stop_days=int(resolved_policy["model_stop_days"]),
    )
    diagnostics["model_freshness"] = model_diag
    _merge_reference_freshness_result(reasons, warnings, model_diag)

    evidence_diag = _evaluate_reference_freshness(
        reference_name="evidence",
        reference_date=evidence_end_date,
        trade_date=trade_date,
        warning_days=int(resolved_policy["evidence_warning_days"]),
        stop_days=int(resolved_policy["evidence_stop_days"]),
    )
    diagnostics["evidence_freshness"] = evidence_diag
    _merge_reference_freshness_result(reasons, warnings, evidence_diag)

    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "eligible_count": eligible_count,
            "history_median_eligible_count": history_median,
            "total_candidates": total_candidates,
            "requested_vs_available_trading_gap": requested_gap,
            "model_freshness_gap_days": int(model_diag.get("gap_days", 0)),
            "evidence_freshness_gap_days": int(evidence_diag.get("gap_days", 0)),
        },
        "thresholds": {
            "snapshot_max_delay_days": snapshot_tolerance_days,
            "model_warning_days": int(resolved_policy["model_warning_days"]),
            "model_stop_days": int(resolved_policy["model_stop_days"]),
            "evidence_warning_days": int(resolved_policy["evidence_warning_days"]),
            "evidence_stop_days": int(resolved_policy["evidence_stop_days"]),
            "min_eligible_count": threshold,
        },
        "diagnostics": diagnostics,
    }


def evaluate_score_runtime_gates(
    scored_frame: pd.DataFrame,
    calibration_bundle: dict,
) -> dict:
    """在模型打分后检查分布是否明显塌缩。"""
    reasons = []
    if scored_frame is None or len(scored_frame) == 0:
        reasons.append("score_distribution_empty")
        return {"passed": False, "reasons": reasons, "metrics": {}}

    scores = pd.to_numeric(scored_frame["prediction"], errors="coerce").dropna()
    if scores.empty:
        reasons.append("score_distribution_all_nan")
        return {"passed": False, "reasons": reasons, "metrics": {}}

    score_std = float(scores.std(ddof=0))
    top_spread = float(scores.quantile(0.95) - scores.quantile(0.50))
    reference = calibration_bundle.get("score_sanity_reference", {})
    if score_std <= float(reference.get("prediction_std_p01", 0.0)):
        reasons.append("score_distribution_std_below_p01")
    if top_spread <= float(reference.get("top_spread_p05", 0.0)):
        reasons.append("score_distribution_spread_below_p05")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "prediction_std": score_std,
            "top_spread": top_spread,
        },
    }


def _resolve_freshness_policy(
    *,
    freshness_policy: dict | None,
    freshness_tolerance_days: int,
) -> dict:
    """把 manifest freshness 配置和旧参数兼容成统一结构。"""
    resolved = {
        "snapshot_max_delay_days": int(freshness_tolerance_days),
        "model_warning_days": 20,
        "model_stop_days": 40,
        "evidence_warning_days": 20,
        "evidence_stop_days": 40,
    }
    for key, value in (freshness_policy or {}).items():
        if key in resolved and value is not None:
            resolved[key] = int(value)
    return resolved


def _evaluate_reference_freshness(
    *,
    reference_name: str,
    reference_date: str | None,
    trade_date: pd.Timestamp,
    warning_days: int,
    stop_days: int,
) -> dict:
    """评估模型/证据日期相对 trade_date 的新鲜度等级。"""
    if not reference_date:
        return {
            "name": reference_name,
            "status": "unknown",
            "gap_days": 0,
            "reference_date": None,
            "warning_days": int(warning_days),
            "stop_days": int(stop_days),
        }
    reference_ts = pd.Timestamp(reference_date).normalize()
    gap_days = _count_business_gap(reference_ts, trade_date)
    status = "ok"
    if gap_days > int(stop_days):
        status = "stop"
    elif gap_days > int(warning_days):
        status = "warning"
    return {
        "name": reference_name,
        "status": status,
        "gap_days": int(gap_days),
        "reference_date": str(reference_ts.date()),
        "warning_days": int(warning_days),
        "stop_days": int(stop_days),
    }


def _merge_reference_freshness_result(
    reasons: list[str],
    warnings: list[dict],
    diagnostic: dict,
) -> None:
    """把 reference freshness 评估结果合并到 gate 返回值。"""
    name = str(diagnostic.get("name"))
    status = str(diagnostic.get("status"))
    gap_days = int(diagnostic.get("gap_days", 0))
    reference_date = diagnostic.get("reference_date")
    if status == "warning":
        warnings.append(
            {
                "level": "warning",
                "code": "{}_freshness_warning".format(name),
                "message": (
                    "WARNING: {}_end_date={} is {} trading days behind trade_date; "
                    "advisor is still running in warning mode."
                ).format(
                    name,
                    reference_date,
                    gap_days,
                ),
            }
        )
    elif status == "stop":
        reasons.append(
            "{}_freshness_exceeded:{}_end_date={} gap={}".format(
                name,
                name,
                reference_date,
                gap_days,
            )
        )
        warnings.append(
            {
                "level": "critical",
                "code": "{}_freshness_stop".format(name),
                "message": (
                    "WARNING: {}_end_date={} is too far behind trade_date; "
                    "advisor stopped to avoid stale {} evidence."
                ).format(
                    name,
                    reference_date,
                    name,
                ),
            }
        )


def _count_business_gap(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """用工作日近似交易日差，给 freshness gate 提供稳定口径。"""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts <= start_ts:
        return 0
    return max(len(pd.bdate_range(start_ts, end_ts)) - 1, 0)
