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
) -> dict:
    """在出分前校验快照是否满足运行要求。"""
    reasons = []
    warnings = []
    requested_trade_date = pd.Timestamp(
        snapshot.get("requested_trade_date", snapshot["trade_date"])
    ).normalize()
    trade_date = pd.Timestamp(snapshot["trade_date"]).normalize()
    latest_available_trade_date = pd.Timestamp(
        snapshot.get("latest_available_trade_date", snapshot["trade_date"])
    ).normalize()
    raw_data_end_date = pd.Timestamp(snapshot["raw_data_end_date"]).normalize()
    requested_gap = int(snapshot.get("requested_vs_available_trading_gap", 0) or 0)
    if requested_gap > int(freshness_tolerance_days):
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
    if (trade_date - raw_data_end_date).days > int(freshness_tolerance_days):
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

    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "eligible_count": eligible_count,
            "history_median_eligible_count": history_median,
            "total_candidates": total_candidates,
            "requested_vs_available_trading_gap": requested_gap,
        },
        "thresholds": {
            "freshness_tolerance_days": int(freshness_tolerance_days),
            "min_eligible_count": threshold,
        },
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
