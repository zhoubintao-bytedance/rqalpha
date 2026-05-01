"""Phase 0 feature audit for AX1 feature-selection experiments."""

from __future__ import annotations

import math
from typing import Any


def build_phase0_feature_audit(
    training_summary: dict[str, Any],
    *,
    min_coverage: float = 0.50,
    low_std_threshold: float = 0.01,
    stable_negative_ic_threshold: float = -0.02,
    min_positive_fold_ratio: float = 0.50,
) -> dict[str, Any]:
    diagnostics = dict((training_summary or {}).get("feature_diagnostics") or {})
    conflicts = dict((training_summary or {}).get("feature_conflicts") or {})
    representative_by_feature = _representative_by_feature(conflicts)
    features: dict[str, Any] = {}

    for feature, metrics in sorted(diagnostics.items()):
        reasons: list[str] = []
        coverage = _num(metrics.get("coverage"))
        mean_std = _num(metrics.get("mean_cross_sectional_std"))
        rank_ic_mean = _num(metrics.get("rank_ic_mean"))
        spread = _num(metrics.get("top_bucket_spread_mean"))
        positive_ratio = _positive_fold_ratio(metrics)
        fold_values = _fold_rank_ics(metrics)
        t_stat, p_value = _mean_t_test(fold_values)
        representative = representative_by_feature.get(str(feature), str(feature))

        if coverage < float(min_coverage):
            reasons.append("low_coverage")
        if coverage >= float(min_coverage) and mean_std <= float(low_std_threshold):
            reasons.append("low_cross_sectional_variance")
        if rank_ic_mean <= float(stable_negative_ic_threshold) and positive_ratio < float(min_positive_fold_ratio):
            reasons.append("stable_negative_ic")
        if representative != str(feature):
            reasons.append("redundant_non_representative")

        decision = "hard_exclude" if reasons else "keep_candidate"
        if not reasons and p_value > 0.20 and positive_ratio < 0.67:
            decision = "watch"
            reasons.append("weak_statistical_evidence")

        features[str(feature)] = {
            "decision": decision,
            "reasons": reasons,
            "coverage": coverage,
            "mean_cross_sectional_std": mean_std,
            "rank_ic_mean": rank_ic_mean,
            "top_bucket_spread_mean": spread,
            "positive_fold_ratio": positive_ratio,
            "fold_count": int(len(fold_values)),
            "ic_t_stat": t_stat,
            "ic_p_value": p_value,
            "cluster_representative": representative,
        }

    return {
        "schema_version": 1,
        "label_column": ((training_summary or {}).get("feature_diagnostics_meta") or {}).get("label_column"),
        "feature_count": int(len(features)),
        "decision_counts": {
            decision: sum(1 for item in features.values() if item["decision"] == decision)
            for decision in ("keep_candidate", "watch", "hard_exclude")
        },
        "features": features,
    }


def _representative_by_feature(conflicts: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in conflicts.get("high_corr_groups", []) or []:
        representative = str(group.get("representative") or "")
        if not representative:
            continue
        for feature in group.get("features", []) or []:
            result[str(feature)] = representative
    return result


def _positive_fold_ratio(metrics: dict[str, Any]) -> float:
    fold_summary = metrics.get("fold_summary") or {}
    if "positive_ratio" in fold_summary:
        return _num(fold_summary.get("positive_ratio"))
    values = _fold_rank_ics(metrics)
    return float(sum(value > 0.0 for value in values) / len(values)) if values else 0.0


def _fold_rank_ics(metrics: dict[str, Any]) -> list[float]:
    values = []
    for fold in metrics.get("folds", []) or []:
        value = _num(fold.get("rank_ic_mean"))
        if math.isfinite(value):
            values.append(value)
    return values


def _mean_t_test(values: list[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0, 1.0
    mean = sum(clean) / float(len(clean))
    variance = sum((value - mean) ** 2 for value in clean) / float(len(clean) - 1)
    std = math.sqrt(max(variance, 0.0))
    if std <= 0.0:
        if mean > 0.0:
            return math.inf, 0.0
        if mean < 0.0:
            return -math.inf, 0.0
        return 0.0, 1.0
    t_stat = mean / (std / math.sqrt(float(len(clean))))
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return float(t_stat), float(p_value)


def _num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0
