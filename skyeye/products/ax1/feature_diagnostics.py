"""AX1 feature diagnostics helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CONFLICT_ABS_CORR_THRESHOLD = 0.85
DEFAULT_MIN_COVERAGE = 0.50
# Diagnostics run on the pre-standardization panel, so this threshold should be
# large enough to catch features that are effectively constant cross-sectionally.
DEFAULT_LOW_STD_THRESHOLD = 0.01
DEFAULT_NEGATIVE_IC_THRESHOLD = -0.02


@dataclass(frozen=True)
class _FoldInput:
    fold_id: str
    features: pd.DataFrame
    labels: pd.DataFrame


def analyze_feature_diagnostics(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str | None = None,
    fold_id: int | str | None = None,
    top_k: int = 5,
    n_groups: int = 5,
    conflict_abs_corr_threshold: float = DEFAULT_CONFLICT_ABS_CORR_THRESHOLD,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    low_std_threshold: float = DEFAULT_LOW_STD_THRESHOLD,
    negative_ic_threshold: float = DEFAULT_NEGATIVE_IC_THRESHOLD,
) -> dict[str, Any]:
    """Compute JSON-friendly OOS diagnostics for one feature panel."""

    fold_key = "fold_0" if fold_id is None else str(fold_id)
    return analyze_fold_feature_diagnostics(
        fold_results=[{"fold_id": fold_key, "features_df": features, "labels": labels}],
        feature_columns=list(feature_columns),
        label_column=label_column,
        top_k=top_k,
        n_groups=n_groups,
        conflict_abs_corr_threshold=conflict_abs_corr_threshold,
        min_coverage=min_coverage,
        low_std_threshold=low_std_threshold,
        negative_ic_threshold=negative_ic_threshold,
    )


def analyze_fold_feature_diagnostics(
    *,
    fold_results: Iterable[dict[str, Any]],
    feature_columns: Iterable[str] | None = None,
    label_column: str | None = None,
    top_k: int = 5,
    n_groups: int = 5,
    conflict_abs_corr_threshold: float = DEFAULT_CONFLICT_ABS_CORR_THRESHOLD,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    low_std_threshold: float = DEFAULT_LOW_STD_THRESHOLD,
    negative_ic_threshold: float = DEFAULT_NEGATIVE_IC_THRESHOLD,
) -> dict[str, Any]:
    """Compute feature diagnostics from runner-style walk-forward fold results."""

    folds = _coerce_fold_inputs(fold_results)
    resolved_features = _resolve_feature_columns(folds, feature_columns)
    resolved_label = _resolve_label_column(folds, label_column)
    per_feature_fold_metrics: dict[str, list[dict[str, Any]]] = {feature: [] for feature in resolved_features}
    merged_folds: list[pd.DataFrame] = []

    for fold in folds:
        merged = _merge_features_and_labels(fold.features, fold.labels, resolved_features, resolved_label)
        if merged.empty:
            continue
        merged["_fold_id"] = fold.fold_id
        merged_folds.append(merged)
        for feature in resolved_features:
            per_feature_fold_metrics[feature].append(
                _compute_feature_metrics(
                    merged=merged,
                    feature=feature,
                    label_column=resolved_label,
                    fold_id=fold.fold_id,
                    top_k=top_k,
                    n_groups=n_groups,
                )
            )

    diagnostics = {
        feature: _aggregate_feature_metrics(fold_metrics)
        for feature, fold_metrics in per_feature_fold_metrics.items()
    }
    combined = pd.concat(merged_folds, ignore_index=True) if merged_folds else pd.DataFrame()
    conflicts = compute_feature_conflicts(
        combined,
        feature_columns=resolved_features,
        diagnostics=diagnostics,
        abs_corr_threshold=conflict_abs_corr_threshold,
    )
    review = build_feature_review_summary(
        diagnostics=diagnostics,
        conflicts=conflicts,
        min_coverage=min_coverage,
        low_std_threshold=low_std_threshold,
        negative_ic_threshold=negative_ic_threshold,
    )
    scorecards = build_factor_scorecards(diagnostics=diagnostics, review_summary=review)
    return _json_ready(
        {
            "feature_diagnostics": diagnostics,
            "feature_conflicts": conflicts,
            "feature_review_summary": review,
            "factor_scorecards": scorecards,
            "meta": {
                "feature_count": len(resolved_features),
                "fold_count": len(folds),
                "label_column": resolved_label,
                "top_k": int(top_k),
                "n_groups": int(n_groups),
                "conflict_abs_corr_threshold": float(conflict_abs_corr_threshold),
            },
        }
    )


def compute_feature_conflicts(
    panel: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    diagnostics: dict[str, Any] | None = None,
    abs_corr_threshold: float = DEFAULT_CONFLICT_ABS_CORR_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate daily cross-sectional rank correlations into conflict groups."""

    features = [feature for feature in feature_columns if feature in panel.columns]
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    if not panel.empty and "date" in panel.columns:
        for _, day_df in panel.groupby("date", sort=True):
            clean = day_df[features].apply(pd.to_numeric, errors="coerce")
            for left_idx, left in enumerate(features):
                for right in features[left_idx + 1 :]:
                    pair = clean[[left, right]].dropna()
                    if len(pair) < 2:
                        continue
                    left_rank = pair[left].rank(method="average")
                    right_rank = pair[right].rank(method="average")
                    if left_rank.nunique() <= 1 or right_rank.nunique() <= 1:
                        continue
                    corr = left_rank.corr(right_rank)
                    if pd.notna(corr) and np.isfinite(corr):
                        pair_values[(left, right)].append(float(corr))

    pairs: list[dict[str, Any]] = []
    positive_edges: list[tuple[str, str]] = []
    inverse_edges: list[tuple[str, str]] = []
    for (left, right), values in sorted(pair_values.items()):
        mean_corr = float(np.mean(values)) if values else 0.0
        pair = {
            "feature_a": left,
            "feature_b": right,
            "mean_rank_corr": mean_corr,
            "abs_mean_rank_corr": abs(mean_corr),
            "date_count": len(values),
        }
        pairs.append(pair)
        if mean_corr >= abs_corr_threshold:
            positive_edges.append((left, right))
        elif mean_corr <= -abs_corr_threshold:
            inverse_edges.append((left, right))

    return {
        "pairwise": pairs,
        "high_corr_groups": _build_conflict_groups(positive_edges, diagnostics or {}),
        "inverse_corr_groups": _build_conflict_groups(inverse_edges, diagnostics or {}),
        "abs_corr_threshold": float(abs_corr_threshold),
    }


def build_feature_review_summary(
    *,
    diagnostics: dict[str, Any],
    conflicts: dict[str, Any],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    low_std_threshold: float = DEFAULT_LOW_STD_THRESHOLD,
    negative_ic_threshold: float = DEFAULT_NEGATIVE_IC_THRESHOLD,
) -> dict[str, Any]:
    """Classify features as keep/review/drop_candidate without mutating the feature set."""

    redundant_features: set[str] = set()
    for group in conflicts.get("high_corr_groups", []):
        representative = group.get("representative")
        for feature in group.get("features", []):
            if feature != representative:
                redundant_features.add(feature)

    feature_reviews: dict[str, Any] = {}
    for feature, metrics in diagnostics.items():
        reasons: list[str] = []
        coverage = _as_float(metrics.get("coverage"))
        mean_std = _as_float(metrics.get("mean_cross_sectional_std"))
        rank_ic_mean = _as_float(metrics.get("rank_ic_mean"))
        if coverage < min_coverage:
            reasons.append("low_coverage")
        if coverage >= min_coverage and mean_std <= low_std_threshold:
            reasons.append("low_cross_sectional_variance")
        if rank_ic_mean <= negative_ic_threshold:
            reasons.append("negative_rank_ic")
        if feature in redundant_features:
            reasons.append("redundant_weaker_than_group_representative")

        drop_reasons = {
            "low_cross_sectional_variance",
            "negative_rank_ic",
            "redundant_weaker_than_group_representative",
        }
        if drop_reasons.intersection(reasons):
            decision = "drop_candidate"
        elif reasons:
            decision = "review"
        else:
            decision = "keep"
        feature_reviews[feature] = {
            "decision": decision,
            "reasons": reasons,
            "rank_ic_mean": rank_ic_mean,
            "coverage": coverage,
        }
    warnings = [
        {
            "feature": feature,
            "decision": item["decision"],
            "reasons": list(item["reasons"]),
        }
        for feature, item in sorted(feature_reviews.items())
        if item["decision"] != "keep"
    ]
    decision_counts = {
        decision: sum(1 for item in feature_reviews.values() if item["decision"] == decision)
        for decision in ("keep", "review", "drop_candidate")
    }
    return {
        "schema_version": 1,
        "feature_count": len(feature_reviews),
        "warning_count": len(warnings),
        "decision_counts": decision_counts,
        "warnings": warnings,
        "features": feature_reviews,
    }


def build_factor_scorecards(
    *,
    diagnostics: dict[str, Any],
    review_summary: dict[str, Any],
    model_importance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build stable per-factor scores for downstream feature-set search."""

    review_by_feature = (review_summary or {}).get("features") or {}
    importance_by_feature = _importance_by_feature(model_importance or {})
    scorecards: dict[str, Any] = {}
    for feature, metrics in diagnostics.items():
        rank_ic = _as_float(metrics.get("rank_ic_mean"))
        spread = _as_float(metrics.get("top_bucket_spread_mean"))
        monotonicity = _as_float(metrics.get("group_monotonicity"))
        fold_summary = metrics.get("fold_summary") or {}
        positive_ratio = _as_float(fold_summary.get("positive_ratio"))
        cv = abs(_as_float(fold_summary.get("cv")))
        review = review_by_feature.get(feature) or {}
        reasons = set(review.get("reasons") or [])

        signal_score = rank_ic + (0.20 * _signed_unit(spread)) + (0.10 * monotonicity)
        stability_score = positive_ratio - min(cv, 2.0) * 0.10
        redundancy_penalty = 0.35 if "redundant_weaker_than_group_representative" in reasons else 0.0
        data_quality_penalty = 0.0
        if "low_coverage" in reasons:
            data_quality_penalty += 0.25
        if "low_cross_sectional_variance" in reasons:
            data_quality_penalty += 0.35
        if "negative_rank_ic" in reasons:
            data_quality_penalty += 0.20
        model_usage_score = _as_float(importance_by_feature.get(feature))
        final_score = signal_score + (0.25 * stability_score) + (0.10 * model_usage_score) - redundancy_penalty - data_quality_penalty
        scorecards[feature] = {
            "schema_version": 1,
            "signal_score": float(signal_score),
            "stability_score": float(stability_score),
            "redundancy_penalty": float(redundancy_penalty),
            "model_usage_score": float(model_usage_score),
            "data_quality_penalty": float(data_quality_penalty),
            "final_factor_score": float(final_score),
            "decision": str(review.get("decision", "review")),
            "reasons": list(review.get("reasons") or []),
        }
    return scorecards


def _coerce_fold_inputs(fold_results: Iterable[dict[str, Any]]) -> list[_FoldInput]:
    folds: list[_FoldInput] = []
    for idx, fold in enumerate(fold_results):
        features = fold.get("features_df")
        if features is None:
            features = fold.get("features")
        if features is None:
            features = fold.get("predictions_df")
        labels = fold.get("labels")
        if labels is None:
            labels = fold.get("labels_df")
        if features is None or labels is None:
            continue
        fold_id = fold.get("fold_id", idx)
        folds.append(_FoldInput(fold_id=str(fold_id), features=features.copy(), labels=labels.copy()))
    if not folds:
        raise ValueError("feature diagnostics require at least one fold with features_df/features and labels")
    return folds


def _resolve_feature_columns(folds: list[_FoldInput], feature_columns: Iterable[str] | None) -> list[str]:
    if feature_columns is not None:
        columns = [column for column in feature_columns if column]
    else:
        key_columns = {"date", "order_book_id", "fold_id"}
        first = folds[0].features
        columns = [
            column
            for column in first.columns
            if column not in key_columns and pd.api.types.is_numeric_dtype(first[column])
        ]
    missing = [column for column in columns if not any(column in fold.features.columns for fold in folds)]
    if missing:
        raise ValueError(f"feature diagnostics missing feature columns: {missing}")
    return columns


def _resolve_label_column(folds: list[_FoldInput], label_column: str | None) -> str:
    if label_column:
        if not any(label_column in fold.labels.columns for fold in folds):
            raise ValueError(f"feature diagnostics missing label column: {label_column}")
        return label_column
    candidates: list[str] = []
    for fold in folds:
        candidates.extend(
            column
            for column in fold.labels.columns
            if column.startswith("label_relative_net_return_")
            or column.startswith("label_net_return_")
            or column.startswith("label_return_")
        )
    if not candidates:
        raise ValueError("feature diagnostics require label_column or AX1 return label column")
    return sorted(set(candidates))[0]


def _merge_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    keys = ["date", "order_book_id"]
    if "fold_id" in features.columns and "fold_id" in labels.columns:
        keys.append("fold_id")
    missing_keys = [key for key in keys if key not in features.columns or key not in labels.columns]
    if missing_keys:
        raise ValueError(f"feature diagnostics missing panel keys: {missing_keys}")
    left_columns = keys + [column for column in feature_columns if column in features.columns]
    right_columns = keys + [label_column]
    merged = features[left_columns].merge(labels[right_columns], on=keys, how="inner")
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    return merged


def _compute_feature_metrics(
    *,
    merged: pd.DataFrame,
    feature: str,
    label_column: str,
    fold_id: str,
    top_k: int,
    n_groups: int,
) -> dict[str, Any]:
    total_count = int(len(merged))
    values = pd.to_numeric(merged[feature], errors="coerce")
    labels = pd.to_numeric(merged[label_column], errors="coerce")
    valid_mask = values.notna() & labels.notna()
    valid_count = int(valid_mask.sum())
    daily_stds: list[float] = []
    rank_ics: list[float] = []
    spreads: list[float] = []
    hit_rates: list[float] = []
    group_returns: dict[int, list[float]] = {group: [] for group in range(n_groups)}

    for _, day_df in merged.groupby("date", sort=True):
        clean = day_df[[feature, label_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) >= 2:
            day_std = clean[feature].std(ddof=0)
            if pd.notna(day_std) and np.isfinite(day_std):
                daily_stds.append(float(day_std))
        if len(clean) < 2:
            continue
        feature_rank = clean[feature].rank(method="average")
        label_rank = clean[label_column].rank(method="average")
        if feature_rank.nunique() > 1 and label_rank.nunique() > 1:
            rank_ic = feature_rank.corr(label_rank)
            if pd.notna(rank_ic) and np.isfinite(rank_ic):
                rank_ics.append(float(rank_ic))
        ranked = clean.sort_values(feature, ascending=False)
        bucket_size = max(1, min(int(top_k), len(ranked) // 2 if len(ranked) >= 2 else 1))
        top = ranked.head(bucket_size)
        bottom = ranked.tail(bucket_size)
        spreads.append(float(top[label_column].mean() - bottom[label_column].mean()))
        hit_rates.append(float((top[label_column] > 0).mean()))
        _accumulate_group_returns(ranked, label_column, group_returns, n_groups)

    return {
        "fold_id": fold_id,
        "coverage": float(valid_count / total_count) if total_count else 0.0,
        "valid_count": valid_count,
        "row_count": total_count,
        "mean_cross_sectional_std": float(np.mean(daily_stds)) if daily_stds else 0.0,
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "rank_ic_by_date_count": len(rank_ics),
        "top_bucket_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "top_k_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "group_monotonicity": _compute_group_monotonicity(group_returns, n_groups),
    }


def _accumulate_group_returns(
    ranked: pd.DataFrame,
    label_column: str,
    group_returns: dict[int, list[float]],
    n_groups: int,
) -> None:
    row_count = len(ranked)
    if row_count < 2:
        return
    group_count = min(n_groups, row_count)
    for group in range(group_count):
        start = int(row_count * group / group_count)
        end = int(row_count * (group + 1) / group_count)
        if group == group_count - 1:
            end = row_count
        if end <= start:
            continue
        group_returns[group].append(float(ranked.iloc[start:end][label_column].mean()))


def _compute_group_monotonicity(group_returns: dict[int, list[float]], n_groups: int) -> float:
    means = [
        float(np.mean(group_returns.get(group, []))) if group_returns.get(group) else np.nan
        for group in range(n_groups)
    ]
    valid = [(idx, value) for idx, value in enumerate(means) if pd.notna(value)]
    if len(valid) < 3:
        return 0.0
    group_rank = pd.Series([len(valid) - idx for idx, _ in valid])
    return_rank = pd.Series([value for _, value in valid])
    corr = group_rank.corr(return_rank, method="spearman")
    return float(corr) if pd.notna(corr) and np.isfinite(corr) else 0.0


def _signed_unit(value: float) -> float:
    if value > 0:
        return min(float(value) * 100.0, 1.0)
    if value < 0:
        return max(float(value) * 100.0, -1.0)
    return 0.0


def _importance_by_feature(importance: dict[str, Any]) -> dict[str, float]:
    aggregate = importance.get("aggregate") or {}
    gain = aggregate.get("gain") or []
    return {
        str(item.get("feature")): _as_float(item.get("normalized_importance"))
        for item in gain
        if item.get("feature") is not None
    }


def _aggregate_feature_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        return {
            "coverage": 0.0,
            "valid_count": 0,
            "mean_cross_sectional_std": 0.0,
            "rank_ic_mean": 0.0,
            "top_bucket_spread_mean": 0.0,
            "top_k_hit_rate": 0.0,
            "group_monotonicity": 0.0,
            "fold_summary": _fold_summary([]),
            "folds": [],
        }
    total_rows = sum(int(metric.get("row_count", 0)) for metric in fold_metrics)
    valid_count = sum(int(metric.get("valid_count", 0)) for metric in fold_metrics)
    return {
        "coverage": float(valid_count / total_rows) if total_rows else 0.0,
        "valid_count": int(valid_count),
        "mean_cross_sectional_std": _weighted_mean(fold_metrics, "mean_cross_sectional_std", "valid_count"),
        "rank_ic_mean": _mean(fold_metrics, "rank_ic_mean"),
        "top_bucket_spread_mean": _mean(fold_metrics, "top_bucket_spread_mean"),
        "top_k_hit_rate": _mean(fold_metrics, "top_k_hit_rate"),
        "group_monotonicity": _mean(fold_metrics, "group_monotonicity"),
        "fold_summary": _fold_summary([_as_float(metric.get("rank_ic_mean")) for metric in fold_metrics]),
        "folds": fold_metrics,
    }


def _fold_summary(values: list[float]) -> dict[str, float | int]:
    clean = [value for value in values if np.isfinite(value)]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "worst": 0.0, "positive_ratio": 0.0, "count": 0}
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=0)) if len(clean) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0,
        "worst": float(np.min(clean)),
        "positive_ratio": float(np.mean([value > 0.0 for value in clean])),
        "count": len(clean),
    }


def _mean(metrics: list[dict[str, Any]], key: str) -> float:
    values = [_as_float(metric.get(key)) for metric in metrics]
    clean = [value for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else 0.0


def _weighted_mean(metrics: list[dict[str, Any]], key: str, weight_key: str) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for metric in metrics:
        value = _as_float(metric.get(key))
        weight = _as_float(metric.get(weight_key))
        if np.isfinite(value) and np.isfinite(weight) and weight > 0:
            weighted_sum += value * weight
            weight_sum += weight
    return float(weighted_sum / weight_sum) if weight_sum > 0 else 0.0


def _build_conflict_groups(edges: list[tuple[str, str]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    groups = _connected_components(edges)
    result: list[dict[str, Any]] = []
    for group in groups:
        representative = _select_representative(group, diagnostics)
        result.append(
            {
                "features": sorted(group),
                "representative": representative,
                "size": len(group),
            }
        )
    return sorted(result, key=lambda item: (-int(item["size"]), str(item["representative"])))


def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    groups: list[set[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.add(current)
            stack.extend(sorted(adjacency[current] - seen))
        if len(group) > 1:
            groups.append(group)
    return groups


def _select_representative(features: set[str], diagnostics: dict[str, Any]) -> str:
    return sorted(
        features,
        key=lambda feature: (
            -abs(_as_float(diagnostics.get(feature, {}).get("rank_ic_mean"))),
            -_as_float(diagnostics.get(feature, {}).get("coverage")),
            feature,
        ),
    )[0]


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else 0.0
    if isinstance(value, float):
        return value if np.isfinite(value) else 0.0
    if not isinstance(value, (str, bytes, bool)) and pd.isna(value):
        return None
    return value
