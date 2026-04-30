"""Lightweight OOS validation report for AX1 allocation parameters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


def build_parameter_validation_summary(
    config: dict[str, Any],
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a small deterministic candidate set without changing defaults."""
    frame = _merge_predictions_and_labels(predictions, labels)
    score_column = _select_score_column(frame)
    label_column = _select_label_column(frame)
    current = _current_params(config)
    lgbm_param_policy = build_lgbm_param_policy_summary(config)
    if frame.empty or score_column is None or label_column is None:
        warning_count = 1 + int(lgbm_param_policy.get("warning_count", 0))
        return {
            "schema_version": 1,
            "status": "insufficient_data",
            "current": current,
            "candidate_metrics": [],
            "best_candidate_id": None,
            "fragile": False,
            "warning_count": warning_count,
            "warnings": ["insufficient_oos_predictions_or_labels"],
            "lgbm_param_policy": lgbm_param_policy,
        }

    candidates = _candidate_params(current)
    metrics = [
        _evaluate_candidate(frame, candidate, score_column=score_column, label_column=label_column)
        for candidate in candidates
    ]
    best = max(metrics, key=lambda item: (item["mean_realized_return"], -item["mean_turnover"], item["candidate_id"]))
    current_metric = next(item for item in metrics if item["candidate_id"] == "current")
    warnings = []
    if best["candidate_id"] != "current" and best["mean_realized_return"] > current_metric["mean_realized_return"] + 1e-12:
        warnings.append("current_params_not_top_candidate")
    returns = [float(item["mean_realized_return"]) for item in metrics]
    if len(returns) > 1 and float(np.nanmax(returns) - np.nanmin(returns)) > abs(current_metric["mean_realized_return"]) + 1e-12:
        warnings.append("high_candidate_dispersion")
    policy_warning_count = int(lgbm_param_policy.get("warning_count", 0))
    return {
        "schema_version": 1,
        "status": "evaluated",
        "score_column": score_column,
        "label_column": label_column,
        "current": current,
        "candidate_metrics": metrics,
        "best_candidate_id": best["candidate_id"],
        "fragile": bool(warnings) or policy_warning_count > 0,
        "warning_count": int(len(warnings) + policy_warning_count),
        "warnings": warnings,
        "lgbm_param_policy": lgbm_param_policy,
    }


def build_lgbm_param_policy_summary(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model") or {})
    params = dict(model.get("params") or {})
    policies = dict(model.get("param_policy") or {})
    current: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    hard_blocks: list[dict[str, Any]] = []
    normalized_policies: dict[str, dict[str, Any]] = {}

    for name in ("min_child_samples", "learning_rate", "reg_lambda"):
        policy = dict(policies.get(name) or {})
        value = params.get(name)
        current[name] = value
        candidates = [item for item in policy.get("candidates", [])]
        warning_range = [float(item) for item in policy.get("warning_range", [])]
        hard_range = [float(item) for item in policy.get("hard_range", [])]
        normalized_policies[name] = {
            "default": policy.get("default"),
            "candidates": candidates,
            "warning_range": warning_range,
            "hard_range": hard_range,
            "reason": str(policy.get("reason", "")),
        }
        if value is None:
            hard_blocks.append(_param_issue(name, "lgbm_param_missing_from_policy", "LGBM parameter is missing"))
            continue
        numeric = float(value)
        if len(hard_range) == 2 and not (hard_range[0] <= numeric <= hard_range[1]):
            hard_blocks.append(
                _param_issue(
                    name,
                    "lgbm_param_outside_hard_range",
                    "LGBM parameter is outside declared hard range",
                    value=numeric,
                    range=hard_range,
                )
            )
        elif len(warning_range) == 2 and not (warning_range[0] <= numeric <= warning_range[1]):
            warnings.append(
                _param_issue(
                    name,
                    "lgbm_param_outside_warning_range",
                    "LGBM parameter deviates from declared warning range",
                    value=numeric,
                    range=warning_range,
                )
            )
    status = "blocked" if hard_blocks else "deviates_from_policy" if warnings else "within_policy"
    return {
        "schema_version": 1,
        "status": status,
        "current": current,
        "policies": normalized_policies,
        "hard_block_count": int(len(hard_blocks)),
        "warning_count": int(len(warnings)),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
    }


def _merge_predictions_and_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    if labels is None or labels.empty:
        return frame
    keys = [column for column in ("date", "order_book_id", "fold_id") if column in frame.columns and column in labels.columns]
    if len(keys) < 2:
        return frame
    left = frame.copy()
    right = labels.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    left["order_book_id"] = left["order_book_id"].astype(str)
    right["order_book_id"] = right["order_book_id"].astype(str)
    label_columns = [
        column
        for column in right.columns
        if column.startswith("label_net_return_") or column.startswith("label_return_")
    ]
    existing = [column for column in label_columns if column in left.columns]
    left = left.drop(columns=existing, errors="ignore")
    return left.merge(right[keys + label_columns].drop_duplicates(keys, keep="last"), on=keys, how="left")


def _current_params(config: dict[str, Any]) -> dict[str, Any]:
    constraints = dict(config.get("constraints") or {})
    allocation = dict(config.get("allocation") or {})
    return {
        "allocation_kind": str(allocation.get("kind", "opportunity_pool_optimizer")),
        "exposure_groups": deepcopy(allocation.get("exposure_groups") or {}),
        "min_allocatable_score": float(allocation.get("min_allocatable_score", 0.0)),
        "max_single_weight": float(constraints.get("max_single_weight", 1.0)),
        "max_industry_weight": (
            float(constraints["max_industry_weight"])
            if constraints.get("max_industry_weight") is not None
            else None
        ),
        "max_position_count": int(constraints.get("max_position_count", 10)),
        "cash_buffer": float(constraints.get("cash_buffer", 0.0)),
    }


def _candidate_params(current: dict[str, Any]) -> list[dict[str, Any]]:
    tighter = deepcopy(current)
    tighter["candidate_id"] = "tighter_caps"
    tighter["max_single_weight"] = max(0.01, float(current["max_single_weight"]) * 0.80)
    if tighter.get("max_industry_weight") is not None:
        tighter["max_industry_weight"] = max(0.01, float(current["max_industry_weight"]) * 0.80)

    looser = deepcopy(current)
    looser["candidate_id"] = "looser_caps"
    looser["max_single_weight"] = min(1.0, float(current["max_single_weight"]) * 1.20)
    if looser.get("max_industry_weight") is not None:
        looser["max_industry_weight"] = min(1.0, float(current["max_industry_weight"]) * 1.20)

    base = deepcopy(current)
    base["candidate_id"] = "current"
    return [base, tighter, looser]


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    *,
    score_column: str,
    label_column: str,
) -> dict[str, Any]:
    working = frame.dropna(subset=["date", "order_book_id", score_column, label_column]).copy()
    if working.empty:
        return _empty_metric(candidate)
    working["date"] = pd.to_datetime(working["date"])
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    realized_returns = []
    for date, day_df in working.groupby("date", sort=True):
        day_weights = _candidate_day_weights(day_df, candidate, score_column=score_column)
        if not day_weights:
            continue
        weights_by_date[pd.Timestamp(date)] = day_weights
        label_by_id = day_df.set_index("order_book_id")[label_column].astype(float).to_dict()
        realized_returns.append(
            sum(float(weight) * float(label_by_id.get(order_book_id, 0.0)) for order_book_id, weight in day_weights.items())
        )
    turnovers = _turnovers(weights_by_date)
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "max_single_weight": float(candidate["max_single_weight"]),
        "max_industry_weight": (
            float(candidate["max_industry_weight"])
            if candidate.get("max_industry_weight") is not None
            else None
        ),
        "mean_realized_return": float(np.mean(realized_returns)) if realized_returns else 0.0,
        "hit_rate": float(np.mean([value > 0.0 for value in realized_returns])) if realized_returns else 0.0,
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "date_count": int(len(realized_returns)),
    }


def _candidate_day_weights(day_df: pd.DataFrame, candidate: dict[str, Any], *, score_column: str) -> dict[str, float]:
    max_count = max(1, int(candidate.get("max_position_count", 10)))
    max_single = max(0.0, float(candidate.get("max_single_weight", 1.0)))
    gross = max(0.0, 1.0 - float(candidate.get("cash_buffer", 0.0)))
    ranked = day_df.sort_values(score_column, ascending=False).head(max_count).copy()
    if ranked.empty or gross <= 0:
        return {}
    scores = pd.to_numeric(ranked[score_column], errors="coerce").fillna(0.0)
    scores = scores - min(0.0, float(scores.min()))
    if float(scores.sum()) <= 0.0:
        raw_weights = pd.Series(gross / len(ranked), index=ranked.index)
    else:
        raw_weights = scores / float(scores.sum()) * gross
    capped = raw_weights.clip(upper=max_single)
    if float(capped.sum()) > gross + 1e-12:
        capped = capped / float(capped.sum()) * gross
    return {
        str(order_book_id): float(weight)
        for order_book_id, weight in zip(ranked["order_book_id"].astype(str), capped)
        if float(weight) > 0.0
    }


def _turnovers(weights_by_date: dict[pd.Timestamp, dict[str, float]]) -> list[float]:
    previous: dict[str, float] | None = None
    turnovers = []
    for date in sorted(weights_by_date):
        current = weights_by_date[date]
        if previous is not None:
            ids = set(previous) | set(current)
            turnovers.append(sum(abs(current.get(order_book_id, 0.0) - previous.get(order_book_id, 0.0)) for order_book_id in ids) / 2.0)
        previous = current
    return turnovers


def _empty_metric(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "max_single_weight": float(candidate["max_single_weight"]),
        "max_industry_weight": candidate.get("max_industry_weight"),
        "mean_realized_return": 0.0,
        "hit_rate": 0.0,
        "mean_turnover": 0.0,
        "date_count": 0,
    }


def _select_score_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "adjusted_expected_return",
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
    ):
        if column in frame.columns:
            return column
    return None


def _select_label_column(frame: pd.DataFrame) -> str | None:
    for column in ("label_net_return_20d", "label_net_return_10d", "label_return_20d", "label_return_10d"):
        if column in frame.columns:
            return column
    columns = sorted(
        column
        for column in frame.columns
        if column.startswith("label_net_return_") or column.startswith("label_return_")
    )
    return columns[0] if columns else None


def _param_issue(name: str, reason_code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "parameter": str(name),
        "reason_code": str(reason_code),
        "message": str(message),
    }
    payload.update(extra)
    return payload
