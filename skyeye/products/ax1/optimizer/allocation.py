"""AX1 opportunity-pool optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from skyeye.products.ax1._common import allocate_with_cap, apply_group_cap, require_columns

_logger = logging.getLogger(__name__)


_REQUIRED_COLUMNS = ["date", "order_book_id", "confidence"]
_METADATA_COLUMNS = [
    "universe_layer",
    "exposure_group",
    "asset_type",
    "industry",
    "expected_relative_net_return_10d",
    "confidence",
    "view_score_10d",
    "rebalance_allowed",
]


@dataclass(frozen=True)
class OpportunityPoolOptimizer:
    """Allocate all enabled assets from one alpha-ranked opportunity pool."""

    def optimize(self, views: pd.DataFrame, *, constraints=None, allocation_config=None, risk_model=None) -> pd.DataFrame:
        if views is None:
            return _empty_targets()
        require_columns(views, _REQUIRED_COLUMNS, entity="views")
        if views.empty:
            return _empty_targets()

        optimizer_constraints = _normalize_constraints(constraints)
        config = allocation_config or {}
        frame = views.dropna(subset=["date", "order_book_id"]).copy()
        if frame.empty:
            return _empty_targets()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame["exposure_group"] = _exposure_group(frame, config)
        frame["_raw_score"] = _raw_score(frame, config)
        frame["_score"] = _allocatable_score(frame, config)
        frame["_score"] = _apply_score_multipliers(frame, config)
        frame["_score"] = _apply_risk_penalty(frame, risk_model)

        rows: list[dict[str, Any]] = []
        for date, day_df in frame.groupby("date", sort=True):
            day_config = _allocation_config_for_date(config, date)
            budget = _available_budget(day_config, optimizer_constraints)
            day_rows = _allocate_day(
                day_df,
                date,
                budget,
                optimizer_constraints,
                day_config,
                _risk_model_for_date(risk_model, date),
            )
            rows.extend(day_rows)

        return _targets_frame(rows)


def _empty_targets() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "order_book_id", "intended_weight", "target_weight", "component"])


def _normalize_constraints(constraints) -> dict[str, float | int]:
    raw = constraints or {}
    target_gross_exposure = float(raw.get("target_gross_exposure", 1.0))
    cash_buffer = float(raw.get("cash_buffer", 0.0))
    max_single_weight = float(raw.get("max_single_weight", 1.0))
    max_industry_weight = raw.get("max_industry_weight")
    max_industry_weight = float(max_industry_weight) if max_industry_weight is not None else None
    max_portfolio_volatility = raw.get("max_portfolio_volatility")
    max_portfolio_volatility = float(max_portfolio_volatility) if max_portfolio_volatility is not None else None
    min_position_count = int(raw.get("min_position_count", 1))
    max_position_count = int(raw.get("max_position_count", max(min_position_count, 1_000_000)))
    if target_gross_exposure < 0 or cash_buffer < 0:
        raise ValueError("target_gross_exposure and cash_buffer must be non-negative")
    if max_single_weight <= 0:
        raise ValueError("max_single_weight must be positive")
    if max_industry_weight is not None and max_industry_weight <= 0:
        raise ValueError("max_industry_weight must be positive")
    if max_portfolio_volatility is not None and max_portfolio_volatility <= 0:
        raise ValueError("max_portfolio_volatility must be positive")
    if min_position_count < 1 or max_position_count < 1:
        raise ValueError("position counts must be positive")
    if min_position_count > max_position_count:
        raise ValueError("min_position_count cannot exceed max_position_count")
    return {
        "target_gross_exposure": target_gross_exposure,
        "cash_buffer": cash_buffer,
        "max_single_weight": max_single_weight,
        "max_industry_weight": max_industry_weight,
        "min_position_count": min_position_count,
        "max_position_count": max_position_count,
        "max_portfolio_volatility": max_portfolio_volatility,
    }


def _is_date_keyed_config(config: dict) -> bool:
    if not isinstance(config, dict) or not config:
        return False
    allocation_keys = {
        "kind",
        "score_column",
        "exposure_groups",
        "layer_exposure_groups",
        "cash_buffer",
        "configured_cash_buffer",
        "execution_drift_buffer",
        "target_gross_exposure",
        "allow_gross_underfill",
        "min_allocatable_score",
        "cash_fallback",
    }
    if any(str(key) in allocation_keys for key in config):
        return False
    return all(isinstance(value, dict) for value in config.values())


def _allocation_config_for_date(config: dict, date) -> dict:
    if not _is_date_keyed_config(config):
        return config
    if not config:
        return {}
    target_date = pd.Timestamp(date)
    by_date: dict[pd.Timestamp, dict] = {}
    for key, value in config.items():
        try:
            by_date[pd.Timestamp(key)] = dict(value or {})
        except (TypeError, ValueError):
            continue
    if target_date in by_date:
        return by_date[target_date]
    eligible_dates = [key for key in by_date if key <= target_date]
    if not eligible_dates:
        return {}
    return by_date[max(eligible_dates)]


def _available_budget(config: dict, constraints: dict[str, float | int]) -> float:
    cash_buffer = float(config.get("cash_buffer", constraints["cash_buffer"]))
    target_gross_exposure = float(config.get("target_gross_exposure", constraints["target_gross_exposure"]))
    if cash_buffer < 0 or target_gross_exposure < 0:
        raise ValueError("target_gross_exposure and cash_buffer must be non-negative")
    return max(0.0, target_gross_exposure - cash_buffer)


def _exposure_group(frame: pd.DataFrame, config: dict) -> pd.Series:
    if "exposure_group" in frame.columns:
        existing = frame["exposure_group"].fillna("").astype(str)
    else:
        existing = pd.Series("", index=frame.index, dtype=str)
    layer_map = {str(key): str(value) for key, value in (config.get("layer_exposure_groups") or {}).items()}
    layer = frame["universe_layer"].astype(str) if "universe_layer" in frame.columns else pd.Series("", index=frame.index)
    mapped = layer.map(layer_map).fillna("")
    result = existing.where(existing.str.len() > 0, mapped)
    result = result.where(result.str.len() > 0, layer)
    return result.fillna("unknown").astype(str)


def _raw_score(frame: pd.DataFrame, config: dict) -> pd.Series:
    score_column = str(config.get("score_column", "expected_relative_net_return_10d"))
    if score_column in frame.columns:
        return pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0)
    if "view_score_10d" in frame.columns:
        return pd.to_numeric(frame["view_score_10d"], errors="coerce").fillna(0.0)
    for column in (
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
        "expected_relative_net_return_5d",
        "adjusted_expected_return",
    ):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    raise ValueError("views missing opportunity score column")


def _allocatable_score(frame: pd.DataFrame, config: dict) -> pd.Series:
    min_score = float(config.get("min_allocatable_score", 0.0))
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(1.0).clip(lower=0.0)
    raw_score = pd.to_numeric(frame["_raw_score"], errors="coerce").fillna(0.0)
    score = raw_score.where(raw_score > min_score, 0.0)
    return score.clip(lower=0.0) * confidence


def _apply_score_multipliers(frame: pd.DataFrame, config: dict) -> pd.Series:
    score = pd.to_numeric(frame["_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    exposure_groups = config.get("exposure_groups") if isinstance(config.get("exposure_groups"), dict) else {}
    if not exposure_groups:
        return score
    multipliers = {
        str(group): float((payload or {}).get("score_multiplier", 1.0))
        for group, payload in exposure_groups.items()
    }
    group_multiplier = frame["exposure_group"].astype(str).map(multipliers).fillna(1.0).astype(float)
    return score * group_multiplier.clip(lower=0.0)


def _apply_risk_penalty(frame: pd.DataFrame, risk_model) -> pd.Series:
    score = pd.to_numeric(frame["_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if risk_model is None:
        return score
    adjusted = score.copy()
    if isinstance(risk_model, dict):
        for date, group in frame.groupby("date", sort=False):
            risk_penalty = _risk_penalty_from_model(
                _risk_model_for_date(risk_model, date),
                order_book_ids=group["order_book_id"].astype(str).tolist(),
            )
            if not risk_penalty:
                continue
            risk = group["order_book_id"].astype(str).map(risk_penalty).fillna(0.0).astype(float).clip(lower=0.0)
            adjusted.loc[group.index] = score.loc[group.index] * (1.0 / (1.0 + risk))
        return adjusted
    risk_penalty = _risk_penalty_from_model(risk_model, order_book_ids=frame["order_book_id"].astype(str).tolist())
    if not risk_penalty:
        return score
    risk = frame["order_book_id"].astype(str).map(risk_penalty).fillna(0.0).astype(float).clip(lower=0.0)
    return score * (1.0 / (1.0 + risk))


def _allocate_day(
    day_df: pd.DataFrame,
    date,
    budget: float,
    constraints: dict[str, float | int],
    config: dict,
    risk_model=None,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    candidates = day_df[day_df["_score"] > 0].sort_values("_score", ascending=False, kind="mergesort")
    if candidates.empty:
        return []
    candidates = candidates.head(int(constraints["max_position_count"]))
    scores = candidates.set_index("order_book_id")["_score"].astype(float)
    weights = allocate_with_cap(scores, budget=budget, max_weight=float(constraints["max_single_weight"]))
    weights = _apply_exposure_caps(weights, candidates, config, budget)
    max_industry_weight = constraints["max_industry_weight"]
    if max_industry_weight is not None and "industry" in candidates.columns:
        industry_map = candidates.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")["industry"]
        # Skip max_industry_weight when all ETFs are 'Unknown' (ETF industry classification is ambiguous)
        unique_industries = set(industry_map.dropna().astype(str).unique())
        if unique_industries == {"Unknown"}:
            _logger.warning(
                "max_industry_weight constraint skipped: all instruments have industry='Unknown'. "
                "ETF industry classification is ambiguous; use exposure_group constraints for sector risk control."
            )
        else:
            weights = apply_group_cap(weights, industry_map.to_dict(), budget, float(max_industry_weight))
    max_portfolio_volatility = constraints.get("max_portfolio_volatility")
    if max_portfolio_volatility is not None and risk_model is not None:
        weights = _apply_volatility_cap(weights, risk_model, float(max_portfolio_volatility), trading_days_per_year=int(config.get("trading_days_per_year", 244)))
    if not weights:
        return []

    row_by_id = candidates.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
    rows: list[dict[str, Any]] = []
    for order_book_id, target_weight in weights.items():
        if target_weight <= 0:
            continue
        row = {
            "date": date,
            "order_book_id": order_book_id,
            "intended_weight": target_weight,
            "target_weight": target_weight,
            "component": "opportunity_pool",
        }
        for column in _METADATA_COLUMNS:
            if column in row_by_id.columns:
                value = row_by_id.loc[order_book_id, column]
                if pd.notna(value):
                    row[column] = value
        rows.append(row)
    return rows


def _apply_exposure_caps(weights: dict[str, float], candidates: pd.DataFrame, config: dict, budget: float) -> dict[str, float]:
    exposure_groups = config.get("exposure_groups") if isinstance(config.get("exposure_groups"), dict) else {}
    max_weights = {
        str(group): float((payload or {}).get("max_weight", 1.0))
        for group, payload in exposure_groups.items()
        if (payload or {}).get("max_weight") is not None
    }
    if not weights or not max_weights:
        return weights
    weight_series = pd.Series(weights, dtype=float)
    exposure_map = candidates.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")["exposure_group"]
    groups = pd.Series(
        {order_book_id: str(exposure_map.get(order_book_id, "unknown") or "unknown") for order_book_id in weight_series.index}
    )
    adjusted: dict[str, float] = {}
    for group, group_weight in weight_series.groupby(groups).sum().items():
        max_weight = max_weights.get(str(group))
        target_group_weight = float(group_weight)
        if max_weight is not None:
            target_group_weight = min(target_group_weight, float(max_weight))
        if target_group_weight <= 0 or float(group_weight) <= 0:
            continue
        scale = target_group_weight / float(group_weight)
        for order_book_id in groups[groups == group].index:
            adjusted[str(order_book_id)] = float(weight_series[order_book_id]) * scale
    return {order_book_id: weight for order_book_id, weight in adjusted.items() if weight > 0}


def _targets_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_targets()
    columns = ["date", "order_book_id", "intended_weight", "target_weight", "component"]
    for column in _METADATA_COLUMNS:
        if any(column in row for row in rows):
            columns.append(column)
    return pd.DataFrame(rows, columns=columns)


def _risk_penalty_from_model(risk_model, order_book_ids=None) -> dict[str, float]:
    return _risk_context_from_model(risk_model, order_book_ids=order_book_ids)["total"]


def _risk_context_from_model(risk_model, order_book_ids=None) -> dict[str, dict[str, float]]:
    if risk_model is None or not hasattr(risk_model, "get_covariance_matrix"):
        return {"variance": {}, "correlation": {}, "total": {}}
    covariance = risk_model.get_covariance_matrix()
    if covariance is None or covariance.empty:
        return {"variance": {}, "correlation": {}, "total": {}}
    covariance = covariance.copy()
    covariance.index = covariance.index.map(str)
    covariance.columns = covariance.columns.map(str)
    if order_book_ids is not None:
        requested = [str(order_book_id) for order_book_id in dict.fromkeys(order_book_ids)]
        available = [order_book_id for order_book_id in requested if order_book_id in covariance.index and order_book_id in covariance.columns]
        covariance = covariance.reindex(index=available, columns=available).fillna(0.0)
    if covariance.empty:
        return {"variance": {}, "correlation": {}, "total": {}}
    variance = {}
    for order_book_id in covariance.index:
        try:
            variance[str(order_book_id)] = max(0.0, float(covariance.loc[order_book_id, order_book_id]))
        except (KeyError, TypeError, ValueError):
            continue
    diagonal = pd.Series(variance, dtype=float).reindex(covariance.index).fillna(0.0)
    std = diagonal.pow(0.5).replace(0.0, pd.NA)
    correlation = covariance.div(std, axis=0).div(std, axis=1)
    correlation = correlation.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for order_book_id in correlation.index:
        if order_book_id in correlation.columns:
            correlation.loc[order_book_id, order_book_id] = 0.0
    denominator = max(len(correlation.columns) - 1, 1)
    # Positive correlations increase concentration risk; negative correlations
    # (diversification) reduce it.  Avoid clipping to zero so that assets
    # with net-negative average off-diagonal correlation earn a discount.
    raw_avg = correlation.sum(axis=1) / float(denominator)
    correlation_penalty = {
        str(order_book_id): float(max(0.0, value)) for order_book_id, value in raw_avg.items()
    }
    correlation_discount = {
        str(order_book_id): float(min(0.0, value)) for order_book_id, value in raw_avg.items()
    }
    max_var = max(variance.values()) if variance else 0.0
    total = {
        order_book_id: (
            float(variance.get(order_book_id, 0.0)) / max_var if max_var > 0 else 0.0
        ) + float(correlation_penalty.get(order_book_id, 0.0))
        + float(correlation_discount.get(order_book_id, 0.0))
        for order_book_id in set(variance) | set(correlation_penalty) | set(correlation_discount)
    }
    return {"variance": variance, "correlation": correlation_penalty, "correlation_discount": correlation_discount, "total": total}


def _risk_model_for_date(risk_model, date):
    if not isinstance(risk_model, Mapping):
        return risk_model
    if not risk_model:
        return None
    target_date = pd.Timestamp(date)
    by_date = {}
    for key, value in risk_model.items():
        try:
            by_date[pd.Timestamp(key)] = value
        except (TypeError, ValueError):
            continue
    if target_date in by_date:
        return by_date[target_date]
    eligible_dates = [key for key in by_date if key <= target_date]
    if not eligible_dates:
        return None
    return by_date[max(eligible_dates)]


def _compute_portfolio_volatility(
    weights: dict[str, float],
    covariance: pd.DataFrame,
    trading_days_per_year: int = 244,
) -> float:
    if not weights or covariance.empty:
        return 0.0
    asset_ids = [str(asset_id) for asset_id in weights if str(asset_id) in covariance.index]
    if not asset_ids:
        return 0.0
    weight_series = pd.Series(weights).reindex(asset_ids).fillna(0.0)
    cov_sub = covariance.reindex(index=asset_ids, columns=asset_ids).fillna(0.0)
    variance = float(weight_series.to_numpy(dtype=float).T @ cov_sub.to_numpy(dtype=float) @ weight_series.to_numpy(dtype=float))
    return float(np.sqrt(max(0.0, variance) * trading_days_per_year))


def _apply_volatility_cap(
    weights: dict[str, float],
    risk_model,
    max_portfolio_volatility: float | None,
    trading_days_per_year: int = 244,
) -> dict[str, float]:
    if not weights or max_portfolio_volatility is None or max_portfolio_volatility <= 0:
        return weights
    if risk_model is None or not hasattr(risk_model, "get_covariance_matrix"):
        return weights
    try:
        covariance = risk_model.get_covariance_matrix()
    except (ValueError, AttributeError):
        return weights
    if covariance is None or covariance.empty:
        return weights
    current_vol = _compute_portfolio_volatility(weights, covariance, trading_days_per_year)
    if current_vol <= float(max_portfolio_volatility) + 1e-9:
        return weights
    scale = float(max_portfolio_volatility) / max(current_vol, 1e-12)
    scale = max(0.0, min(1.0, scale))
    return {
        order_book_id: float(weight) * scale
        for order_book_id, weight in weights.items()
        if float(weight) * scale > 1e-15
    }
