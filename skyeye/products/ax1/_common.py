"""AX1 shared utility functions.

Consolidates helpers previously duplicated across allocation, smoother,
executable, metrics, labels, and features/view modules.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def require_columns(
    frame: pd.DataFrame | None,
    columns: Sequence[str],
    *,
    entity: str = "DataFrame",
) -> None:
    """Validate that *frame* contains all required *columns*."""
    if frame is None:
        raise ValueError(f"{entity} must not be None")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{entity} missing required columns: {missing}")


def normalize_asset_type(value) -> str:
    """Normalize an asset type string to a canonical lowercase form.

    Returns ``"stock"`` for any null-like or unrecognized empty value.
    """
    if value is None:
        return "stock"
    try:
        if pd.isna(value):
            return "stock"
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().lower()
    if normalized in {"", "nan", "none", "null", "<na>"}:
        return "stock"
    if normalized in {"etf", "fund", "index_fund"}:
        return "etf"
    if normalized in {"stock", "cs", "equity", "common_stock"}:
        return "stock"
    return normalized or "stock"


def allocate_with_cap(
    scores: pd.Series, budget: float, max_weight: float
) -> dict[str, float]:
    """Allocate *budget* proportionally to *scores* with per-item cap."""
    scores = scores.astype(float).clip(lower=0.0)
    scores = scores[scores.index.notna()]
    if scores.empty or budget <= 0 or max_weight <= 0:
        return {}
    if scores.sum() <= 0:
        scores = pd.Series(1.0, index=scores.index)
    target_budget = min(float(budget), float(max_weight) * len(scores))
    remaining_scores = scores.copy()
    remaining_budget = target_budget
    allocated: dict[str, float] = {}
    while not remaining_scores.empty and remaining_budget > 1e-15:
        score_sum = float(remaining_scores.sum())
        if score_sum <= 0:
            raw = pd.Series(
                remaining_budget / len(remaining_scores),
                index=remaining_scores.index,
            )
        else:
            raw = remaining_scores / score_sum * remaining_budget
        over_cap = raw > float(max_weight) + 1e-15
        if not bool(over_cap.any()):
            allocated.update(
                {
                    str(order_book_id): float(weight)
                    for order_book_id, weight in raw.items()
                    if weight > 0
                }
            )
            break
        capped_ids = list(raw[over_cap].index)
        for order_book_id in capped_ids:
            allocated[str(order_book_id)] = float(max_weight)
        remaining_budget -= float(max_weight) * len(capped_ids)
        remaining_scores = remaining_scores.drop(index=capped_ids)
    return {
        order_book_id: weight
        for order_book_id, weight in allocated.items()
        if weight > 0
    }


def apply_group_cap(
    weights: dict[str, float],
    group_map: dict[str, str],
    budget: float,
    max_group_weight: float,
) -> dict[str, float]:
    """Re-scale *weights* so no group exceeds *max_group_weight*."""
    if not weights or max_group_weight <= 0:
        return weights
    weight_series = pd.Series(weights, dtype=float)
    groups = pd.Series(
        {
            order_book_id: str(group_map.get(order_book_id, "Unknown") or "Unknown")
            for order_book_id in weight_series.index
        }
    )
    group_scores = weight_series.groupby(groups).sum()
    feasible_budget = min(float(budget), float(weight_series.sum()))
    group_allocations = allocate_with_cap(
        group_scores, budget=feasible_budget, max_weight=max_group_weight
    )
    if not group_allocations:
        return {}
    adjusted = {}
    for group, group_weight in group_scores.items():
        target_group_weight = float(group_allocations.get(str(group), 0.0))
        if group_weight <= 0 or target_group_weight <= 0:
            continue
        scale = target_group_weight / float(group_weight)
        members = groups[groups == group].index
        for order_book_id in members:
            adjusted[str(order_book_id)] = float(weight_series[order_book_id]) * scale
    return {
        order_book_id: weight
        for order_book_id, weight in adjusted.items()
        if weight > 0
    }


def coerce_cost_config(cost_config):
    """Coerce *cost_config* (dict or CostConfig) into a CostConfig instance."""
    if cost_config is None:
        return None
    from skyeye.products.tx1.cost_layer import CostConfig

    if isinstance(cost_config, CostConfig):
        return cost_config
    if isinstance(cost_config, dict):
        if cost_config.get("enabled") is False:
            return None
        return CostConfig(
            commission_rate=float(cost_config.get("commission_rate", 0.0008)),
            stamp_tax_rate=float(cost_config.get("stamp_tax_rate", 0.0005)),
            slippage_bps=float(cost_config.get("slippage_bps", 5.0)),
            min_commission=float(cost_config.get("min_commission", 0.0)),
        )
    raise TypeError("cost_config must be a dict or CostConfig")
