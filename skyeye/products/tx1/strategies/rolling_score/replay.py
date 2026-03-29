"""Pure helpers for replaying frozen TX1 target weights."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def sanitize_target_weights(
    target_weights: Mapping[str, float] | None,
    single_stock_cap: float | None = None,
) -> dict[str, float]:
    """Drop invalid weights and apply an optional hard cap."""
    sanitized = {}
    for order_book_id, raw_weight in (target_weights or {}).items():
        if not order_book_id:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight) or weight <= 0:
            continue
        if single_stock_cap is not None:
            weight = min(weight, float(single_stock_cap))
        if weight <= 0:
            continue
        sanitized[str(order_book_id)] = weight
    return sanitized


def compute_turnover_ratio(
    current_weights: Mapping[str, float] | None,
    target_weights: Mapping[str, float] | None,
) -> float:
    """Approximate one-way turnover using portfolio weight deltas."""
    union = set(current_weights or {}) | set(target_weights or {})
    if not union:
        return 0.0
    total_delta = 0.0
    for order_book_id in union:
        current_weight = float((current_weights or {}).get(order_book_id, 0.0) or 0.0)
        target_weight = float((target_weights or {}).get(order_book_id, 0.0) or 0.0)
        total_delta += abs(target_weight - current_weight)
    return 0.5 * total_delta


def smooth_target_weights(
    raw_weights: dict[str, float],
    ema_state: dict[str, float] | None,
    halflife: float,
    min_weight: float = 0.005,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply EMA smoothing to target weights.

    For each stock: new_ema = alpha * raw + (1-alpha) * prev_ema.
    New stocks (no prior state) use their raw weight directly.
    Stocks below *min_weight* after smoothing are removed.
    The returned weights are renormalized to sum to 1.0.

    Returns (smoothed_weights, new_ema_state).
    The ema_state stores pre-normalization values to avoid cumulative drift.
    """
    if halflife <= 0:
        return (dict(raw_weights), dict(raw_weights))

    prev = ema_state or {}
    alpha = 2.0 / (halflife + 1.0)

    all_stocks = set(raw_weights) | set(prev)
    new_state: dict[str, float] = {}

    for stock in all_stocks:
        raw = raw_weights.get(stock, 0.0)
        old = prev.get(stock)
        if old is None:
            # New stock — start at raw weight
            ema_val = raw
        else:
            ema_val = alpha * raw + (1.0 - alpha) * old

        if ema_val >= min_weight:
            new_state[stock] = ema_val

    if not new_state:
        return ({}, {})

    total = sum(new_state.values())
    normalized = {s: w / total for s, w in new_state.items()}
    return (normalized, dict(new_state))


def build_execution_universe(
    target_weights: Mapping[str, float] | None,
    current_holdings: Mapping[str, float] | Iterable[str] | None,
) -> list[str]:
    """Return a deterministic universe covering targets and current holdings."""
    order_book_ids = set(target_weights or {})
    if isinstance(current_holdings, Mapping):
        order_book_ids.update(current_holdings.keys())
    elif current_holdings is not None:
        order_book_ids.update(current_holdings)
    return sorted(order_book_ids)
