"""Market regime detection driven by ETF price trend."""

from __future__ import annotations

import numpy as np


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _scaled(value, lower, upper):
    if upper <= lower:
        return _clamp01(value)
    return _clamp01((float(value) - lower) / float(upper - lower))


def _interpolate(value, anchors):
    x = _clamp01(value)
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        if x <= right_x:
            if right_x == left_x:
                return float(right_y)
            ratio = (x - left_x) / float(right_x - left_x)
            return float(left_y + ratio * (right_y - left_y))
    return float(anchors[-1][1])


def _average(prices, window):
    return float(np.mean(prices[-int(window):]))


def _window_return(prices, window):
    anchor_index = max(0, len(prices) - int(window) - 1)
    anchor_price = float(prices[anchor_index])
    if anchor_price <= 0:
        return 0.0
    return float(prices[-1] / anchor_price - 1.0)


def core_floor_from_trend_strength(trend_strength, min_floor, max_floor):
    span = max(0.0, float(max_floor) - float(min_floor))
    anchors = (
        (0.00, min_floor),
        (0.25, min_floor + span * 0.20),
        (0.45, min_floor + span * 0.40),
        (0.65, min_floor + span * 0.62),
        (0.82, min_floor + span * 0.82),
        (1.00, max_floor),
    )
    return _interpolate(trend_strength, anchors)


def regime_label(trend_strength):
    if trend_strength >= 0.78:
        return "bull"
    if trend_strength >= 0.58:
        return "constructive"
    if trend_strength >= 0.38:
        return "neutral"
    return "defensive"


def compute_market_state(
    close_history,
    short_window,
    mid_window,
    long_window,
    breakout_window,
    core_floor_min,
    core_floor_max,
):
    prices = np.asarray(close_history, dtype=float)
    prices = prices[np.isfinite(prices)]
    required = max(int(short_window), int(mid_window), int(long_window), int(breakout_window)) + 1
    if len(prices) < required:
        raise ValueError("close history {} < required {}".format(len(prices), required))

    close = float(prices[-1])
    ma_short = _average(prices, short_window)
    ma_mid = _average(prices, mid_window)
    ma_long = _average(prices, long_window)
    ret_short = _window_return(prices, short_window)
    ret_mid = _window_return(prices, mid_window)
    recent_high = float(np.max(prices[-int(breakout_window):]))
    drawdown = 0.0 if recent_high <= 0 else float(close / recent_high - 1.0)

    components = {
        "close_vs_ma_short": _scaled(close / ma_short - 1.0, -0.04, 0.05),
        "close_vs_ma_mid": _scaled(close / ma_mid - 1.0, -0.08, 0.10),
        "ma_short_vs_mid": _scaled(ma_short / ma_mid - 1.0, -0.03, 0.05),
        "ma_mid_vs_long": _scaled(ma_mid / ma_long - 1.0, -0.04, 0.06),
        "ret_short": _scaled(ret_short, -0.06, 0.08),
        "ret_mid": _scaled(ret_mid, -0.10, 0.15),
        "drawdown": _scaled(drawdown, -0.12, 0.00),
    }
    weights = {
        "close_vs_ma_short": 0.9,
        "close_vs_ma_mid": 1.2,
        "ma_short_vs_mid": 1.0,
        "ma_mid_vs_long": 1.2,
        "ret_short": 0.8,
        "ret_mid": 1.0,
        "drawdown": 1.0,
    }
    weight_sum = float(sum(weights.values()))
    trend_strength = sum(weights[name] * components[name] for name in components) / weight_sum
    core_floor = core_floor_from_trend_strength(
        trend_strength=trend_strength,
        min_floor=core_floor_min,
        max_floor=core_floor_max,
    )

    return {
        "close": close,
        "ma_short": ma_short,
        "ma_mid": ma_mid,
        "ma_long": ma_long,
        "ret_short": ret_short,
        "ret_mid": ret_mid,
        "drawdown": drawdown,
        "trend_strength": trend_strength,
        "core_floor": core_floor,
        "mode": regime_label(trend_strength),
        "components": components,
    }
