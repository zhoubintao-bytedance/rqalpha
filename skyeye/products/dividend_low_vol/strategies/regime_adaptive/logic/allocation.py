"""Target-allocation helpers for the regime-adaptive strategy."""

from __future__ import annotations


def _clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def _interpolate(value, anchors):
    x = _clamp(value, 0.0, 1.0)
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        if x <= right_x:
            if right_x == left_x:
                return float(right_y)
            ratio = (x - left_x) / float(right_x - left_x)
            return float(left_y + ratio * (right_y - left_y))
    return float(anchors[-1][1])


def valuation_target_from_percentile(percentile, max_target_percent):
    max_target = float(max_target_percent)
    scale = max_target / 0.90
    anchors = (
        (0.00, min(max_target, 0.90 * scale)),
        (0.10, min(max_target, 0.90 * scale)),
        (0.20, min(max_target, 0.80 * scale)),
        (0.40, min(max_target, 0.70 * scale)),
        (0.52, min(max_target, 0.55 * scale)),
        (0.68, min(max_target, 0.55 * scale)),
        (0.80, min(max_target, 0.40 * scale)),
        (0.90, min(max_target, 0.30 * scale)),
        (1.00, min(max_target, 0.25 * scale)),
    )
    return _interpolate(percentile, anchors)


def trend_support_target(
    percentile_ref,
    trend_strength,
    drawdown,
    hot_count,
    trend_support_threshold,
    trend_support_min_percentile,
    trend_support_max_percentile,
    trend_support_shallow_drawdown,
    trend_support_max_hot_count,
    trend_support_min_target,
    trend_support_max_target,
    trend_support_extreme_percentile,
    trend_support_extreme_max_target,
):
    if (
        float(percentile_ref) < float(trend_support_min_percentile)
        or float(percentile_ref) > float(trend_support_max_percentile)
        or float(trend_strength) < float(trend_support_threshold)
        or float(drawdown) < float(trend_support_shallow_drawdown)
        or int(hot_count) > int(trend_support_max_hot_count)
    ):
        return {
            "target": 0.0,
            "mode": "none",
        }

    max_target = float(trend_support_max_target)
    if float(percentile_ref) >= float(trend_support_extreme_percentile):
        max_target = min(max_target, float(trend_support_extreme_max_target))

    lower = float(trend_support_threshold)
    upper = max(lower + 0.10, 0.90)
    if float(trend_strength) >= upper:
        target = max_target
    else:
        ratio = (float(trend_strength) - lower) / float(upper - lower)
        ratio = _clamp(ratio, 0.0, 1.0)
        target = float(trend_support_min_target) + ratio * (max_target - float(trend_support_min_target))

    return {
        "target": target,
        "mode": "trend_support",
    }


def combine_targets(
    valuation_target,
    support_target,
):
    target = float(valuation_target)
    mode = "valuation"
    active_floor = 0.0

    if float(support_target) > target:
        target = float(support_target)
        active_floor = float(support_target)
        mode = "trend_support"

    return {
        "target": target,
        "mode": mode,
        "active_floor": active_floor,
    }


def step_limit(
    current_target,
    desired_target,
    trend_strength,
    default_raise_step,
    default_cut_step,
    initial_step_limit,
    strong_trend_raise_step,
    weak_trend_raise_step,
    breakdown_cut_step,
    strong_trend_cut_step,
):
    current = float(current_target)
    desired = float(desired_target)
    if desired > current:
        if current < 0.01 and desired >= 0.40:
            return max(float(default_raise_step), min(float(initial_step_limit), desired))
        if float(trend_strength) >= 0.75:
            return max(float(default_raise_step), float(strong_trend_raise_step))
        if float(trend_strength) <= 0.35:
            return min(float(default_raise_step), float(weak_trend_raise_step))
        return float(default_raise_step)

    if desired < current:
        if float(trend_strength) <= 0.30:
            return max(float(default_cut_step), float(breakdown_cut_step))
        if float(trend_strength) >= 0.75:
            return min(float(default_cut_step), float(strong_trend_cut_step))
        return float(default_cut_step)

    return 0.0


def bounded_target(current_target, desired_target, max_step):
    current = float(current_target)
    desired = float(desired_target)
    delta = desired - current
    if delta > max_step:
        return current + float(max_step)
    if delta < -max_step:
        return current - float(max_step)
    return desired
