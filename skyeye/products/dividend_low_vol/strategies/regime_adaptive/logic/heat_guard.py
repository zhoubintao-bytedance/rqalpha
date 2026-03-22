"""Heat detection that only caps the tactical layer."""

from __future__ import annotations


def safe_feature_percentile(score, feature_name):
    features = score.get("features") or {}
    feature = features.get(feature_name) or {}
    percentile = feature.get("percentile")
    if percentile is None:
        return None
    return float(percentile)


def safe_feature_raw(score, feature_name):
    features = score.get("features") or {}
    feature = features.get(feature_name) or {}
    raw = feature.get("raw")
    if raw is None:
        return None
    return float(raw)


def compute_hot_signals(score):
    price_pct = safe_feature_percentile(score, "price_percentile")
    premium_values = (
        safe_feature_percentile(score, "premium_rate"),
        safe_feature_percentile(score, "premium_rate_ma20"),
    )
    valid_premium_values = [value for value in premium_values if value is not None]
    premium_pct = max(valid_premium_values) if valid_premium_values else None
    rsi_raw = safe_feature_raw(score, "rsi_20")
    rsi_pct = safe_feature_percentile(score, "rsi_20")

    hot_price = price_pct is not None and price_pct >= 0.85
    hot_premium = premium_pct is not None and premium_pct >= 0.85
    hot_rsi = (rsi_raw is not None and rsi_raw >= 70.0) or (rsi_pct is not None and rsi_pct >= 0.90)

    return {
        "hot_price": hot_price,
        "hot_premium": hot_premium,
        "hot_rsi": hot_rsi,
        "hot_count": int(hot_price) + int(hot_premium) + int(hot_rsi),
    }


def _trigger_heat_cap(
    percentile_ref,
    hot_count,
    trend_strength,
    drawdown,
    heat_trigger_percentile,
    heat_extreme_percentile,
    heat_constructive_trend_threshold,
    heat_strong_trend_threshold,
    heat_shallow_drawdown,
    heat_cap_hot_weak,
    heat_cap_hot_constructive,
    heat_cap_hot_strong,
    heat_cap_extreme_weak,
    heat_cap_extreme_constructive,
    heat_cap_extreme_strong,
):
    shallow_pullback = float(drawdown) >= float(heat_shallow_drawdown)
    strong_trend = float(trend_strength) >= float(heat_strong_trend_threshold) and shallow_pullback
    constructive_trend = float(trend_strength) >= float(heat_constructive_trend_threshold) and shallow_pullback

    if float(percentile_ref) >= float(heat_extreme_percentile) and int(hot_count) >= 3:
        if strong_trend:
            return float(heat_cap_extreme_strong), "extreme_strong"
        if constructive_trend:
            return float(heat_cap_extreme_constructive), "extreme_constructive"
        return float(heat_cap_extreme_weak), "extreme_weak"

    if float(percentile_ref) >= float(heat_trigger_percentile) and int(hot_count) >= 2:
        if strong_trend:
            return float(heat_cap_hot_strong), "hot_strong"
        if constructive_trend:
            return float(heat_cap_hot_constructive), "hot_constructive"
        return float(heat_cap_hot_weak), "hot_weak"

    return None, "none"


def apply_tactical_heat_ceiling(
    context,
    base_target,
    support_floor,
    percentile_ref,
    hot_signals,
    trend_strength,
    drawdown,
    heat_trigger_percentile,
    heat_extreme_percentile,
    heat_cooldown_min_percentile,
    heat_cooldown_release_buffer,
    heat_constructive_trend_threshold,
    heat_strong_trend_threshold,
    heat_shallow_drawdown,
    heat_cap_hot_weak,
    heat_cap_hot_constructive,
    heat_cap_hot_strong,
    heat_cap_extreme_weak,
    heat_cap_extreme_constructive,
    heat_cap_extreme_strong,
):
    trigger_cap, trigger_mode = _trigger_heat_cap(
        percentile_ref=percentile_ref,
        hot_count=hot_signals["hot_count"],
        trend_strength=trend_strength,
        drawdown=drawdown,
        heat_trigger_percentile=heat_trigger_percentile,
        heat_extreme_percentile=heat_extreme_percentile,
        heat_constructive_trend_threshold=heat_constructive_trend_threshold,
        heat_strong_trend_threshold=heat_strong_trend_threshold,
        heat_shallow_drawdown=heat_shallow_drawdown,
        heat_cap_hot_weak=heat_cap_hot_weak,
        heat_cap_hot_constructive=heat_cap_hot_constructive,
        heat_cap_hot_strong=heat_cap_hot_strong,
        heat_cap_extreme_weak=heat_cap_extreme_weak,
        heat_cap_extreme_constructive=heat_cap_extreme_constructive,
        heat_cap_extreme_strong=heat_cap_extreme_strong,
    )
    if trigger_cap is not None:
        cap = max(float(support_floor), float(trigger_cap))
        context.heat_cooldown_left = int(context.heat_cooldown_weeks)
        context.heat_cooldown_cap = cap
        return {
            "target": min(float(base_target), cap),
            "mode": trigger_mode,
            "cap": cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    if (
        context.heat_cooldown_left > 0
        and context.heat_cooldown_cap is not None
        and float(percentile_ref) >= float(heat_cooldown_min_percentile)
    ):
        release_cap = max(float(support_floor), float(context.heat_cooldown_cap) + float(heat_cooldown_release_buffer))
        context.heat_cooldown_left -= 1
        if context.heat_cooldown_left == 0:
            context.heat_cooldown_cap = None
        return {
            "target": min(float(base_target), release_cap),
            "mode": "cooldown",
            "cap": release_cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    context.heat_cooldown_left = 0
    context.heat_cooldown_cap = None
    return {
        "target": float(base_target),
        "mode": "none",
        "cap": None,
        "cooldown_left": 0,
    }
