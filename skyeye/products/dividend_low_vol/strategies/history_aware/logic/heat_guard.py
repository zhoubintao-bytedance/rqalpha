"""Heat-signal detection and cooldown guard."""


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
    premium_pct = max(
        value
        for value in (
            safe_feature_percentile(score, "premium_rate"),
            safe_feature_percentile(score, "premium_rate_ma20"),
        )
        if value is not None
    ) if any(
        value is not None
        for value in (
            safe_feature_percentile(score, "premium_rate"),
            safe_feature_percentile(score, "premium_rate_ma20"),
        )
    ) else None
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


def trigger_heat_cap(percentile_ref, hot_signals):
    hot_count = hot_signals["hot_count"]
    if percentile_ref >= 0.90 and hot_count >= 3:
        return 0.05
    if percentile_ref >= 0.80 and hot_count >= 2:
        return 0.10
    return None


def apply_heat_override(context, base_target, percentile_ref, hot_signals):
    trigger_cap = trigger_heat_cap(percentile_ref, hot_signals)
    if trigger_cap is not None:
        context.heat_cooldown_left = context.heat_cooldown_weeks
        context.heat_cooldown_cap = trigger_cap
        return {
            "target": min(base_target, trigger_cap),
            "mode": "triggered",
            "cap": trigger_cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    if (
        context.heat_cooldown_left > 0
        and context.heat_cooldown_cap is not None
        and percentile_ref >= context.heat_cooldown_min_percentile
    ):
        release_cap = min(0.25, context.heat_cooldown_cap + context.heat_cooldown_release_buffer)
        context.heat_cooldown_left -= 1
        if context.heat_cooldown_left == 0:
            context.heat_cooldown_cap = None
        return {
            "target": min(base_target, release_cap),
            "mode": "cooldown",
            "cap": release_cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    context.heat_cooldown_left = 0
    context.heat_cooldown_cap = None
    return {
        "target": base_target,
        "mode": "none",
        "cap": None,
        "cooldown_left": 0,
    }
