"""High-percentile reentry guard."""


def apply_high_reentry_guard(
    context,
    current_target,
    desired_target,
    percentile_ref,
    trend,
    hot_signals,
    heat_override,
):
    if heat_override["mode"] == "triggered":
        context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "triggered",
            "clear_count": context.high_reentry_clear_count,
        }

    if heat_override["mode"] == "cooldown":
        return {
            "target": desired_target,
            "mode": "cooldown",
            "clear_count": context.high_reentry_clear_count,
        }

    if desired_target <= current_target:
        if percentile_ref < context.high_reentry_guard_percentile:
            context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    if percentile_ref < context.high_reentry_guard_percentile:
        context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    confirmation_ready = hot_signals["hot_count"] <= 1 and trend <= context.high_reentry_confirm_trend
    if confirmation_ready:
        context.high_reentry_clear_count += 1
    else:
        context.high_reentry_clear_count = 0

    if context.high_reentry_clear_count < context.high_reentry_confirm_weeks:
        return {
            "target": current_target,
            "mode": "waiting",
            "clear_count": context.high_reentry_clear_count,
        }

    return {
        "target": desired_target,
        "mode": "confirmed",
        "clear_count": context.high_reentry_clear_count,
    }
